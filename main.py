import argparse
import re
import os
import time
import numpy as np
import math
from collections import OrderedDict
import torch
from utils import random_seed, create_result_dir, Logger, TableLogger, AverageMeter
from attack import AttackPGD
from adamw import AdamW
from model.model import Model, set_eps, get_eps
from model.norm_dist import set_p_norm, get_p_norm

parser = argparse.ArgumentParser(description='Adversarial Robustness')
parser.add_argument('--dataset', default='MNIST', type=str)
parser.add_argument('--model', default='MLPFeature(depth=4,width=4)', type=str) #mlp,conv
parser.add_argument('--predictor-hidden-size', default=512, type=int) # 0 means not to use linear predictor
parser.add_argument('--loss', default='cross_entropy', type=str) #cross_entropy, hinge

parser.add_argument('--p-start', default=8.0, type=float)
parser.add_argument('--p-end', default=1000.0, type=float)
parser.add_argument('--kappa', default=1.0, type=float)
parser.add_argument('--epochs', default='0,50,50,350,400', type=str) # epoch1-epoch3: inc eps; epoch2-epoch4: inc p

parser.add_argument('--eps-train', default=None, type=float)
parser.add_argument('--eps-test', default=None, type=float)

parser.add_argument('-b', '--batch-size', default=256, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.99, type=float)
parser.add_argument('--epsilon', default=1e-10, type=float)
parser.add_argument('--wd', default=0.0, type=float)

parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--checkpoint', default=None, type=str)

parser.add_argument('--gpu', default=-1, type=int, help='GPU id to use')
parser.add_argument('--dist-url', default='tcp://localhost:23456')
parser.add_argument('--world-size', default=1)
parser.add_argument('--rank', default=0)

parser.add_argument('-p', '--print-freq', default=50, type=int, metavar='N', help='print frequency')
parser.add_argument('--result-dir', default='result/', type=str)
parser.add_argument('--filter-name', default='', type=str)
parser.add_argument('--seed', default=2020, type=int)
parser.add_argument('--visualize', action='store_true')

def cal_acc(outputs, targets):
    predicted = torch.max(outputs.data, 1)[1]
    return (predicted == targets).float().mean()

def parallel_reduce(*argv):
    tensor = torch.FloatTensor(argv).cuda()
    torch.distributed.all_reduce(tensor)
    ret = tensor.cpu() / torch.distributed.get_world_size()
    return ret.tolist()

def train(net, loss_fun, epoch, trainloader, optimizer, schedule, logger, train_logger, gpu, parallel, print_freq):
    if logger is not None:
        logger.print('Epoch %d training start' % (epoch))
    net.train()
    batch_time, data_time, losses, accs = [AverageMeter() for _ in range(4)]
    start = time.time()
    train_loader_len = len(trainloader)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        schedule(epoch, batch_idx)
        data_time.update(time.time() - start)
        inputs = inputs.cuda(gpu, non_blocking=True)
        targets = targets.cuda(gpu, non_blocking=True)
        outputs, worse_outputs = net(inputs, targets=targets)
        loss = loss_fun(outputs, worse_outputs, targets)
        with torch.no_grad():
            losses.update(loss.data.item(), targets.size(0))
            accs.update(cal_acc(outputs.data, targets).mean().item(), targets.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)
        if (batch_idx + 1) % print_freq == 0 and logger is not None:
            logger.print('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'lr {lr:.4f}\tp {p:.2f}\teps {eps:.4f}\tkappa{kappa:.4f}\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Acc {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                         epoch, batch_idx + 1, train_loader_len, batch_time=batch_time,
                         lr=optimizer.param_groups[0]['lr'],
                         p=get_p_norm(net), eps=get_eps(net), kappa=loss_fun.kappa,
                         loss=losses, acc=accs))
        start = time.time()

    loss, acc = losses.avg, accs.avg
    if parallel:
        loss, acc = parallel_reduce(losses.avg, accs.avg)
    if train_logger is not None:
        train_logger.log({'epoch': epoch, 'loss': loss, 'acc': acc})
    if logger is not None:
        logger.print('Epoch {0}:  train loss {loss:.4f}   acc {acc:.4f}'
                     '   lr {lr:.4f}   p {p:.2f}   eps {eps:.4f}   kappa {kappa:.4f}'.format(
                     epoch, loss=loss, acc=acc, lr=optimizer.param_groups[0]['lr'],
                     p=get_p_norm(net), eps=get_eps(net), kappa=loss_fun.kappa))
    return loss, acc

@torch.no_grad()
def test(net, loss_fun, epoch, testloader, logger, test_logger, gpu, parallel, print_freq):
    net.eval()
    batch_time, data_time, losses, accs = [AverageMeter() for _ in range(4)]
    start = time.time()
    test_loader_len = len(testloader)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.cuda(gpu, non_blocking=True)
        targets = targets.cuda(gpu, non_blocking=True)
        outputs = net(inputs)
        loss = loss_fun(outputs, targets)
        losses.update(loss.mean().item(), targets.size(0))
        accs.update(cal_acc(outputs, targets).item(), targets.size(0))
        batch_time.update(time.time() - start)
        start = time.time()
        if (batch_idx + 1) % print_freq == 0 and logger is not None:
            logger.print('Test: [{0}/{1}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Acc {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                         batch_idx + 1, test_loader_len, batch_time=batch_time, loss=losses, acc=accs))

    loss, acc = losses.avg, accs.avg
    if parallel:
        loss, acc = parallel_reduce(losses.avg, accs.avg)
    if test_logger is not None:
        test_logger.log({'epoch': epoch, 'loss': loss, 'acc': acc})
    if logger is not None:
        logger.print('Epoch %d:  '%epoch + 'test loss  ' + f'{loss:.4f}' + '   acc ' + f'{acc:.4f}')
    return loss, acc

def gen_adv_examples(model, attacker, test_loader, gpu, parallel, logger, fast=False):
    model.eval()
    correct = 0
    tot_num = 0
    size = len(test_loader)

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.cuda(gpu, non_blocking=True)
        targets = targets.cuda(gpu, non_blocking=True)
        result = torch.ones(targets.size(0), dtype=torch.bool, device=targets.device)
        for i in range(1):
            perturb = attacker.find(inputs, targets)
            with torch.no_grad():
                outputs = model(perturb)
                predicted = torch.max(outputs.data, 1)[1]
                result &= (predicted == targets)
        correct += result.float().sum().item()
        tot_num += inputs.size(0)
        if fast and batch_idx * 10 >= size: break

    acc = correct / tot_num * 100
    if parallel:
        acc, = parallel_reduce(acc)
    if logger is not None:
        logger.print('adversarial attack acc ' + f'{acc:.4f}')
    return acc

@torch.no_grad()
def certified_test(net, eps, up, down, epoch, testloader, logger, gpu, parallel):
    save_p = get_p_norm(net)
    save_eps = get_eps(net)
    set_eps(net, eps)
    set_p_norm(net, float('inf'))
    net.eval()
    outputs = []
    labels = []

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.cuda(gpu, non_blocking=True)
        lower = torch.max(inputs - eps, down)
        upper = torch.min(inputs + eps, up)
        targets = targets.cuda(gpu, non_blocking=True)
        # outputs.append(net(inputs, targets=targets)[1])
        outputs.append(net(inputs, lower=lower, upper=upper, targets=targets)[1])
        labels.append(targets)
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    res = (outputs.max(dim=1)[1] == labels).float().mean().item()

    if parallel:
        res, = parallel_reduce(res)
    if logger is not None:
        logger.print('Epoch %d: '%epoch + ' certify acc ' + f'{res:.4f}')
    set_p_norm(net, save_p)
    set_eps(net, save_eps)
    return res

def parse_function_call(s):
    s = re.split(r'[()]', s)
    if len(s) == 1:
        return s[0], {}
    name, params, _ = s
    params = re.split(r',\s*', params)
    params = dict([p.split('=') for p in params])
    for key, value in params.items():
        try:
            params[key] = int(params[key])
        except ValueError:
            try:
                params[key] = float(params[key])
            except ValueError:
                pass
    return name, params

def create_schedule(args, batch_per_epoch, model, loss, optimizer):
    epoch0, epoch1, epoch2, epoch3, tot_epoch = args.epochs
    speed = math.log(args.p_end / args.p_start)
    def num_batches(epoch, minibatch=0):
        return epoch * batch_per_epoch + minibatch

    def schedule(epoch, minibatch):
        ratio = max(num_batches(epoch - epoch1, minibatch) / num_batches(tot_epoch - epoch1), 0)
        lr_now = 0.5 * args.lr * (1 + math.cos((ratio * math.pi)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now

        ratio = min(max(num_batches(epoch - epoch1, minibatch) / num_batches(epoch3 - epoch1), 0), 1)
        if ratio >= 1:
            p_norm = float('inf')
        else:
            p_norm = args.p_start * math.exp(speed * ratio)
        set_p_norm(model, p_norm)

        if epoch2 > 0:
            ratio = min(max(num_batches(epoch - epoch0, minibatch) / num_batches(epoch2), 0), 1)
        else:
            ratio = 1.0
        set_eps(model, args.eps_train * ratio)
        loss.kappa = args.kappa

    return schedule

import torch.nn.functional as F
def cross_entropy():
    return F.cross_entropy

# The hinge loss function is a combination of max_hinge_loss and average_hinge_loss.
def hinge(mix=0.75):
    def loss_fun(outputs, targets):
        return mix * outputs.max(dim=1)[0].clamp(min=0).mean() + (1 - mix) * outputs.clamp(min=0).mean()
    return loss_fun

class Loss():
    def __init__(self, loss, kappa):
        self.loss = loss
        self.kappa = kappa
    def __call__(self, *args):
        margin_output = args[0] - torch.gather(args[0], dim=1, index=args[-1].view(-1, 1))
        if len(args) == 2:
            return self.loss(margin_output, args[-1])
        # args[1] which corresponds to worse_outputs, is already a margin vector.
        return self.kappa * self.loss(args[1], args[-1]) + (1 - self.kappa) * self.loss(margin_output, args[-1])

def main_worker(gpu, parallel, args, result_dir):
    if parallel:
        args.rank = args.rank + gpu
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)
    torch.backends.cudnn.benchmark = True
    random_seed(args.seed + args.rank) # make data aug different for different processes
    torch.cuda.set_device(gpu)

    assert args.batch_size % args.world_size == 0
    from dataset import load_data, get_statistics, default_eps, input_dim
    train_loader, test_loader = load_data(args.dataset, 'data/', args.batch_size // args.world_size, parallel,
                                          augmentation=True, classes=None)
    mean, std = get_statistics(args.dataset)
    num_classes = len(train_loader.dataset.classes)

    from model.bound_module import Predictor, BoundFinalIdentity
    from model.mlp import MLPFeature, MLP
    from model.conv import ConvFeature, Conv
    model_name, params = parse_function_call(args.model)
    if args.predictor_hidden_size > 0:
        model = locals()[model_name](input_dim=input_dim[args.dataset], **params)
        predictor = Predictor(model.out_features, args.predictor_hidden_size, num_classes)
    else:
        model = locals()[model_name](input_dim=input_dim[args.dataset], num_classes=num_classes, **params)
        predictor = BoundFinalIdentity()
    model = Model(model, predictor, eps=0)
    model = model.cuda(gpu)
    if parallel:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    loss_name, params = parse_function_call(args.loss)
    loss = Loss(globals()[loss_name](**params), args.kappa)

    output_flag = not parallel or gpu == 0
    if output_flag:
        logger = Logger(os.path.join(result_dir, 'log.txt'))
        for arg in vars(args):
            logger.print(arg, '=', getattr(args, arg))
        logger.print(train_loader.dataset.transform)
        logger.print(model)
        logger.print('number of params: ', sum([p.numel() for p in model.parameters()]))
        logger.print('Using loss', loss)
        train_logger = TableLogger(os.path.join(result_dir, 'train.log'), ['epoch', 'loss', 'acc'])
        test_logger = TableLogger(os.path.join(result_dir, 'test.log'), ['epoch', 'loss', 'acc'])
    else:
        logger = train_logger = test_logger = None

    optimizer = AdamW(model, lr=args.lr, weight_decay=args.wd, betas=(args.beta1,args.beta2), eps=args.epsilon)

    if args.checkpoint:
        assert os.path.isfile(args.checkpoint)
        if parallel:
            torch.distributed.barrier()
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda(gpu))
        state_dict = checkpoint['state_dict']
        if next(iter(state_dict))[0:7] == 'module.' and not parallel:
            new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
            state_dict = new_state_dict
        elif next(iter(state_dict))[0:7] != 'module.' and parallel:
            new_state_dict = OrderedDict([('module.' + k, v) for k, v in state_dict.items()])
            state_dict = new_state_dict
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded '{}'".format(args.checkpoint))
        if parallel:
            torch.distributed.barrier()

    if args.eps_test is None:
        args.eps_test = default_eps[args.dataset]
    if args.eps_train is None:
        args.eps_train = args.eps_test
    args.eps_train /= std
    args.eps_test /= std
    up = torch.FloatTensor((1 - mean) / std).view(-1, 1, 1).cuda(gpu)
    down = torch.FloatTensor((0 - mean) / std).view(-1, 1, 1).cuda(gpu)
    attacker = AttackPGD(model, args.eps_test, step_size=args.eps_test / 4, num_steps=20, up=up, down=down)
    args.epochs = [int(epoch) for epoch in args.epochs.split(',')]
    schedule = create_schedule(args, len(train_loader), model, loss, optimizer)

    if args.visualize and output_flag:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(result_dir)
    else: writer = None

    for epoch in range(args.start_epoch, args.epochs[-1]):
        if parallel:
            train_loader.sampler.set_epoch(epoch)
        train_loss, train_acc = train(model, loss, epoch, train_loader, optimizer, schedule,
                                      logger, train_logger, gpu, parallel, args.print_freq)
        test_loss, test_acc = test(model, loss, epoch, test_loader, logger, test_logger, gpu, parallel, args.print_freq)
        if writer is not None:
            writer.add_scalar('curve/p', get_p_norm(model), epoch)
            writer.add_scalar('curve/train loss', train_loss, epoch)
            writer.add_scalar('curve/test loss', test_loss, epoch)
            writer.add_scalar('curve/train acc', train_acc, epoch)
            writer.add_scalar('curve/test acc', test_acc, epoch)
        if epoch % 50 == 49:
            if logger is not None:
                logger.print('Generate adversarial examples (fast, inaccurate)')
            robust_train_acc = gen_adv_examples(model,attacker, train_loader, gpu, parallel, logger, fast=True)
            robust_test_acc = gen_adv_examples(model, attacker, test_loader, gpu, parallel, logger, fast=True)
            if writer is not None:
                writer.add_scalar('curve/robust train acc', robust_train_acc, epoch)
                writer.add_scalar('curve/robust test acc', robust_test_acc, epoch)
        if epoch % 5 == 4:
            certify_acc = certified_test(model, args.eps_test, up, down, epoch, test_loader, logger, gpu, parallel)
            if writer is not None:
                writer.add_scalar('curve/certify acc', certify_acc, epoch)
        if epoch > args.epochs[-1] - 3:
            if logger is not None:
                logger.print("Generate adversarial examples on test dataset")
            gen_adv_examples(model, attacker, test_loader, gpu, parallel, logger)
            certified_test(model, args.eps_test, up, down, epoch, test_loader, logger, gpu, parallel)

    schedule(args.epochs[-1], 0)
    if output_flag:
        logger.print("Calculate certified accuracy")
    certified_test(model, args.eps_test, up, down, args.epochs[-1], train_loader, logger, gpu, parallel)
    certified_test(model, args.eps_test, up, down, args.epochs[-1], test_loader, logger, gpu, parallel)

    # if output_flag:
    #     torch.save({
    #         'state_dict': model.state_dict(),
    #         'optimizer' : optimizer.state_dict(),
    #     }, os.path.join(result_dir, 'model.pth'))
    if writer is not None:
        writer.close()

def main(father_handle, **extra_argv):
    args = parser.parse_args()
    for key,val in extra_argv.items():
        setattr(args, key, val)
    result_dir = create_result_dir(args)
    if father_handle is not None:
        father_handle.put(result_dir)
    if args.gpu != -1:
        main_worker(args.gpu, False, args, result_dir)
    else:
        n_procs = torch.cuda.device_count()
        args.world_size *= n_procs
        args.rank *= n_procs
        torch.multiprocessing.spawn(main_worker, nprocs=n_procs, args=(True, args, result_dir))

if __name__ == '__main__':
    main(None)
