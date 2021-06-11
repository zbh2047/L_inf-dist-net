# L_inf-dist Net: Towards Certifying L-infinity Robustness using Neural Networks with L_inf-dist Neurons  

## Introduction

This is the official code for training the L_inf-dist net, a theoretically principled neural network that inherently resists L_inf-norm perturbations. We consistently achieve state-of-the-art performance on commonly used datasets: **93.09%** certiﬁed accuracy on MNIST under eps = 0.3, **79.23%** on Fashion-MNIST under eps = 0.1, **35.42%** on CIFAR-10 under eps = 8/255 and **16.31%** on TinyImageNet under eps = 1/255.  [Our paper](  https://arxiv.org/abs/2102.05363  ) has been accepted for ICML 2021.

## Dependencies

- Pytorch 1.6.0
- Tensorboard (optional)

## Getting Started with the Code

### Installation

After cloning this repo into your computer, first run the following command to install the CUDA extension, which can speed up the training procedure considerably.

```
python setup.py install --user
```

### Reproducing SOTA Results

In this repo, we provide complete training scripts to reproduce the results on MNIST, Fashion-MNIST and CIFAR-10 datasets in our paper. These scripts are in the `command` folder. 

For example, to reproduce the results of CIFAR-10 using the $\ell_\infty$-dist Net alone, simply run

```
bash command/ell_inf_dist_net_cifar10.sh
```

To reproduce the results of CIFAR-10 using the $\ell_\infty$-dist Net+MLP, simply run

```
bash command/ell_inf_dist_net++_cifar10.sh
```

For TinyImageNet dataset, the dataset can be download from  http://cs231n.stanford.edu/tiny-imagenet-200.zip. Also you should run `tiny_imagenet.sh` to tiny the dataset.

## Advanced Training Options

### Multi-GPU Training

We also support multi-GPU training using distributed data parallel. By default the code will use all available GPUs for training. To use a single GPU, add the following parameter `--gpu GPU_ID` where `GPU_ID` is the GPU ID. You can also specify `--world-size`, `--rank` and `--dist-url` for advanced multi-GPU training.

### Saving and Loading

The model is automatically saved when the training procedure finishes. Use `--checkpoint model_file_name.pth` to load a specified model before training. You can use `--start-epoch NUM_EPOCHS` to skip training and only test the model's performance for a pretrained model, where `NUM_EPOCHS` is the number of epochs in total.

### Displaying training curves

By default the code will generate three files named `train.log`, `test.log` and `log.txt` which contain all training logs. If you want to further display training curves, you can add the parameter `--visualize` to show these curves using Tensorboard. 

## Contact

Please contact [zhangbohang@pku.edu.cn](zhangbohang@pku.edu.cn)  if you have any question on our paper or the codes. Enjoy! 

## Citation

```
@article{zhang2021certifying,
      title={Towards Certifying $\ell_\infty$ Robustness using Neural Networks with $\ell_\infty$-dist Neurons}, 
      author={Bohang Zhang and Tianle Cai and Zhou Lu and Di He and Liwei Wang},
      year={2021},
      eprint={2102.05363},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

