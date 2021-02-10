#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void norm_dist_forward_cuda(const float* input, const float* weight,
                            int B, int CO, int CI, int G, int HW, float* output, float p);
void norm_dist_backward_input_cuda(const float* grad_output, const float* input, const float* weight, const float* output,
                                   int B, int CO, int CI, int G, int HW, float* grad_input, float p);
void norm_dist_backward_weight_cuda(const float* grad_output, const float* input, const float* weight, const float* output,
                                    int B, int CO, int CI, int G, int HW, float* grad_weight, float p);
void inf_dist_forward_cuda(const float* input, const float* weight,
                           int B, int CO, int CI, int G, int HW, float* output, int* pos);
void inf_dist_bound_forward_cuda(const float* input_lower, const float* input_upper, const float* weight,
                                 int B, int CO, int CI, int G, int HW, float* output_lower, float* output_upper);
void inf_dist_backward_input_cuda(const float* grad_output, const int* pos,
                                  int B, int CO, int CI, int G, int HW, float* grad_input);
void inf_dist_backward_weight_cuda(const float* grad_output, const int* pos,
                                   int B, int CO, int CI, int G, int HW, float* grad_weight);

void norm_dist_forward(torch::Tensor& input, torch::Tensor& weight, torch::Tensor& output, int G, float p) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    int B = input.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    norm_dist_forward_cuda(input.data_ptr<float>(), weight.data_ptr<float>(), B, CO, CI, G, HW, output.data_ptr<float>(), p);
}

void norm_dist_backward_input(torch::Tensor& grad_output, torch::Tensor& input, torch::Tensor& weight,
                              torch::Tensor& output, torch::Tensor& grad_input, int G, float p) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    CHECK_INPUT(grad_input);
    int B = grad_output.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    norm_dist_backward_input_cuda(grad_output.data_ptr<float>(), input.data_ptr<float>(), weight.data_ptr<float>(),
                                  output.data_ptr<float>(), B, CO, CI, G, HW, grad_input.data_ptr<float>(), p);
}

void norm_dist_backward_weight(torch::Tensor& grad_output, torch::Tensor& input, torch::Tensor& weight,
                               torch::Tensor& output, torch::Tensor& grad_weight, int G, float p) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    CHECK_INPUT(grad_weight);
    int B = grad_output.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    norm_dist_backward_weight_cuda(grad_output.data_ptr<float>(), input.data_ptr<float>(), weight.data_ptr<float>(),
                                   output.data_ptr<float>(), B, CO, CI, G, HW, grad_weight.data_ptr<float>(), p);
}

void inf_dist_forward(torch::Tensor& input, torch::Tensor& weight, torch::Tensor& output, torch::Tensor& pos, int G) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    CHECK_INPUT(pos);
    int B = input.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    inf_dist_forward_cuda(input.data_ptr<float>(), weight.data_ptr<float>(), B, CO, CI, G, HW,
                          output.data_ptr<float>(), pos.data_ptr<int>());
}

void inf_dist_bound_forward(torch::Tensor& input_lower, torch::Tensor& input_upper, torch::Tensor& weight,
                            torch::Tensor& output_lower, torch::Tensor& output_upper, int G) {
    CHECK_INPUT(input_lower);
    CHECK_INPUT(input_upper);
    CHECK_INPUT(weight);
    CHECK_INPUT(output_lower);
    CHECK_INPUT(output_upper);
    int B = input_lower.size(0), CO = output_lower.size(1), CI = input_lower.size(1), HW = input_lower.size(2);
    inf_dist_bound_forward_cuda(input_lower.data_ptr<float>(), input_upper.data_ptr<float>(), weight.data_ptr<float>(),
                                B, CO, CI, G, HW, output_lower.data_ptr<float>(), output_upper.data_ptr<float>());
}

void inf_dist_backward_input(torch::Tensor& grad_output, torch::Tensor& pos, torch::Tensor& grad_input, int G) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(pos);
    CHECK_INPUT(grad_input);
    int B = grad_output.size(0), CO = grad_output.size(1), CI = grad_input.size(1), HW = grad_input.size(2);
    inf_dist_backward_input_cuda(grad_output.data_ptr<float>(), pos.data_ptr<int>(), B, CO, CI, G, HW, grad_input.data_ptr<float>());
}

void inf_dist_backward_weight(torch::Tensor& grad_output, torch::Tensor& pos, torch::Tensor& grad_weight, int G) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(pos);
    CHECK_INPUT(grad_weight);
    int B = grad_output.size(0), CO = grad_output.size(1), CI = grad_weight.size(1) * G, HW = grad_output.size(2);
    inf_dist_backward_weight_cuda(grad_output.data_ptr<float>(), pos.data_ptr<int>(), B, CO, CI, G, HW, grad_weight.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("norm_dist_forward", &norm_dist_forward);
    m.def("norm_dist_backward_input", &norm_dist_backward_input);
    m.def("norm_dist_backward_weight", &norm_dist_backward_weight);
    m.def("inf_dist_forward", &inf_dist_forward);
    m.def("inf_dist_bound_forward", &inf_dist_bound_forward);
    m.def("inf_dist_backward_input", &inf_dist_backward_input);
    m.def("inf_dist_backward_weight", &inf_dist_backward_weight);
}