#include "example.h"

int add(int i, int j) {
    return i + j;
}

torch::Tensor add_tensors(const torch::Tensor& a, const torch::Tensor& b) {
    // 检查张量类型
    std::cout << "a type: " << a.scalar_type() << ", b type: " << b.scalar_type() << std::endl;
    // 确保所有张量都转换为浮点类型
    auto a_converted = a.scalar_type() == at::kFloat ? a : a.to(at::kFloat);
    auto b_converted = b.scalar_type() == at::kFloat ? b : b.to(at::kFloat);
    return a_converted + b_converted;
}