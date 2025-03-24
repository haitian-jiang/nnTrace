#include "bitwise_rewrite.hpp"
#include "bitwise_rewrite_kernel.hpp"

void rewrite_low_bits_bf16(  // a wrapper for the template function
    torch::Tensor activation,
    torch::Tensor seeds,
    int N 
) {
    assert(N > 0 && N < 7); // N must be in (0, 7)
    assert(activation.is_cuda());
    assert(seeds.is_cuda());
    assert(seeds.numel() == activation.numel());
    assert(seeds.is_contiguous());
    assert(activation.is_contiguous());
    assert(activation.dtype() == torch::kBFloat16);
    assert(seeds.dtype() == torch::kInt64);
    // Call the appropriate template function based on N
    switch (N) {
        case 1: _rewrite_low_bits_bf16<1>(activation, seeds); break;
        case 2: _rewrite_low_bits_bf16<2>(activation, seeds); break;
        case 3: _rewrite_low_bits_bf16<3>(activation, seeds); break;
        case 4: _rewrite_low_bits_bf16<4>(activation, seeds); break;
        case 5: _rewrite_low_bits_bf16<5>(activation, seeds); break;
        case 6: _rewrite_low_bits_bf16<6>(activation, seeds); break;
        default: throw std::invalid_argument("N must be in (0, 7)");
    }
}
