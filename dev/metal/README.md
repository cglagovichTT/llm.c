# dev/metal

This directory is scratch space for developing various versions of the needed metal kernels. Each file develops a kernel, and usually multiple versions of that kernel that could have different running times and of different code or time complexity.

Philosophy:
- Limited dependencies. To begin, we allow tt-metal runtime and tt::numpy for simplicity.
- Start with FP32 everywhere for correctness
- Slow, correct kernels first, then optimize
- It should be easy to change dataformats and measure correctness vs CPU


TODO:
- [ ] Layernorm Forward
    - [x] Multi-core test passes for various inputs
    - [ ] Optimize compile-time and runtime args

- [ ] Layernorm Backward
    - What are the inputs?
    - Does the op know about recompute?
    - How do we reduce weight and bias gradients when multiple cores are computing partials?
        1. Parallelize across C so there's no contention for W and B gradients
        2. Parallelize across B and T, reduce W and B gradients at the end
I haven't figured out yet how to parallelize layernorm backward, or any backward kernel. I see that Moreh
has implemented two layernorm_backward ops: one which computes grads for input, and one which computes grads for
weight and bias. I would ideally fuse these into one op, though the parallelization over input_grad and WB grad are
opposing. Either way, there must be communication/reduction. Whichever scheme I pick here I will likely use in every
backward op. 

Let's commit to something up front: it is worth it to test multiple strategies for gradient accumulation in backwards ops, 
so we don't need to commit to any one strategy yet.

One option: parallelize over input B and T just like the forwards kernel. This may generally be a good strategy for
backwards kernels since weight grads are on average smaller than input grads. 
Given B,T parallelization over input/output grads (input/output shapes should be similar, only differing in C dim for some matmuls),
reduce weight and bias grads at the end with some chosen parallelization scheme. After reduction of current weight and bias grads,
certain cores are responsible for accumulating portions of the gradient into gradient dram buffers.

First try at layernorm backwards:
- Single core implementation, no reduction. Better to get functionality first then worry about performance after!

Second try at layernorm backwards:
- dinp: [B, T, C] parallelized on B and T
- each core computes local dweight and dbias: [C]
- (AllReduce) each row reduces dweight and dbias [C] to leftmost column
- (ReduceScatter) leftmost column reduces [C/num_rows] on each core
- (Gradient accumulation) leftmost column cores read gradient buffer and add local reduction, then write back.

- [ ] Implement program caching
    - Very involved, requires reverse-engineering op infra's program caching