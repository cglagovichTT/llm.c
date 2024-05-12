# dev/metal

This directory is scratch space for developing various versions of the needed metal kernels. Each file develops a kernel, and usually multiple versions of that kernel that could have different running times and of different code or time complexity.

Philosophy:
- Limited dependencies. To begin, we allow tt-metal runtime and tt::numpy for simplicity.
- Start with FP32 everywhere for correctness
- Slow, correct kernels first, then optimize
- It should be easy to change dataformats and measure correctness vs CPU
