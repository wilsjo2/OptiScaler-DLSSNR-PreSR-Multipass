// Small helpers for the kernel lab and the NVFP4 hybrid.
struct FillParams { int* data; int count; int value; };

// Sets `count` 32-bit words to `value` (the runtime's own flag reset stores -1 the same way).
extern "C" __global__ void nr_fill_i32(FillParams p)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < p.count) p.data[index] = p.value;
}
