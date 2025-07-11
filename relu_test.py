import pyopencl as cl
import numpy as np
import time

# Input vector size
N = 16  # You can try 256, 1024, etc.

# Generate sample input data (some negative, some positive)
input_array = np.random.uniform(-10, 10, N).astype(np.float32)
output_array = np.empty_like(input_array)

# Setup OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

# Buffers
input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_array)
output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, input_array.nbytes)

# Load kernel
with open("kernels/relu.cl", "r") as f:
    kernel_src = f.read()

try:
    program = cl.Program(ctx, kernel_src).build()
except Exception as e:
    print("❌ Kernel Compilation Error:")
    print(e)
    exit()

program = cl.Program(ctx, kernel_src).build()

# Run kernel
start_gpu = time.perf_counter()
program.relu(queue, (N,), None, input_buf, output_buf, np.int32(N))
cl.enqueue_copy(queue, output_array, output_buf)
queue.finish()
end_gpu = time.perf_counter()

# Run CPU ReLU
start_cpu = time.perf_counter()
relu_cpu = np.maximum(input_array, 0)
end_cpu = time.perf_counter()

# Output
print("Input Array:")
print(input_array)

print("\nGPU ReLU Output:")
print(output_array)

print("\nCPU ReLU Output:")
print(relu_cpu)

# Verify
if np.allclose(output_array, relu_cpu):
    print("\n✅ ReLU Output Verified: GPU and CPU match!")
else:
    print("\n❌ Mismatch in results!")

# Timing
print(f"\n⏱ GPU Time: {end_gpu - start_gpu:.6f} seconds")
print(f"⏱ CPU Time: {end_cpu - start_cpu:.6f} seconds")