import pyopencl as cl
import numpy as np
import time

# ---------- Step 1: Setup ----------
N = 1024  # Matrix size (NxN)
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.empty((N, N)).astype(np.float32)

# ---------- Step 2: Setup OpenCL ----------
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

# ---------- Step 3: Load Kernel ----------
with open("kernels/matmul.cl", "r") as f:
    kernel_source = f.read()

program = cl.Program(ctx, kernel_source).build()

# ---------- Step 4: Execute Kernel (GPU) ----------
global_size = (N, N)

start_gpu = time.time()
program.mat_mul(queue, global_size, None, A_buf, B_buf, C_buf, np.int32(N))
cl.enqueue_copy(queue, C, C_buf)
queue.finish()
end_gpu = time.time()

# ---------- Step 5: CPU Matrix Multiply ----------
start_cpu = time.time()
C_cpu = np.dot(A, B)
end_cpu = time.time()

# ---------- Step 6: Display Results ----------
print("Matrix A:")
print(A)

print("\nMatrix B:")
print(B)

print("\nGPU Result Matrix C (A x B):")
print(C)

print("\nCPU Result Matrix C (A x B):")
print(C_cpu)

# Check if they match
if np.allclose(C, C_cpu, atol=1e-5):
    print("\n✅ Result Verified: GPU and CPU outputs match!")
else:
    print("\n❌ Warning: GPU and CPU results do not match!")

# Execution time
print(f"\n⏱ GPU Execution Time: {end_gpu - start_gpu:.6f} seconds")
print(f"⏱ CPU Execution Time: {end_cpu - start_cpu:.6f} seconds")