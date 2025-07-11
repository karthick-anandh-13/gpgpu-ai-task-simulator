import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import time

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

sizes = [32, 64, 128, 256]
gpu_times = []
cpu_times = []

with open("kernels/matmul.cl", "r") as f:
    kernel_source = f.read()
program = cl.Program(ctx, kernel_source).build()

for N in sizes:
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.empty((N, N)).astype(np.float32)

    A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

    global_size = (N, N)

    # GPU timing
    start_gpu = time.perf_counter()
    program.mat_mul(queue, global_size, None, A_buf, B_buf, C_buf, np.int32(N))
    cl.enqueue_copy(queue, C, C_buf)
    queue.finish()
    end_gpu = time.perf_counter()
    gpu_times.append(end_gpu - start_gpu)

    # CPU timing
    start_cpu = time.perf_counter()
    C_cpu = np.dot(A, B)
    end_cpu = time.perf_counter()
    cpu_times.append(end_cpu - start_cpu)

# Plotting
plt.plot(sizes, gpu_times, label='GPU Time', marker='o')
plt.plot(sizes, cpu_times, label='CPU Time', marker='s')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Execution Time (seconds)')
plt.title('GPU vs CPU Matrix Multiply Performance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Keeps window open in some environments
input("Press Enter to exit...")