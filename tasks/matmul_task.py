import pyopencl as cl
import numpy as np
import time

def run_matmul(file_a=None, file_b=None, output_file=None):
    if file_a is None or file_b is None or output_file is None:
        print("❌ Missing matrix files.")
        return

    try:
        A = np.loadtxt(file_a, delimiter=",").astype(np.float32)
        B = np.loadtxt(file_b, delimiter=",").astype(np.float32)
    except Exception as e:
        print(f"❌ Failed to load input matrices: {e}")
        return

    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        print("❌ Matrix shapes are incompatible for multiplication.")
        return

    output_array = np.empty((M, N), dtype=np.float32)

    try:
        platform = cl.get_platforms()[0]
        device = platform.get_devices()[0]
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output_array.nbytes)

        with open("kernels/matmul.cl", "r") as f:
            kernel_src = f.read()

        program = cl.Program(ctx, kernel_src).build()
    except Exception as e:
        print(f"❌ OpenCL error: {e}")
        return

    start_gpu = time.perf_counter()
    program.matmul(queue, (M, N), None, A_buf, B_buf, C_buf,
                   np.int32(M), np.int32(N), np.int32(K))
    cl.enqueue_copy(queue, output_array, C_buf)
    queue.finish()
    end_gpu = time.perf_counter()

    try:
        np.savetxt(output_file, output_array, delimiter=",", fmt="%.5f")
        print(f"💾 Matrix multiplication result saved to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save result: {e}")

    print(f"⏱ GPU Time: {end_gpu - start_gpu:.6f}s")
