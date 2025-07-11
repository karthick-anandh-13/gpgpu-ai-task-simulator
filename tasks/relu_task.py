import pyopencl as cl
import numpy as np
import time

def run_relu(input_file=None, output_file=None):
    if input_file is None or output_file is None:
        print("❌ Input or output file not provided.")
        return

    try:
        input_array = np.loadtxt(input_file, delimiter=",").astype(np.float32)
    except Exception as e:
        print(f"❌ Failed to load '{input_file}': {e}")
        return

    N = input_array.size
    output_array = np.empty_like(input_array)

    try:
        platform = cl.get_platforms()[0]
        device = platform.get_devices()[0]
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_array)
        output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output_array.nbytes)

        with open("kernels/relu.cl", "r") as f:
            kernel_src = f.read()

        program = cl.Program(ctx, kernel_src).build()
    except Exception as e:
        print(f"❌ OpenCL Init/Build error: {e}")
        return

    start_gpu = time.perf_counter()
    program.relu(queue, (N,), None, input_buf, output_buf, np.int32(N))
    cl.enqueue_copy(queue, output_array, output_buf)
    queue.finish()
    end_gpu = time.perf_counter()

    try:
        np.savetxt(output_file, output_array, delimiter=",", fmt="%.5f")
        print(f"💾 Output saved to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save output: {e}")

    print(f"⏱ GPU Time: {end_gpu - start_gpu:.6f}s")
