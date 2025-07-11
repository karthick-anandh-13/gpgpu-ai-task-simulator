import numpy as np
import time

def run_softmax(input_file=None, output_file=None):
    if input_file is None or output_file is None:
        print("❌ Input or output file not provided.")
        return

    try:
        input_array = np.loadtxt(input_file, delimiter=",").astype(np.float32)
    except Exception as e:
        print(f"❌ Failed to load '{input_file}': {e}")
        return

    start = time.perf_counter()
    exp_values = np.exp(input_array - np.max(input_array, axis=1, keepdims=True))
    result = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    end = time.perf_counter()

    try:
        np.savetxt(output_file, result, delimiter=",", fmt="%.5f")
        print(f"💾 Output saved to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save output: {e}")

    print(f"⏱ CPU Softmax Time: {end - start:.6f}s")
