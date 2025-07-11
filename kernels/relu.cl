__kernel void relu(
    __global float* input,
    __global float* output,
    const int size)
{
    int id = get_global_id(0);
    if (id < size) {
        float val = input[id];
        output[id] = val > 0 ? val : 0.0f;
    }
}