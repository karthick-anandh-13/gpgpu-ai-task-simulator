__kernel void sigmoid(
    __global float* input,
    __global float* output,
    const int size)
{
    int id = get_global_id(0);
    if (id < size) {
        float x = input[id];
        output[id] = 1.0f / (1.0f + exp(-x));
    }
}