__kernel void tanh_activation(__global float* input,
                              __global float* output,
                              int N)
{
    int i = get_global_id(0);
    if (i < N) {
        float x = input[i];
        output[i] = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }
}