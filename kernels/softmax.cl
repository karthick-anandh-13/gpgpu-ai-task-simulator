__kernel void softmax(__global float* input,
                      __global float* output,
                      int N)
{
    int i = get_global_id(0);
    float max_val = input[0];
    for (int j = 1; j < N; j++) {
        if (input[j] > max_val)
            max_val = input[j];
    }

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        sum += exp(input[j] - max_val);
    }

    if (i < N) {
        output[i] = exp(input[i] - max_val) / sum;
    }
}