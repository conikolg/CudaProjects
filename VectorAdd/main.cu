#include <cstdlib>
#include <iostream>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void printArray(int *arr, int len);
__global__ void vectorAdd(int *a, int *b, int *c);

int main()
{
    std::cout << "Running VectorAdd program..." << std::endl;

    // Generate cpu vectors
    const int LEN = 1024;
    const int VECTOR_SIZE = LEN * sizeof(int);
    int *a = (int *)malloc(VECTOR_SIZE);
    int *b = (int *)malloc(VECTOR_SIZE);
    int *c = (int *)malloc(VECTOR_SIZE);

    // Fill a and b
    srand(time(0));
    for (int i = 0; i < LEN; i++)
    {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }
    printArray(a, LEN);
    printArray(b, LEN);

    // Create gpu vectors
    int *aGpu=0, *bGpu=0, *cGpu=0;
    cudaMalloc(&aGpu, VECTOR_SIZE);
    cudaMalloc(&bGpu, VECTOR_SIZE);
    cudaMalloc(&cGpu, VECTOR_SIZE);
    cudaMemcpy(aGpu, a, VECTOR_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(bGpu, b, VECTOR_SIZE, cudaMemcpyHostToDevice);

    // Perform computation
    vectorAdd<<<1, LEN>>>(aGpu, bGpu, cGpu);

    // Get results back from GPU
    cudaMemcpy(c, cGpu, VECTOR_SIZE, cudaMemcpyDeviceToHost);
    printArray(c, LEN);

    // Free memory
    free(a);
    free(b);
    free(c);
    cudaFree(aGpu);
    cudaFree(bGpu);
    cudaFree(cGpu);
    return 0;
}

/**
 * Prints the contents of an array, with a max display of 10 items.
*/
void printArray(int *arr, int len)
{
    len = len > 10 ? 10 : len;
    for (int i = 0; i < len; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

/**
 * Performs an element-wise add of two vectors, storing the result in a third vector.
*/
__global__ void vectorAdd(int *a, int *b, int *c) {
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx];
}
