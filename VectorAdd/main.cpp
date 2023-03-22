#include <cstdlib>
#include <iostream>
#include <time.h>
#include <chrono>

#define LEN 100000000

void printArray(int *arr, int len);
void vectorAdd(int *a, int *b, int *c, int len);

int main()
{
    std::cout << "Running CPU VectorAdd program..." << std::endl;

    // Generate a and b vector arrays
    int *a = (int *)malloc(LEN * sizeof(int));
    int *b = (int *)malloc(LEN * sizeof(int));
    srand(time(0));
    for (int i = 0; i < LEN; i++)
    {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    // Display them in console
    printArray(a, LEN);
    printArray(b, LEN);

    // Allocate vector c
    int *c = (int *)malloc(LEN * sizeof(int));

    // Perform computation
    auto startTime = std::chrono::high_resolution_clock::now();
    vectorAdd(a, b, c, LEN);
    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedTimeMs = std::chrono::duration<double, std::milli>(endTime-startTime).count();
    
    // Display results
    printArray(c, LEN);
    std::cout << "Compute time: " << elapsedTimeMs << "ms" << std::endl;

    // Free memory
    free(a);
    free(b);
    free(c);
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
void vectorAdd(int *a, int *b, int *c, int len)
{
    for (int i = 0; i < len; i++)
        c[i] = a[i] + b[i];
}
