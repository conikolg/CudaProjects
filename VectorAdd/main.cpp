#include <cstdlib>
#include <iostream>
#include <time.h>

void printArray(int *arr, int len);
void vectorAdd(int *a, int *b, int *c, int len);

int main()
{
    std::cout << "Running VectorAdd program..." << std::endl;

    // Generate a and b vector arrays
    const int LEN = 1024;
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
    vectorAdd(a, b, c, LEN);
    printArray(c, LEN);

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
