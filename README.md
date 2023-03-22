# CudaProjects

This repository contains Cuda programs I'm developing to learn how to write
code that is suitable for parallelism on a GPU.

## Compiling

Each project will have a `Makefile` provided to facilitate compilation.

To compile the normal C++ code, simply run `make cpu`. You need `g++` in your `PATH`.
To compile Cuda C++ code, run `make gpu` instead. You'll need `nvcc` and `cl` in your `PATH`.
To compile both, run `make` or `make all`.

You can then run `./cpu.exe` and/or `./gpu.exe` after successful compilation
