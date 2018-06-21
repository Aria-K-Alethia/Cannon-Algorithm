# Cannon-Algorithm
Parallel Cannon Algorithm for matrix multiplication using MPI.

## Environment
* C
* MPI

## Introduction
Cannon Algorithm is a parallel algorithm fro matrix multiplication, ideally, the time complexity is O(n).

## I/O mode
I implemented two I/O modes to read&write the matrix to disk.  
First is the standard POSIX I/O, this method is simple, but reduces performance significantly when the matrix size is huge.  
Second is MPI I/O, this method uses MPI I/O to read&write in a parallel way, which is time-efficient when you have huge matrix.

## Parallel Matrix Generation
When the matrix become huge, generate a matrix by a single machine would deplete the limited memory.  
So I implemented a parallel matrix generation code using MPI I/O. By using this code, you can easily generate 100k matrix.

## Usage
For matrix generation:  

    ./generate row col outfile
	
**NOTE**:  currenly you should use 81 processes to run this code  

For cannon algorithm:  

	./cannon filea fileb filec

## Licence
This code is under MIT licence.
