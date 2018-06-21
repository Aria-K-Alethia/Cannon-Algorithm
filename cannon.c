/*
	Copyright (c) 2018 Aria-K-Alethia@github.com
	Cannon Algorithm for Matrix Multiplication
	

	Usage:
		./cannon mode filea fileb filec
	params:
		mode: must be parallel or serial, parallel mode would use MPI IO to read and write, serial mode would only use naive IO
		filea: matrix A file path
		fileb: matrix B file path
		filec: output file path

	THIS CODE IS UNDER MIT LICENCE.
*/

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"
#include "mpi.h"
#include "time.h"
// for debugging
#define DEBUG 0
// mode coding
#define SERIAL 0
#define PARALLEL 1
// parameter number
#define PARAMETER_NUMBER 5
// para position
#define MODE  1
#define FILEA 2
#define FILEB 3
#define FILEC 4
// function definitions, for detail info, see function code
FILE* safe_open(char* filename, const char* mode);
int setup(int argc, char* argv[], char** fstreama, char** fstreamb,int dim[], int mode);
void scatter_matrix(double* fstream, int row, int col, double* m, int root, int myrank, int buf_size, int isa);
void parallel_read(int row, int col, double* m, int root, int myrank, char* filename, int isa);
void cannon(double* A, double* bufA, int bufA_size, double* B, double* bufB, int bufB_size, double* C, int bufC_size, int dim[], int root, int myrank);
void gather_matrix(double* fstreamc, int dim[], double* C, int bufC_size, int root, int myrank, int numprocs);
double* read_matrix(FILE* file, int* dim);
int check(char* a, char* b, char* c);
int maxlength(int n, int root);
int block2procs(int i, int j, int root);
void loadblock(double* fstream, double* m, int i ,int j, int row, int col, int root);
void swap(double** a, double** b);
void time_and_plus(double* A, double* B, double* C);
void shiftblock(double* A, double* bufA, int bufA_size, double* B, double* bufB, int bufB_size, int myrank, int root, int dim[]);
void write_matrix(double* fstreamc, int dim[], double* C, int root, int myrank);
void print_matrix(double* m, int row, int col);
void print_matrix2(double* m, int row, int col);
double* array(double* m);
void parallel_write(char* filename, int dim[], double* C, int root, int myrank);

int main(int argc, char *argv[])
{
	// A: n1 x n2 ; B: n2 x n3 ; C: n1 x n3
	int n1,n2,n3;
	// rank and numprocs
	int myrank, numprocs;
	// matrix buffers
	double *A=NULL, *B=NULL, *C=NULL, *bufA=NULL, *bufB=NULL;
	char *fstreama=NULL, *fstreamb=NULL, *fstreamc=NULL;
	// dim buffer
	int dim[3];
	int *dima=NULL, *dimb=NULL, *dimc=NULL;
	// time
	double elapsed_time, total_time, io_time, scatter_time;
	// output file
	FILE* outfile=NULL;
	// output file size
	int fsizec;
	// the matrix block size
	int maxrow_a, maxcol_a, maxrow_b, maxcol_b;
	// the matrix size, according to the max row and col
	int bufA_size, bufB_size, bufC_size;
	// buffer to save all matrix
	char *buf=NULL;
	int root;
	int flag;
	// mode
	int mode;
	// initialize, get myrank and numprocs
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	if(DEBUG && myrank == 0) printf("numprocs: %d\n", numprocs);
	// the processor number should be rooted, otherwise quit
	root = sqrt(numprocs);
	if(numprocs != root * root){
		MPI_Finalize();
		printf("Error: processor number must be a square!\n");
		exit(-1);
	}
	// get mode
	if(strcmp(argv[MODE], "parallel") == 0)
		mode = PARALLEL;
	else if(strcmp(argv[MODE], "serial") == 0)
		mode = SERIAL;
	else{
		MPI_Finalize();
		printf("Error: no such mode: %s, the mode must be either serial or parallel!\n", argv[MODE]);
		exit(-1);
	}
	// get A and B from file, also their dimension
	// procs 0 to do the initialization
	total_time = MPI_Wtime();
	io_time = MPI_Wtime();
	if(myrank == 0){
		// call setup function, check the status
		// if serial mode, we read all the matrix
		// if parallel mode, we only read the dimension information from the file
		if(setup(argc, argv, &fstreama, &fstreamb, dim, mode)){
			MPI_Finalize();
			printf("Error in setup\n"); 
			exit(-1);
		}
		if(DEBUG) printf("setup success\n");
	}
	io_time = MPI_Wtime() - io_time;
	// tell all procs the dim of the two matrix
	MPI_Bcast(dim, 3, MPI_INT, 0, MPI_COMM_WORLD);
	n1 = dim[0];
	n2 = dim[1];
	n3 = dim[2];
	// the number of elements in A or B must be greater than the procs number
	if(n1 < root || n2 < root || n3 < root){
		printf("The number of elements in each dimension must be greater or equal to sqrt(procs number)\n");
		printf("Please set the procs number to min(%d, %d, %d)^2\n", n1, n2, n3);
		MPI_Finalize();
		exit(-1);
	}
	
	// get the max row and col for A and B
	// we would use this value to divide the group
	// some of the block may have relatively small size
	maxrow_a = maxlength(n1, root);
	maxcol_a = maxlength(n2, root);
	maxrow_b = maxcol_a;
	maxcol_b = maxlength(n3, root);
	// get the size of each sub-matrix
	bufA_size = sizeof(int)*2 + sizeof(double)*maxrow_a*maxcol_a;
	bufB_size = sizeof(int)*2 + sizeof(double)*maxrow_b*maxcol_b;
	bufC_size = sizeof(int)*2 + sizeof(double)*maxrow_a*maxcol_b;
	// allocate the memory to save all the sub-matrix
	if(!(buf = (char*)malloc(bufA_size*2 + bufB_size*2 + bufC_size))){
		printf("Error: no memory\n");
		MPI_Finalize();
		exit(-1);
	}
	// set the correct value for each matrix pointer
	A = (double*)buf;
	bufA = (double*)(buf + bufA_size);
	B = (double*)(buf + bufA_size * 2);
	bufB = (double*)(buf + bufA_size * 2 + bufB_size);
	C = (double*)(buf + bufA_size * 2 + bufB_size * 2);
	// clear the mem in C
	memset((char*)C, 0, bufC_size);
	// if serial mode, procs 0 scatter matrix A and B to each procs
	// record scatter time
	scatter_time = MPI_Wtime();
	if(mode == SERIAL){
		scatter_matrix((double*)(fstreama+sizeof(int)*2), n1, n2, A, root, myrank, bufA_size, 1);
		scatter_matrix((double*)(fstreamb+sizeof(int)*2), n2, n3, B, root, myrank, bufB_size, 0);
	}
	// else parallel mode, each procs read its own sub-matrix
	else{
		parallel_read(n1, n2, A, root, myrank, argv[FILEA], 1);
		parallel_read(n2, n3, B, root, myrank, argv[FILEB], 0);
	}
	// get the scatter time
	scatter_time = MPI_Wtime() - scatter_time;
	if(DEBUG) printf("myrank: %d, scatter matrix done\n", myrank);
	// set the dimension of C
	((int*)C)[0] = ((int*)A)[0];
	((int*)C)[1] = ((int*)B)[1];
	// synchronize
	MPI_Barrier(MPI_COMM_WORLD);
	elapsed_time = MPI_Wtime();
	// call the cannon algorithm to compute A*B
	cannon(A, bufA, bufA_size, B, bufB, bufB_size, C, bufC_size,\
		dim, root, myrank);
	if(DEBUG) printf("myrank: %d, cannon done\n", myrank);
	MPI_Barrier(MPI_COMM_WORLD);
	elapsed_time = MPI_Wtime() - elapsed_time;
	// procs 0 gathers the outcome from other procs and output
	// get the write length
	fsizec = sizeof(int)*2 + sizeof(double)*n1*n3;
	if(myrank == 0){
		// open the file
		outfile = safe_open(argv[FILEC], "w");
		fstreamc = (char*)malloc(fsizec);
		((int*)fstreamc)[0] = n1;
		((int*)fstreamc)[1] = n3;
	}
	// if serial mode, other procs send their sub-matrix to procs 0, and procs 0 write the matrix
	if(mode == SERIAL){
		gather_matrix((double*)(fstreamc+sizeof(int)*2), dim, C, bufC_size, root, myrank, numprocs);
		if(myrank == 0){
			// procs 0 write and close the file
			fwrite(fstreamc, sizeof(char), fsizec, outfile);
			fclose(outfile);
		}
	}
	// else if parallel mode, use MPI IO to write all matrix
	else{
		parallel_write(argv[FILEC], dim, C, root, myrank);
	}
	if(DEBUG) printf("myrank: %d, gather matrix done\n", myrank);
	// synchronize
	MPI_Barrier(MPI_COMM_WORLD);
	// procs 0 output the result and free the memory
	if(myrank == 0){
		// output the i/o time
		printf("I/O time: %.2f sec\n", io_time);
		// check the ans, use procs 0 to compute a naive matrix-multiplication
		// compare the outcome with the cannon's outcome
		/*
		flag = check(fstreama, fstreamb, fstreamc);
		if(flag){
			printf("Error in check!\n");
		}
		*/
		total_time = MPI_Wtime() - total_time;
		// output total time and cannon time
		printf("Scatter time: %.2f sec\n", scatter_time);
		printf("Cannon time: %.2f sec\n", elapsed_time);
		printf("A: %d x %d, B: %d x %d, total time: %.2f sec\n",\
			n1, n2, n2, n3, total_time);
		free(fstreama);
		free(fstreamb);
		free(fstreamc);
	}
	free(buf);
	MPI_Finalize();
	return 0;
}

double* read_matrix(FILE* file, int* dim)
{
	double* matrix;
	fread(dim, sizeof(int), 2, file);
	matrix = (double*)malloc(dim[0] * dim[1] * sizeof(double));
	fread(matrix, sizeof(double), dim[0] * dim[1], file);
	return matrix;
}
FILE* safe_open(char* filename, const char* mode)
{
	/*
		a wrapper for fopen
		try to open the file in mode
		if can not open, exit
	*/
	FILE* r;
	if(!(r = fopen(filename, mode))){
		printf("Error: can not open the file:%s\n", filename);
		MPI_Finalize();
		exit(-1);
	}
	return r;
}
int setup(int argc, char* argv[], char** fstreama, char** fstreamb, int dim[], int mode)
{
	/*
		overview:
			setup the matrix save A and B's data in fstreama and fstreamb
			save the dimension info in dim
			however, if parallel mode, only read the dimension info from file
		params:
			argc: para number
			argv: paras from cmd
			fstreama: pointer for A
			fstreamb: pointer for B
			dim: the dimension info	
			mode: the mode, parallel or serial	
		return:
			0 if success
			1 otherwise
	*/
	// check the number of papameter
	if(argc != PARAMETER_NUMBER){
		printf("Error: the number of parameters must be 3\n");
		return 1;
	}
	FILE *filea, *fileb;
	int dima[2], dimb[2];
	// open A and B
	filea = safe_open(argv[FILEA], "r");
	fileb = safe_open(argv[FILEB], "r");
	// read the dim
	fread(dima, sizeof(int), 2, filea);
	fread(dimb, sizeof(int), 2, fileb);
	// check the dimension of the two matrix
	if(dima[1] != dimb[0]){
		printf("Error: matrix A's col and B's row must be equal!\n");
		return 1;
	}
	// get the dim
	dim[0] = dima[0];
	dim[1] = dima[1];
	dim[2] = dimb[1];
	// if parallel mode, we have done, nothing to do
	if(mode == SERIAL){
		// serial mode, allocate the memory of fstream
		*fstreama = (char*)malloc(2 * sizeof(int) + dima[0] * dima[1] * sizeof(double));
		*fstreamb = (char*)malloc(2 * sizeof(int) + dimb[0] * dimb[1] * sizeof(double));
		// set the dim info
		((int*)*fstreama)[0] = dima[0];
		((int*)*fstreama)[1] = dima[1];
		((int*)*fstreamb)[0] = dimb[0];
		((int*)*fstreamb)[1] = dimb[1];
		// read the matrix data
		fread(*fstreama + 2 * sizeof(int), sizeof(double), dima[0] * dima[1], filea);
		fread(*fstreamb + 2 * sizeof(int), sizeof(double), dimb[0] * dimb[1], fileb);
	}
	// close the two files
	fclose(filea);
	fclose(fileb);
	// success, return 0
	return 0;
}

int maxlength(int n, int root)
{
	/*
		overview:
			given length n and root,
			return the max block length
	*/
	return (n + root - 1) / root;
}

int block2procs(int i, int j, int root)
{
	/*
		overview:
			given the block pos (i,j) and root
			return the procs number to procs this block
	*/
	return i * root + j;
}

int procs2block(int rank, int r, int c, int isa)
{
	/*
 		overview:
			given procs rank, mesh row r and col c,
			return the initial block number for this procs.
	*/
	int i,j;
	// get block position i,j
	i = rank / c;
	j = rank % c;
	if(isa)
		j = (j + i) % c;
	else
		i = (i + j) % r;
	// return
	return i * c + j;
}

void loadblock(double* fstream, double* m, int i ,int j, int row, int col, int root)
{
	/*
		overview:
			load a block (i,j) from fstream, which is a row x col matrix.
			the dimension info would be save in (int*)m[0...1]
	*/
	int maxrow, maxcol, r, c, temp, k, l;
	double* mm;
	// get the maxrow and col
	maxrow = maxlength(row, root);
	maxcol = maxlength(col, root);
	// get the dimension info
	temp = maxrow * (i+1);
	r = temp > row ? (row - maxrow * i) : maxrow;
	temp = maxcol * (j+1);
	c = temp > col ? (col - maxcol * j) : maxcol;
	// set the dimension info
	((int*)m)[0] = r;
	((int*)m)[1] = c;
	// load the matrix
	mm = (double*)((char*)m + sizeof(int) * 2);
	for(k = 0 ; k < r ; ++k){
		for(l = 0 ; l < c; ++l){
			mm[k*c + l] = fstream[(i * maxrow + k) * col + (j * maxcol + l)];
		}
	}
}

void parallel_read(int row, int col, double* m, int root, int myrank, char* filename, int isa)
{
	/*
 		overview:
			read sub-matrix for each procs from file using MPI IO
		params:
			row: row of matrix
			col: col of matrix
			m: the sub-matrix
			root: root of numprocs
			myrank: rank number
			buf_size: size of m
			isa: if isa == 1, read A, otherwise read B
		NOTE:
			this function has exactly same effect of scatter_matrix
			i.e. read sub-matrix and set the dimension info
	*/
	// MPI file
	MPI_File fh;
	// subarray datatype
	MPI_Datatype subarray;
	// offset
	MPI_Offset disp;
	// status
	MPI_Status status;
	// starts
	int starts[2];
	// subarray size
	int subsizes[2];
	// complete size, which is the matrix size
	int bigsizes[2] = {row, col};
	// other vars
	int r, c, b, maxrow, maxcol,temp,i,j;
	double* mm;
	r = (row >= root ? root : row);
	c = (col >= root ? root : col);
	// collective open
	MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
	// get disp
	disp = 2*sizeof(int);
	// get starts, subsizes
	b = procs2block(myrank, r, c, isa);
	i = b / c;
	j = b % c;
	// get maxlength
	maxrow = maxlength(row, root);
	maxcol = maxlength(col, root);
	// starts
	starts[0] = i * maxrow;
	starts[1] = j * maxcol;
	// subsizes	
	temp = maxrow * (i+1);
	subsizes[0] = temp > row ? (row - maxrow * i) : maxrow;
	temp = maxcol * (j+1);
	subsizes[1] = temp > col ? (col - maxcol * j) : maxcol;
	// set the dimension info
	((int*)m)[0] = subsizes[0];
	((int*)m)[1] = subsizes[1];
	// create subarray type
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarray);
	// commit
	MPI_Type_commit(&subarray);
	// set the view for each file
	MPI_File_set_view(fh, disp, MPI_DOUBLE, subarray, "native", MPI_INFO_NULL);
	// set the matrix pointer and read
	mm =(double*)((char*)m + 2 * sizeof(int));
	MPI_File_read_all(fh, mm, subsizes[0]*subsizes[1], MPI_DOUBLE, &status);
	// collective close 
	MPI_File_close(&fh);

}
void scatter_matrix(double* fstream, int row, int col, double* m, int root,\
					int myrank, int buf_size, int isa)
{
	/*
		overview:
			procs 0 send the sub-matrix to each procs
			where fstream saves the whole matrix,
			m is the sub-matrix, root = sqrt(numprocs)
			if isa == 1, send A, otherwise send B
	*/
	int i, j;
	int t,dim[2];
	int r,c;
	MPI_Status status;
	r = (row >= root ? root : row);
	c = (col >= root ? root : col);
	if (myrank == 0){
		// procs 0 send sub-matrix to other procs
		for(i = 0 ; i < r ; ++i){
			for(j = 0 ; j < c ; ++j){
				// find the target procs to send
				if(isa == 1) t = block2procs(i, (j - i + c) % c, c);
				else t = block2procs((i - j + r) % r, j , c);
				// if t == 0, skip
				if(t != 0){
					// load this block from matrix, including its dimension
					loadblock(fstream, m, i, j, row, col, root);
					// send
					MPI_Send(m, buf_size , MPI_CHAR, t, isa == 1 ? t : t + 1, MPI_COMM_WORLD);
				}

			}
		}
		// finally, procs 0 should load its own block, but not send it.
		loadblock(fstream, m, 0, 0, row, col, root);
	}
	else{
		// other procs receive the matrix
		MPI_Recv(m, buf_size, MPI_CHAR, 0, isa == 1 ? myrank : myrank + 1, MPI_COMM_WORLD, &status);
	}
}

void swap(double** a, double** b)
{
	// swap a and b
	double* temp;
	temp = *a;
	*a = *b;
	*b = temp;
}

double* array(double* m)
{
	return (double*)((char*)m + 2 * sizeof(int));
}

void time_and_plus(double* A, double* B, double* C)
{
	/*
		overview:
			execute C += A * B
	*/
	int i,j,k;
	int dim[3];
	double *aa, *bb, *cc;
	// get the dim info
	dim[0] = ((int*)C)[0];
	dim[1] = ((int*)A)[1];
	dim[2] = ((int*)C)[1];
	aa = array(A);
	bb = array(B);
	cc = array(C);
	// multiply A and B, add the value to C
	for(i = 0 ; i < dim[0] ; ++i){
		for(j = 0 ; j < dim[2] ; ++j){
			for(k = 0 ; k < dim[1] ; ++k){
				cc[i * dim[2] + j] += aa[i * dim[1] + k] * bb[k * dim[2] + j];
			}
		}
	}

}

void shiftblock(double* A, double* bufA, int bufA_size, double* B, double* bufB,\
	int bufB_size, int myrank, int root, int dim[])
{
	/*
		overview:
			left shift A and up shift B by one position,
			according to the cannon algorithm.
			the received block would be saved in buf
		NOTE:
			here in order to avoid deadlock
			the odd position procs would first receive then send,
			in contrast,
			the even position procs would first
			send then receive.
	*/
	// left shift A
	// if there's only one column block in A we don't need to shift
	int pos, t, col;
	MPI_Status status;
	if(root <= dim[1]){
		pos = myrank % root;
		// odd position procs send and receive
		if(pos % 2 == 1){
			// send to myrank - 1
			t = myrank - 1;
			MPI_Send(A, bufA_size, MPI_CHAR, t, t, MPI_COMM_WORLD);
			// receive from myrank + 1
			t = (pos == root - 1) ? (myrank / root) * root : myrank + 1;
			MPI_Recv(bufA, bufA_size, MPI_CHAR, t, myrank, MPI_COMM_WORLD, &status);
		}
		// even procs receive and send
		else{
			// receive from myrank + 1
			t = (pos == root - 1) ? (myrank / root) * root : myrank + 1;
			MPI_Recv(bufA, bufA_size, MPI_CHAR, t, myrank, MPI_COMM_WORLD, &status);
			// send to myrank - 1
			t = (pos == 0) ? myrank + root - 1 : myrank - 1;
			MPI_Send(A, bufA_size, MPI_CHAR, t, t, MPI_COMM_WORLD);
		}
	}
	// synchronize
	MPI_Barrier(MPI_COMM_WORLD);
	// up shift B
	// if there's only one row block in B, no need to shift
	if(root <= dim[1]){
		// get the col number
		col = (root <= dim[2]) ? root : 1;
		// get the row position
		pos = myrank / col;
		// odd procs send and recevie
		if(pos % 2 == 1){
			// get the target
			t = (pos - 1) * col + myrank % col;
			// send
			MPI_Send(B, bufB_size, MPI_CHAR, t, t + 1, MPI_COMM_WORLD);
			// get the receive target
			t = (pos == root - 1) ? myrank % col : (pos + 1) * col + myrank % col;
			// receive
			MPI_Recv(bufB, bufB_size, MPI_CHAR, t, myrank + 1, MPI_COMM_WORLD, &status);
		}
		// even procs receive and send
		else{
			// get the receive position
			t = (pos == root - 1) ? myrank % col : (pos + 1) * col + myrank % col;
			// receive
			MPI_Recv(bufB, bufB_size, MPI_CHAR, t, myrank + 1, MPI_COMM_WORLD, &status);
			// get the send position
			t = (pos == 0) ? (root - 1) * col + myrank % col : (pos - 1) * col + myrank % col;
			// send
			MPI_Send(B, bufB_size, MPI_CHAR, t, t + 1, MPI_COMM_WORLD);
		}
	}
	// synchronize
	MPI_Barrier(MPI_COMM_WORLD);
}

void cannon(double* A, double* bufA, int bufA_size, double* B, double* bufB,\
	int bufB_size, double* C, int bufC_size, int dim[], int root,\
	int myrank)
{
	/*
		overview:
			The cannon algorithm
			Run the following step root time:
				1. for each procs, compute C += A * B
				   namely the time-plus operation
				2. left shift A , up shift B by one position 
	*/
	int step = root;
	double* temp;
	while(step--){
		// execute the time-plus operation
		time_and_plus(A, B, C);
		// synchronize
		MPI_Barrier(MPI_COMM_WORLD);
		// shift the block, the received block would be saved in buf
		if(step == 0) break; // if step == 0, no need to shift
		shiftblock(A, bufA, bufA_size, B, bufB, bufB_size, myrank, root, dim);
		// swap the matrix pointer and buf pointer for A and B
		swap(&A, &bufA);
		swap(&B, &bufB);
	}
}


void write_matrix(double* fstreamc, int dim[], double* C, int root, int myrank)
{
	/*
		overview:
			given procs number and C,
			write block C to correct position in complete C
		param:
			fstream: complete C, with no dimension info
			C: block C, with dimension info 
			myrank: procs number
	*/
	int i, j, r ,c, row, col, row_offset, col_offset;
	double* cc;
	cc = (double*)((char*)C + 2*sizeof(int));
	// get row and col number of block C
	r = ((int*)C)[0];
	c = ((int*)C)[1];
	// get row offset and col offset
	row_offset = (dim[2] < root ? myrank / dim[2] : myrank / root) * maxlength(dim[0], root);
	col_offset = (dim[2] < root ? myrank % dim[2] : myrank % root) * maxlength(dim[2], root);
	for(i = 0 ; i < r; ++i){
		for(j = 0 ; j < c; ++j){
			fstreamc[(row_offset + i) * dim[2] + col_offset + j] = cc[i*c + j];
		}
	}
}

void gather_matrix(double* fstreamc, int dim[], double* C, int bufC_size, int root, int myrank, int numprocs)
{
	/*
		overview:
			collect the cannon's outcome from each procs to procs 0
			write the outcome to fstreamc
		param:
			fstream: matrix, with no dimension info
			C: block, with dimension info
	*/
	int i;
	MPI_Status status;
	// procs 0 receive C from other procs and write it in fstream
	if(myrank == 0){
		// first, write its own C
		write_matrix(fstreamc, dim, C, root, myrank);
		for(i = 1; i < numprocs ; ++i){
			// receive
			MPI_Recv(C, bufC_size, MPI_CHAR, i, i, MPI_COMM_WORLD, &status);
			// write
			write_matrix(fstreamc, dim, C, root, i);
		}
	}
	// other procs send C to procs 0
	else{
		MPI_Send(C, bufC_size, MPI_CHAR, 0, myrank, MPI_COMM_WORLD);
	}
}

void parallel_write(char* filename, int dim[], double* C, int root, int myrank)
{
	/*
 		overview:
			use MPI IO to write all the sub-matrix to the outfile
		params:
			filename: the output file path
			C: the sub-matrix
			myrank: procs rank
			dim: matrix dimension for A and B
		NOTE:
			this function would not gather matrix, but write from each procs
	*/
	// vars
	MPI_File fh;
	MPI_Datatype subarray;
	MPI_Offset disp;
	MPI_Status status;
	int starts[2], subsizes[2], bigsizes[2];
	int r,c,maxrow,maxcol;
	FILE* outfile;
	double* cc;
	// get bigsizes
	bigsizes[0] = dim[0];
	bigsizes[1] = dim[2];
	// procs 0 use standard IO to write the dimension info
	if(myrank == 0){
		outfile = safe_open(filename, "w");
		fwrite(bigsizes, sizeof(int), 2, outfile);
		// close the file
		fclose(outfile);
	}
	//synchronize
	MPI_Barrier(MPI_COMM_WORLD);
	// starts	
	c = (dim[2] >= root ? root : dim[2]);
	maxrow = maxlength(dim[0], root);
	maxcol = maxlength(dim[2], root);
	starts[0] = (myrank / c) * maxrow;
	starts[1] = (myrank % c) * maxcol;
	// subsizes
	subsizes[0] = ((int*)C)[0];
	subsizes[1] = ((int*)C)[1];
	if(DEBUG) printf("rank %d, bigsizes: %d %d, subsizes: %d %d, starts: %d %d\n", myrank, bigsizes[0], bigsizes[1], subsizes[0], subsizes[1], starts[0], starts[1]);
	// collective open, append mode
	MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	// create subarray
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarray);
	// commit
	MPI_Type_commit(&subarray);
	// disp
	//disp = 0;
	disp = 2 * sizeof(int);
	// set view
	MPI_File_set_view(fh, disp, MPI_DOUBLE, subarray, "native", MPI_INFO_NULL);
	//printf("OK2");
	// collective write
	cc = (double*)((char*)C + 2 * sizeof(int));
	MPI_File_write_all(fh, cc, subsizes[0]*subsizes[1], MPI_DOUBLE, &status);
	// collective close
	MPI_File_close(&fh);
	if(DEBUG) printf("parallel write done\n");
}
void print_matrix(double* m, int row, int col)
{
	/*
		overview:
			print a row x col matrix
	*/
	int i,j;
	for(i = 0 ; i < row ; ++i){
		for(j = 0 ; j < col; ++j){
			printf("%lf ", m[i * col + j]);
		}
		printf("\n");
	}
	printf("********************************\n");
}

void print_matrix2(double* m, int row, int col)
{
	double *mm;
	printf("%d x %d\n",row ,col);
	mm = (double*)((char*)m + 2*sizeof(int));
	print_matrix(mm, row, col);
}

int check(char* a, char* b, char* c)
{
	/*
		overview:
			use a naive single procs matrix-multiplication
			to compute the outcome of A * B
			and compare the value with C
		return:
			0, if C is correct
			1, otherwise
	*/
	double *dd, *cc, *aa, *bb;
	double serial_time;
	int row, col, mid,i, j, k, flag;
	// get dimension info
	row = ((int*)c)[0];
	col = ((int*)c)[1];
	mid = ((int*)a)[1];
	// set the matrix pointers
	aa = (double*)(a + 2 * sizeof(int));
	bb = (double*)(b + 2 * sizeof(int));
	cc = (double*)(c + 2 * sizeof(int));
	dd = (double*)malloc(row * col * sizeof(double));
	memset(dd, 0, row * col * sizeof(double));
	// naive matrix multiplication
	flag = 0;
	// record the serial multiplication time
	serial_time = MPI_Wtime();
	for(i = 0 ; i < row ; ++i){
		for(j = 0 ; j < col ; ++j){
			for(k = 0 ; k < mid ; ++k){
				dd[i * col + j] += aa[i * mid + k] * bb[k * col + j];
			}
			// check
			if(cc[i * col + j] != dd[i * col + j]){
				flag = 1;
				break;
			}
		}
		if(flag == 1) break;
	}
	serial_time = MPI_Wtime() - serial_time;
	// output the serial time
	printf("Serial time: %.2f sec\n", serial_time);
	// if error and DEBUG, dump the matrix
	if(DEBUG){
		print_matrix(aa, row, mid);
		print_matrix(bb, mid, col);
		print_matrix(cc, row, col);
		print_matrix(dd, row, col);
	}
	return flag;
}

