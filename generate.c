/*
	Copyright (c) 2018 xindetai@Beihang University
	Generate huge dense matrix using MPI IO
	The size is given by cmd line

	Usage:
		./generate n1 n2 file
	params:
		n1, row 
		n2, col
		file: output file
	NOTE:
		YOU MUST USE 81 PROCS TO RUN THIS PROGRAM
		THE FINAL MATRIX SIZE IS 9*n1 x 9*n2
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define MAX 250
#define DEBUG 0
#define ROOT 9
FILE* safe_open(char* filename, const char* mode);
void generate_matrix(char* filename, int row, int col, int bufsize, int myrank);

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
	mm = (double*)((char*)m + 2*sizeof(int));
	print_matrix(mm, row, col);
}

void print_matrix3(char* filename, int row, int col)
{
	FILE* file;
	double* mm;
	int dim[2];
	file = safe_open(filename, "r");
	fread(dim, sizeof(int), 2, file);
	if(dim[0] != row || dim[1] != col) printf("Error: row x col = %d x %d, while actually %d x %d\n", row, col, dim[0], dim[1]);
	mm = (double*)malloc(dim[0] * dim[1] * sizeof(double));	
	fread(mm, sizeof(double), dim[0] * dim[1], file);
	fclose(file);
	int i,j;
	row = dim[0];
	col = dim[1];
	printf("%d x %d\n", row, col);
	for(i = 0 ; i < row ; ++i){
		for(j = 0; j < col; ++j){
			printf("%.2lf ", mm[i*col+j]);
		}
		printf("\n");
	}
	printf("********************************************\n");
	free(mm);
}
int main(int argc, char** argv)
{
	char *buf;
	int bufa_size, bufb_size;
	double *a, *b;
	int n1, n2;
	int myrank, numprocs;
	// check the number of para
	if(argc != 4){
		printf("Error: the number of parameter must be 6\n");
		exit(-1);
	}
	//init
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	if(numprocs != ROOT * ROOT){
		MPI_Finalize();
		printf("Error: number of procs must be 81\n");
		exit(-1);
	}
	// get the matrix size n1, n2, n3
	n1 = atoi(argv[1]);
	n2 = atoi(argv[2]);
	// compute the size of each matrix
	bufa_size = n1 * n2 * sizeof(double);
	// set the seed
	srand((unsigned)time(NULL) + myrank);
	// generate the matrix
	generate_matrix(argv[3], n1, n2, bufa_size, myrank);
	if(myrank == 0){
		if(DEBUG) print_matrix3(argv[3], n1*ROOT, n2*ROOT);
		printf("file: %s, matrix size: %d x %d\n", argv[3], n1*ROOT, n2*ROOT);
	}
	// return
	return 0;

}

FILE* safe_open(char* filename, const char* mode)
{
	FILE* r;
	if(!(r = fopen(filename, mode))){
		printf("Error: can not open the file:%s\n", filename);
		exit(-1);
	}
	return r;
}

void generate_matrix(char* filename, int row, int col, int bufsize, int myrank)
{
	/*
 		overview:
			generate ROOT*row x ROOT*col matrix using MPI IO
			each procs would generate a row x col matrix
	*/
	MPI_File fh;
	MPI_Datatype subarray;
	MPI_Offset disp;
	MPI_Status status;
	int starts[2],subsizes[2]={row, col}, bigsizes[2]={row*ROOT, col*ROOT};
	FILE* outfile;
	double* buf;
	// procs 0 write the dimension info
	if(myrank == 0){
		outfile = safe_open(filename, "w");
		fwrite(bigsizes, sizeof(int), 2, outfile);
		fclose(outfile);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	starts[0] = myrank / ROOT * row;
	starts[1] = myrank % ROOT * col;
	MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarray);
	MPI_Type_commit(&subarray);
	disp = 2 * sizeof(int);
	MPI_File_set_view(fh, disp, MPI_DOUBLE, subarray, "native", MPI_INFO_NULL);
	// generate the matrix
	if(!(buf = (double*)malloc(bufsize))){
		printf("Error: No memory\n");
		MPI_Finalize();
		exit(-1);
	}
	int i,j;
	for(i = 0 ; i < row ; ++i){
		for(j = 0 ; j < col; ++j){
			buf[i*col + j] = rand() % (MAX * 2) - MAX;
		}
	}
	MPI_File_write_all(fh, buf, subsizes[0]*subsizes[1], MPI_DOUBLE, &status);
	MPI_File_close(&fh);
	free(buf);
}
