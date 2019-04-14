#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void print_submatrix (int **matrix, const size_t rows, const size_t cols, FILE *outfile);

int main (int argc, char **argv){

    const size_t N = 18;
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t n_proc, rest;
    rest = N % size;
    n_proc = N/size + (((size_t) rank <  rest) ? 1 : 0);

    size_t start;
    start = rank*n_proc + (((size_t) rank <  rest) ? 0 : rest);

    int **local_mat = (int **) calloc(n_proc, sizeof(int *));
    size_t i;
    for (i = 0; i < n_proc; ++i)
	local_mat[i] = (int *) calloc(N, sizeof(int));

    size_t j;
    for (i = 0; i < n_proc; ++i)
	for (j = 0; j < N; ++j)
	    local_mat[i][j] = ((i + start) == j) ? 1 : 0;

    if (rank == 0){
    
	FILE *output_file = (N <= 10) ? stdout : fopen( "identity_matrix.txt", "w");
	print_submatrix(local_mat, n_proc, N, output_file);

	int proc;
	for (proc = 1; proc < size; ++proc){
	
	    size_t limit = n_proc;
	    if (rest > 0 && (size_t) proc >= rest)
		limit--;
	    for (i = 0; i < limit; ++i){
		MPI_Recv(local_mat[i], N, MPI_INT, proc, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }

	    print_submatrix(local_mat, limit, N, output_file);
	}
    }
    else{
    
	for (i = 0; i < n_proc; ++i){
	    MPI_Send(local_mat[i], N, MPI_INT, 0, 101, MPI_COMM_WORLD);
	}
    }

    for (i = 0; i < n_proc; ++i)
	free(local_mat[i]);
    free(local_mat);

    MPI_Finalize();
}

void print_submatrix (int **matrix, const size_t rows, const size_t cols, FILE *outfile){

    size_t i, j;
    for (i = 0; i < rows; ++i){
	for (j = 0; j < cols - 1; ++j)
	    fprintf(outfile, "%d\t", matrix[i][j]);
	fprintf(outfile, "%d\n", matrix[i][cols -1]);
    }
}
