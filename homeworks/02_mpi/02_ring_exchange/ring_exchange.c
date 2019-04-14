#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv){

    MPI_Init(&argc, &argv);

    int size, rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#ifndef MULTIPLE_DATA_RING
    int sum = 0;
    int message = rank;
#else
    const size_t N = 20;
    int sum[N], message[N];
    size_t j;
    for (j = 0; j < N; ++j){
	sum[j] = 0;
	message[j] = rank;
    }
#endif

    MPI_Request req;

    int i;
    for (i = 0; i < size; ++i){
    
#ifndef MULTIPLE_DATA_RING
	MPI_Isend(&message, 1, MPI_INT, (rank == size - 1) ? 0 : rank + 1, 101, MPI_COMM_WORLD, &req);
	sum += message;
	MPI_Wait(&req, MPI_STATUS_IGNORE);
	MPI_Recv(&message, 1, MPI_INT, (rank == 0) ? size - 1 : rank - 1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#else
	MPI_Isend(&message, N, MPI_INT, (rank == size - 1) ? 0 : rank + 1, 101, MPI_COMM_WORLD, &req);
	for (j = 0; j < N; ++j)
	    sum[j] += message[i];
	MPI_Wait(&req, MPI_STATUS_IGNORE);
	MPI_Recv(&message, N, MPI_INT, (rank == 0) ? size - 1 : rank - 1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
    }

#ifndef MULTIPLE_DATA_RING
    printf("Procces %d result is %d\n", rank, sum);
#else
    printf("Procces %d result is %d\n", rank, sum[N-1]);
#endif

    MPI_Finalize();
}
