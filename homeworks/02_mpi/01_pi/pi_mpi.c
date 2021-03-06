#include <mpi.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

double seconds()

/* Return the second elapsed since Epoch (00:00:00 UTC, January 1, 1970)                                                                     
 */

{

  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;

}

int main(int argc, char* argv[]){


    //number of breaks in the [0,1] interval
    size_t N = 10000000000;
    double h_2 = (1.0/N)/2;

    MPI_Init(&argc, &argv);

    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //compute the number of intervals per process
    size_t n_per_proc = N/size;
    //compute the endpoints of the intervals to process
    size_t init_val = n_per_proc*rank;
    size_t end_val = init_val + n_per_proc;
    //the last process handles remaining intervals
    if (rank == size - 1)
	end_val = N;

    double start_time = seconds();
    double elapsed;

    double local_pi = 0.0;
    double global_pi = 0.0;
    for (size_t i = init_val; i < end_val; ++i){
	double x = (2*i +1)*h_2;
	local_pi += 1.0/(1.0 + x*x);
    }

    MPI_Reduce(&local_pi, &global_pi, 1, MPI_DOUBLE, MPI_SUM, size - 1, MPI_COMM_WORLD);
    elapsed = seconds() - start_time;

    if (rank == size - 1){
	global_pi = 4*global_pi*2*h_2;
	MPI_Send(&global_pi, 1, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);
	MPI_Send(&elapsed, 1, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);
    }
    if (rank == 0){
	MPI_Recv(&global_pi, 1,  MPI_DOUBLE, size - 1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&elapsed, 1,  MPI_DOUBLE, size - 1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	printf("From process %d, computed value of pi is %lf (elapsed %lf seconds)\n", rank, global_pi, elapsed);
    }

    MPI_Finalize();
}
