## Pi approximation with MPI

In this exercise we approximate pi using a parallel approach based on MPI.
[pi_mpi.c](pi_mpi.c) provides the implementation of the program.

Pi is computed by approximating an integral using mid-point rule. To test the
scalability of the program, we ran it with a number of processes
ranging from 1 to 40 and using `10^10` breaks in the `[0,1]` interval.
The resulting times to solution are the following:
![mpi_scaling](mpi_scaling.jpg)

We can see that the the code scales properly as the number of processes is
increased.

### MPI vs OpenMP
We also compared the above implementation with the previous version of the
program obtained using OpenMP. Again the two programs have been tested one against
the other using a number of processes/threads ranging from 1 to 20.
![mpi_vs_openmp](mpi_vs_openmp.jpg)
From our results, the MPI implementation appears to be faster than the OpenMP
one.

## Compiling 
To compile the program run `make`.
