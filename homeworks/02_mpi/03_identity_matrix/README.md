# Identity matrix

In this exercise we initialize an identity matrix distributed among processes and
then use the process with rank 0 in the `MPI_COMM_WORLD` communicator to print it
or save it into a file, depending on the matrix size.

## Compiling
To compile the program run `make`, a binary file called `identity_matrix.x` will
be created.
