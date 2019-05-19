# Ring exchange

In this exercise we implement and test, using MPI, the ring exchange communication
pattern. The method consists in each process first sending its datum to its successor
(identified by the communicator rank) and then receiving from its predecessor data and
forwarding what received to its successor.

The above has been implemented in two versions, in the first one the data of
each process consists in a single value (its rank in the `MPI_COMM_WORLD`
  communicator) while in the second one it is an array of values. Each process
  sends its data to the next one, receives data from its predecessor, sums it
  to its private data (in the case of multiple values sums are performed
    element-wise) and forwards to its successor what has received.
At the end, all the processes have the same values as data and print it to
standard output.

## Compiling
To compile the code run `make`, both version of the program will be created in
separated binary files named `single_data_ring.x` and `multiple_data_ring.x`.
