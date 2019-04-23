## OpenMP Loop Schedules

In this exercise we visualize the behavior of the different scheduling modes
available in OpenMP.

[serial_scheduling.c](serial_scheduling.c) is the serial version of the program while
[parallel_scheduling.c](parallel_scheduling.c) provides its parallel version.

In the parallel program we test and visualize the following scheduling modes:
* `static`
* `static` with chunk size 1
* `static` with chunk size 10
* `dynamic`
* `dynamic` with chunk size 1
* `dynamic` with chunk size 10

### Compiling
To compile both versions of the program run `make`, to compile only one of the two
versions run `make serial` or `make parallel`.
