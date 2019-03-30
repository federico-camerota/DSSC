#load gnu module
module load gnu

echo 'critical = ' >> ./hpc_parallel_programming/Lab2/new_times.txt
for i in 1 2 4 8 16 20; do
    #set number of threads
    export OMP_NUM_THREADS=$i
    #print number of threads
    echo 'Pi calculation using ' $i ' threads'
    #run program
    ./hpc_parallel_programming/Lab2/critical_pi.x >> ./hpc_parallel_programming/Lab2/new_times.txt
done

echo 'atomic = ' >> ./hpc_parallel_programming/Lab2/new_times.txt
for i in 1 2 4 8 16 20; do
    #set number of threads
    export OMP_NUM_THREADS=$i
    #print number of threads
    echo 'Pi calculation using ' $i ' threads'
    #run program
    ./hpc_parallel_programming/Lab2/atomic_pi.x >> ./hpc_parallel_programming/Lab2/new_times.txt
done

echo 'reduction = ' >> ./hpc_parallel_programming/Lab2/new_times.txt
for i in 1 2 4 8 16 20; do
    #set number of threads
    export OMP_NUM_THREADS=$i
    #print number of threads
    echo 'Pi calculation using ' $i ' threads'
    #run program
    ./hpc_parallel_programming/Lab2/reduction_pi.x >> ./hpc_parallel_programming/Lab2/new_times.txt
done

exit
