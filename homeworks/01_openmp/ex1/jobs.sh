#load gnu module
module load gnu

for i in 1 2 4 8 16 20; do
    #set number of threads
    export OMP_NUM_THREADS=$i
    #print number of threads
    echo 'Pi calculation using ' $i ' threads'
    #run program
    ./parallel_pi.x
done

exit
