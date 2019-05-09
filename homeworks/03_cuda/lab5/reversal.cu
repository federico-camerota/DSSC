#include <stdio.h>

#define N 5

__global__ void revert(int *d_in, int *d_out, const size_t n){

    size_t idx = blockIdx.x;
    if (idx < n)
	d_out[idx] = d_in[n - 1 - idx];
}

int main(){

    int *d_in, *d_out;
    d_in = (int *) calloc(N, sizeof(int));
    d_out = (int *) calloc(N, sizeof(int));

    d_in[0] = 100; 
    d_in[1] = 110; 
    d_in[2] = 200; 
    d_in[3] = 220; 
    d_in[4] = 300; 

    int *dev_in, *dev_out;
    cudaMalloc((void **) &dev_in, N*sizeof(int)); 
    cudaMalloc((void **) &dev_out, N*sizeof(int)); 

   cudaMemcpy(dev_in, d_in,  N*sizeof(int), cudaMemcpyHostToDevice); 
   
   revert<<<N, 1>>>(dev_in, dev_out, N);

   cudaMemcpy(d_out, dev_out,  N*sizeof(int), cudaMemcpyDeviceToHost); 

   size_t i;
   printf("d_in: ");
   for (i = 0; i < N; ++i)
      printf("%d\t", d_in[i]);
   putchar('\n');
   printf("d_out: ");
   for (i = 0; i < N; ++i)
      printf("%d\t", d_out[i]);
   putchar('\n');

   free(d_in);
   free(d_out);
   cudaFree(dev_in);
   cudaFree(dev_out);
}
