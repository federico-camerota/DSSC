#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>

# define N 8192
# define N_THRDS 256


__global__ void init_mat (size_t *A_array, size_t **A, const size_t cols){

    size_t i = threadIdx.x;
    while ( i < cols){
	A[i] = (A_array + i*cols);
	i += blockDim.x;
    }
}

__global__ void transpose (size_t **A, size_t **B, const size_t n){

    size_t x = threadIdx.x;
    size_t y = blockIdx.x; 
    while (x < n){
    
	B[x][y] = A[y][x];
	x += blockDim.x;
    }
}

__global__ void fast_transpose(size_t **A, size_t **B, const size_t dim){

    __shared__ size_t a_block[N_THRDS];

    size_t i = threadIdx.y*dim + threadIdx.x;
    a_block[i] = A[dim*blockIdx.y + i/dim][dim*blockIdx.x + i%dim];

    __syncthreads();

    B[dim*blockIdx.x + i/dim][dim*blockIdx.y + i%dim]= a_block[dim*threadIdx.x + threadIdx.y];
}

void fill_mat(size_t *mat, const size_t rows, const size_t cols);
int is_transpose(size_t *mat, size_t *transp, const size_t n);
void print_is_transpose(size_t *mat, size_t *transp, const size_t n);
double seconds();

int main() {

  size_t* mat_array = (size_t*) malloc(N*N*sizeof(size_t));
  size_t* transp_array = (size_t*) malloc(N*N*sizeof(size_t));

  fill_mat(mat_array, N, N);

  size_t *dev_mat_array, *dev_transp_array;
  size_t **dev_mat, **dev_transp;

  cudaMalloc( (void**)&dev_mat_array, N*N*sizeof(size_t) );
  cudaMalloc( (void**)&dev_transp_array, N*N*sizeof(size_t) );
  cudaMalloc( (void***)&dev_mat, N*sizeof(size_t) );
  cudaMalloc( (void***)&dev_transp, N*sizeof(size_t) );

  cudaMemcpy( dev_mat_array, mat_array, N*N*sizeof(size_t), cudaMemcpyHostToDevice ); 

  init_mat<<< 1, 1024 >>>(dev_mat_array, dev_mat,N);
  init_mat<<< 1, 1024 >>>(dev_transp_array, dev_transp,N);

  double start, elapsed;
  start = seconds();
  transpose<<<N, 1024>>>(dev_mat, dev_transp, N);
  cudaDeviceSynchronize();
  elapsed = seconds() - start;

  cudaMemcpy( transp_array, dev_transp_array, N*N*sizeof(size_t),   cudaMemcpyDeviceToHost );

  printf("Transpose result is: %d (%lf seconds)\n", is_transpose(mat_array, transp_array, N), elapsed);

 size_t dim= (size_t)sqrt(N_THRDS);
  dim3 grid,block;
  grid.x=N/dim;
  grid.y=N/dim;
  block.x=dim;
  block.y=dim;

  start = seconds();
  fast_transpose<<< grid, block >>>(dev_mat, dev_transp,dim);
  cudaDeviceSynchronize();
  elapsed = seconds() - start;

  cudaMemcpy( transp_array, dev_transp_array, N*N*sizeof(size_t),   cudaMemcpyDeviceToHost );

  printf("Fast transpose result is: %d (%lf seconds)\n", is_transpose(mat_array, transp_array, N), elapsed);

  free(mat_array); free(transp_array);
  cudaFree( dev_mat_array ); cudaFree( dev_transp_array ); cudaFree(dev_mat);cudaFree(dev_transp);
  return 0;
}
void fill_mat(size_t *mat, const size_t rows, const size_t cols){

    size_t i;
    for (i = 0; i < rows*cols; ++i)
        mat[i] = rand() % 100;	   
}
int is_transpose(size_t *mat, size_t *transp, const size_t n){
    
    size_t i, j;
    for (i = 0; i < n; ++i)
	for (j = i + 1; j < n; ++j)
	    if (mat[i*n + j] != transp[j*n + i])
		return 0;
    return 1;
}
void print_is_transpose(size_t *mat, size_t *transp, const size_t n){
    
    size_t i, j;
    for (i = 0; i < n; ++i){
    for (j = 0; j < n; ++j)
        printf("%d",(mat[i*n + j] != transp[j*n + i]) ? 0 : 1);
    putchar('\n');
    }
}

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
