#include <math.h>
#include <stdio.h>

# define N 80
# define N_THRDS 4

__global__ void init_mat (int *A_array, int **A, const size_t cols){

    size_t i = threadIdx.x;
    A[i] = (A_array + i*cols);
}
__global__ void fast_transpose(int **A, int **B, const size_t dim){

    __shared__ int a_block[N_THRDS];
    __shared__ int b_block[N_THRDS];

    if (threadIdx.x == 0 && threadIdx.y == 0){
    
	size_t i;
	for (i = 0; i < N_THRDS; ++i)
	    a_block[i] = A[dim*blockIdx.y + i/dim][dim*blockIdx.x + i%dim];
    }
    __syncthreads();

    b_block[dim*threadIdx.y + threadIdx.x] = a_block[dim*threadIdx.x + threadIdx.y];

    __syncthreads();

    if(threadIdx.x == 0 && threadIdx.y == 0){
    
	size_t i;
	for (i = 0; i < N_THRDS; ++i)
	    B[dim*blockIdx.x + i/dim][dim*blockIdx.y + i%dim] = b_block[i];
    }
}

void fill_mat(int *mat, const size_t rows, const size_t cols);
int is_transpose(int *mat, int *transp, const size_t n);
void print_is_transpose(int *mat, int *transp, const size_t n);

int main() {

  int* mat_array = (int*) malloc(N*N*sizeof(int));
  int* transp_array = (int*) malloc(N*N*sizeof(int));

  fill_mat(mat_array, N, N);

  int *dev_mat_array, *dev_transp_array;
  int **dev_mat, **dev_transp;

  cudaMalloc( (void**)&dev_mat_array, N*N*sizeof(int) );
  cudaMalloc( (void**)&dev_transp_array, N*N*sizeof(int) );
  cudaMalloc( (void***)&dev_mat, N*sizeof(int) );
  cudaMalloc( (void***)&dev_transp, N*sizeof(int) );

  cudaMemcpy( dev_mat_array, mat_array, N*N*sizeof(int), cudaMemcpyHostToDevice ); 

  init_mat<<< 1, N >>>(dev_mat_array, dev_mat,N);
  init_mat<<< 1, N >>>(dev_transp_array, dev_transp,N);

  int dim= (int)sqrt(N_THRDS);
  dim3 grid,block;
  grid.x=N/dim;
  grid.y=N/dim;
  block.x=dim;
  block.y=dim;

  fast_transpose<<< grid, block >>>(dev_mat, dev_transp,dim);

  cudaMemcpy( transp_array, dev_transp_array, N*N*sizeof(int),   cudaMemcpyDeviceToHost );

//  printf("Matrix:\n");
//  size_t i;
//  for(i=0;i<N*N;i++){
//    if(i%N==0 && i!=0)printf("\n");
//      printf("%d ", mat_array[i]);
//  }
//  printf("\n");
//
//  printf("Transpose:\n");
//  for(i=0;i<N*N;i++){
//    if(i%N==0 && i!=0)printf("\n");
//
//printf("%d ", transp_array[i]);
//  }
//  printf("\n");
//
//  printf("\nIs transpose: %d\n", is_transpose(mat_array, transp_array, N));

  print_is_transpose(mat_array, transp_array, N);

  free(mat_array); free(transp_array);
  cudaFree( dev_mat_array ); cudaFree( dev_transp_array ); cudaFree(dev_mat);cudaFree(dev_transp);
  return 0;
}
void fill_mat(int *mat, const size_t rows, const size_t cols){

    size_t i;
    for (i = 0; i < rows*cols; ++i)
	    mat[i] = rand() % 100;	   
}
int is_transpose(int *mat, int *transp, const size_t n){
    
    size_t i, j;
    for (i = 0; i < n; ++i)
	for (j = 0; j < n; ++j)
	    if (mat[i*n + j] != transp[j*n + i])
		return 0;
    return 1;
}
void print_is_transpose(int *mat, int *transp, const size_t n){
    
    size_t i, j;
    for (i = 0; i < n; ++i){
	for (j = 0; j < n; ++j)
	    printf("%d ",(mat[i*n + j] != transp[j*n + i]) ? 0 : 1);
	putchar('\n');
    }
}
