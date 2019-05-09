#include <stdio.h>
#include <stdlib.h>

__global__ void transpose_matrix(int** mat, int **tran, const size_t rows, const size_t cols){

    size_t i = blockIdx.x; 
    size_t j = threadIdx.x;
    if (i < rows && j < cols)
	tran[j][i] = mat[i][j]; 
}

void fill_mat(int **mat, const size_t rows, const size_t cols);
void print_mat(int **mat, const size_t rows, const size_t cols);

#define N 5
#define M 6

int main(){
    
    int **mat = (int **) calloc(N, sizeof(int *));
    int **transp = (int **) calloc(M, sizeof(int *));
    size_t i;

    int **dev_mat, **dev_transp;
    cudaMalloc((void***) &dev_mat, N*sizeof(int *));
    for (i = 0; i < N; ++i)
	cudaMalloc((void**) &(mat[i]), M*sizeof(int));
    cudaMemcpy(&dev_mat, &mat, N, cudaMemcpyHostToDevice);

    cudaMalloc((void***) &dev_transp, M*sizeof(int *));
    for (i = 0; i < M; ++i)
	cudaMalloc((void**) &(transp[i]), N*sizeof(int));
    cudaMemcpy(&dev_transp, &transp, M, cudaMemcpyHostToDevice);

    for (i = 0; i < N; ++i)
	mat[i] = (int *) calloc(M, sizeof(int));

    for (i = 0; i < M; ++i)
	transp[i] = (int *) calloc(N, sizeof(int));

    fill_mat(mat, N, M);

    for (i = 0; i < N; ++i)
	cudaMemcpy(&(dev_mat[i]), &(mat[i]), M, cudaMemcpyHostToDevice);


    transpose_matrix<<<N,M>>>(dev_mat, dev_transp, N, M);

    for (i = 0; i < M; ++i)
	cudaMemcpy(&(transp[i]), &(dev_transp[i]), N, cudaMemcpyDeviceToHost);

    printf("Matrix:\n");
    print_mat(mat, N, M);
    printf("Transpose:\n");
    print_mat(transp, M, N);
    
}

void fill_mat(int **mat, const size_t rows, const size_t cols){

    size_t i;
    size_t j;
    for (i = 0; i < rows; ++i)
       for (j = 0; j < cols; ++j)
	    mat[i][j] = rand() % 100;	   
}

void print_mat(int **mat, const size_t rows, const size_t cols){

    size_t i, j;
    for (i = 0; i < rows; ++i){

       for (j = 0; j < cols; ++j)
	   printf("%d\t", mat[i][j]); 

       putchar('\n');
    }
}
