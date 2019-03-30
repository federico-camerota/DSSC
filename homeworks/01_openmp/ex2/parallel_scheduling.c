#include <omp.h>
#include <stdio.h>

void print_usage( int * a, int N, int nthreads );

int main(){

    const int N = 150;
    int a[N];
    int nthreads;

    #pragma omp parallel 
    {

	#pragma omp master
	{
	    nthreads = omp_get_num_threads();
	}

	int thread_id = omp_get_thread_num();


	#pragma omp for schedule(static)
	for(int i = 0; i < N; ++i) {
	    a[i] = thread_id;
	}
	#pragma omp single
	{
	    printf("*** Static scheduling ***\n");
	    print_usage(a, N, nthreads);
	}

	#pragma omp for schedule(static, 1)
	for(int i = 0; i < N; ++i) {
	    a[i] = thread_id;
	}
	#pragma omp single
	{
	    printf("*** Static scheduling with chunk size 1 ***\n");
	    print_usage(a, N, nthreads);
	}
	
	#pragma omp for schedule(static, 10)
	for(int i = 0; i < N; ++i) {
	    a[i] = thread_id;
	}
	#pragma omp single
	{
	    printf("*** Static scheduling with chunk size 10 ***\n");
	    print_usage(a, N, nthreads);
	}

	#pragma omp for schedule(dynamic)
	for(int i = 0; i < N; ++i) {
	    a[i] = thread_id;
	}
	#pragma omp single
	{
	    printf("*** Dynamic scheduling ***\n");
	    print_usage(a, N, nthreads);
	}

	#pragma omp for schedule(dynamic, 1)
	for(int i = 0; i < N; ++i) {
	    a[i] = thread_id;
	}
	#pragma omp single
	{
	    printf("*** Dynamic scheduling with chunk size 1 ***\n");
	    print_usage(a, N, nthreads);
	}
	
	#pragma omp for schedule(dynamic, 10)
	for(int i = 0; i < N; ++i) {
	    a[i] = thread_id;
	}
	#pragma omp single
	{
	    printf("*** Dynamic scheduling with chunk size 10 ***\n");
	    print_usage(a, N, nthreads);
	}
    }
}
