#include <stdio.h>
#include <omp.h>

double approx_pi(size_t N){

    double h_2 = (1.0/N)/2;
    double pi = 0.0;

    #pragma omp parallel //reduction(+:pi) //<-- To use omp reduction
    {
	double local_pi = 0.0;

	#pragma omp for schedule(static)
	for (size_t i = 0; i < N; ++i)
	    local_pi += 1.0/(1.0 + (2*i + 1)*h_2*(2*i+1)*h_2);

	//#pragma omp atomic //<-- To use omp atomic
	#pragma omp critical //<-- To use omp critical
	pi += local_pi;
    }

    return 4*pi*2*h_2;
}

int main(){

    double start_time = omp_get_wtime();
    double pi = approx_pi(10000000);
    double end_time = omp_get_wtime();
    printf("pi = %lf (%lf seconds)\n",pi, (end_time - start_time));
}
