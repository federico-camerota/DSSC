#include <stdio.h>
#include <omp.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

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
double approx_pi(size_t N){

    double h = (1.0/N);
    double pi = 0.0;

    #pragma omp parallel //reduction(+:pi) //<-- To use omp reduction
    {
	double local_pi = 0.0;

	size_t i;
	#pragma omp for schedule(static)
	for (i = 0; i < N; ++i){
	    double x = (i + 1/2)*h;
	    local_pi += 1.0/(1.0 + x*x);

	//#pragma omp atomic //<-- To use omp atomic
	#pragma omp critical //<-- To use omp critical
	pi += local_pi;
    }

    return 4*pi*2*h_2;
}

int main(){

    double start_time = seconds();
    double pi = approx_pi(10000000000);
    double end_time = seconds();
    printf("pi = %lf (%lf seconds)\n",pi, (end_time - start_time));
}
