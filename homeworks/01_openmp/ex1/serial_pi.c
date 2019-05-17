#include <stdio.h>

double approx_pi(size_t N){

    double h = (1.0/N);
    double pi = 0.0;

    for (size_t i = 0; i < N; ++i){
	double x = (i + 1.0/2)*h;
	pi += 1.0/(1.0 + x*x);
    }

    return 4*pi*h;
}

int main(){

    printf("pi = %lf\n", approx_pi(1E10));
}
