#include <stdio.h>

double approx_pi(size_t N){

    double h_2 = (1.0/N)/2;
    double pi = 0.0;

    for (size_t i = 0; i < N; ++i)
	pi += 1.0/(1.0 + (2*i + 1)*h_2*(2*i+1)*h_2);

    return 4*pi*2*h_2;
}

int main(){

    printf("pi = %lf\n", approx_pi(10000000));
}
