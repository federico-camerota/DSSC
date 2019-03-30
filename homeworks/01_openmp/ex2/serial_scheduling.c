void print_usage( int * a, int N, int nthreads );

int main(){
    const int N = 250;
    int a[N];
    int thread_id = 0;
    int nthreads = 1;
    for(int i = 0; i < N; ++i) {
	a[i] = thread_id;
    }

    print_usage(a, N, nthreads);
}
