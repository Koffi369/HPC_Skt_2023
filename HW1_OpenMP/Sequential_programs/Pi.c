/*#include <stdio.h>

int main()
{
    const size_t N = 100000;
    double step;

    double x, pi, sum = 0.;

    step = 1. / (double)N;

    for (int i = 0; i < N; ++i)
    {
        x = (i + 0.5) * step;
        sum += 4.0 / (1. + x * x);
    }

    pi = step * sum;

    printf("pi = %.16f\n", pi);

    return 0;
}
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
 
///////////////////Sequential code
void SEQ_Pi_Monte_Carlo(int N)
{

    double x, y; // Cordinates X and Y
    double dist; // distance from (0,0) dist = (x * x) + (y * y)

    int Pts_Circle = 0;  // number of points falling inside the unit circle

    int Pts_Square = 0;  // number of points falling inside the unit square
 
    int i = 0;


        
    srand(time(NULL));

    for (i = 0; i < N; i++) {
        x = rand() / (float)RAND_MAX;   //  random X coordinate beetwin 0 to 1

        y = rand() / (float)RAND_MAX;   //  random Y coordinate beetwin 0 to 1

        dist = (x * x) + (y * y);       //  distance from (0,0) to (x,y)

        if (dist <= 1)  Pts_Circle++; // Increment Pts_Circle by 1
            
        Pts_Square++  ; // Increment Pts_Square by 1
    }

    
    double pi = 4.0 * ((double)Pts_Circle / (double)(Pts_Square)); // Estimated value of PI
 
    printf("Final Estimation with sequential code of Pi = %f\n", pi); // Prints the value in pi
}



///////////////////Parallel Code


void OMP_Pi_Monte_Carlo(int N, int K)
{

    double x, y; // Cordinates X and Y
    double dist; // distance from (0,0) dist = (x * x) + (y * y)

    int Pts_Circle = 0;  // number of points falling inside the unit circle

    int Pts_Square = 0;  // number of points falling inside the unit square
 
    int i = 0;
    int tid;
 
// Parallel calculation of random
// points lying inside a circle
#pragma omp parallel firstprivate(x, y, dist, i) reduction(+ : Pts_Circle, Pts_Square) num_threads(K)
    {
        tid = omp_get_thread_num();
        //printf("Hello World from thread = %d\n", tid);

        
        srand((tid + 1) * time(NULL));      // unidue seed for each thread
 
        for (i = 0; i < N; i++) {
            x = rand() / (float)RAND_MAX;   //  random X coordinate beetwin 0 to 1

            y = rand() / (float)RAND_MAX;   //  random Y coordinate beetwin 0 to 1

            dist = (x * x) + (y * y);       //  distance from (0,0) to (x,y)

            if (dist <= 1)  Pts_Circle++; // Increment Pts_Circle by 1
               
            Pts_Square++  ; // Increment Pts_Square by 1
        }
    }
    
    double pi = 4.0 * ((double)Pts_Circle / (double)(Pts_Square)); // Estimated value of PI
 
    printf("Final Estimation with parallel OMP code of Pi = %f\n", pi); // Prints the value in pi
}
 
// Main fuction
int main()
{
    // Inputs
    const size_t N = 100000;
    double start, end;
    int n_threads = 3;
    omp_set_num_threads(n_threads);

    // Functions call
    start = clock();
    SEQ_Pi_Monte_Carlo(N*n_threads);
    end = clock();
    printf("Time elapsed Sequential code: %f seconds.\n", (double)(end - start) / CLOCKS_PER_SEC);

    start = clock();
    OMP_Pi_Monte_Carlo(N, n_threads);
    end = clock();
    printf("Time elapsed Parallel code: %f seconds.\n", (double)(end - start) / CLOCKS_PER_SEC);

}


