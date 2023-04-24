#include <iostream>
#include <cmath>
#include <omp.h>

using namespace std;

const int MAX_ITER = 100; // maximum number of iterations for convergence
const double TOLERANCE = 1e-6; // tolerance for error of solution

// function to solve linear system Ax = b using Gauss-Seidel method with OpenMP parallelization
void gauss_seidel_omp(int n, double** A, double* b, double* x) {
    double* prev_x = new double[n]; // store previous variables for convergence check

    // initialize variables
    for (int i = 0; i < n; i++) {
        x[i] = 0.0; // initial guess for solution
        prev_x[i] = 0.0; // initialize previous variables
    }

    int iter = 0; // iteration count

    // iterate until convergence or maximum iteration count reached
    while (iter < MAX_ITER) {
        double error = 0.0; // track error of solution

        // solve for each variable in parallel
        #pragma omp parallel for shared(A, b, x, prev_x) reduction(+: error)
        for (int i = 0; i < n; i++) {
            double sum = 0.0;

            for (int j = 0; j < n; j++) {
                if (j != i) {
                    sum += A[i][j] * x[j];
                }
            }

            // update variable with Gauss-Seidel formula
            double new_x = (1 - A[i][i]) * x[i] + (b[i] - sum) / A[i][i];
            error += abs(new_x - x[i]); // update error
            prev_x[i] = x[i]; // store previous variable for convergence check
            x[i] = new_x; // update variable
        }

        // check for convergence
        bool is_converged = true;
        #pragma omp parallel for shared(x, prev_x) reduction(&&: is_converged)
        for (int i = 0; i < n; i++) {
            if (abs(x[i] - prev_x[i]) > TOLERANCE) {
                is_converged = false;
            }
        }

        if (is_converged) {
            break;
        }

        iter++; // increment iteration count
    }

    // delete dynamically allocated memory
    delete[] prev_x;
}

// function to print matrix
void print_matrix(int n, double** A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << A[i][j] << " ";
        }

        cout << endl;
    }
}

// function to print vector
void print_vector(int n, double* v) {
    for (int i = 0; i < n; i++) {
        cout << v[i] << " ";
    }

    cout << endl;
}

int main() {
    int n = 3;
    double** A = new double*[n];
    double* b = new double[n];
    double* x = new double[n];

    // initialize matrix and vector
    A[0] = new double[3] { 2, -1, 0 };
    A[1] = new double[3] { -1, 2, -1 };
    A[2] = new double[3] { 0, -1, 2 };
    b[0] = 1;
    b[1] = 0;
    b[2] = 0;

    cout << "Matrix A:" << endl;
    print_matrix(n, A);

    cout << "Vector b:" << endl;
    print_vector(n, b);

    // solve linear system with Gauss-Seidel method with OpenMP parallelization
    gauss_seidel_omp(n, A, b, x);

    cout << "Solution x:" << endl;
    print_vector(n, x);

    // delete dynamically allocated memory
    for (int i = 0; i < n; i++) {
        delete[] A[i];
    }
    delete[] A;
    delete[] b;
    delete[] x;

    return 0;
}













