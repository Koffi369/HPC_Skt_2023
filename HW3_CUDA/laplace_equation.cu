
//%%cu

#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

struct Grid {
    const size_t N = 6;  // Number of vertical segments
    const size_t M = 6;  // Number of horizontal segments

    const double x_min = 0.0;    // Minimum x-value
    const double x_max = 1.0;    // Maximum x-value
    const double y_min = 0.0;    // Minimum y-value
    const double y_max = 1.0;    // Maximum y-value

    const double x_left_bound = 0.0;    // Dirichlet condition at x = x_min
    const double x_right_bound = 0.0;   // Dirichlet condition at x = x_max
    const double y_bottom_bound = 1.0;  // Dirichlet condition at y = y_min
    const double y_top_bound = 0.0;     // Dirichlet condition at y = y_max

    const double dx = (x_max - x_min) / M;   // Grid spacing in x direction
    const double dy = (y_max - y_min) / N;   // Grid spacing in y direction

    const size_t num_rows = N - 1, num_cols = M - 1;
};

void create_system(Grid& grid, double *matrix_A, double *vector_b) {
    for (int i = 0; i < grid.num_cols; ++i) {
        for (int j = 0; j < grid.num_rows; ++j) {
            matrix_A[(i*grid.num_rows+j)*(grid.num_rows*grid.num_cols)+i*grid.num_rows+j] = -2 * (grid.dx*grid.dx + grid.dy*grid.dy);
            if (i == 0) vector_b[i*grid.num_rows + j] -= grid.dy*grid.dy * grid.x_left_bound;
            else matrix_A[(i*grid.num_rows+j)*(grid.num_rows*grid.num_cols)+(i-1)*grid.num_rows+j] = grid.dy*grid.dy;
            if (i == grid.num_cols-1) vector_b[i*grid.num_rows + j] -= grid.dy*grid.dy * grid.x_right_bound;
            else matrix_A[(i*grid.num_rows+j)*(grid.num_rows*grid.num_cols)+(i+1)*grid.num_rows+j] = grid.dy*grid.dy;
            if (j == 0) vector_b[i*grid.num_rows + j] -= grid.dx*grid.dx * grid.y_bottom_bound;
            else matrix_A[(i*grid.num_rows+j)*(grid.num_rows*grid.num_cols)+i*grid.num_rows+j-1] = grid.dx*grid.dx;
            if (j == grid.num_rows-1) vector_b[i*grid.num_rows + j] -= grid.dx*grid.dx * grid.y_top_bound;
            else matrix_A[(i*grid.num_rows+j)*(grid.num_rows*grid.num_cols)+i*grid.num_rows+j+1] = grid.dx*grid.dx;
        }
    }
}

double * create_field(Grid& grid, double* x) {
    double * u = new double[(grid.N+1)*(grid.M+1)]();
    for (size_t i = 0; i <= grid.N; ++i)
        u[i*(grid.M+1)] = grid.x_left_bound, u[i*(grid.M+1) + grid.M] = grid.x_right_bound;
    for (size_t i = 0; i <= grid.M; ++i)
        u[i] = grid.y_bottom_bound, u[(grid.N+1)*grid.M + i] = grid.y_top_bound;
    for (size_t i = 0; i < grid.num_cols; ++i)
      for (size_t j = 0; j < grid.num_rows; ++j)
          u[(j+1)*(grid.M+1)+i+1] = x[i*grid.num_rows + j];
    return u;
}

void print_field(Grid& grid, double* u) {
    for (int i = grid.N; i >= 0; --i) {
        for (int j = 0; j < grid.M+1; ++j) {
            std::cout << setprecision(7)<< u[i*(grid.M+1) +j] ;
            std::cout << " \t  ";
        }
        std::cout << std::endl;
    }
}

void save_to_file(Grid& grid, double* u) {
    std::ofstream f("out.txt");
    for (int i = grid.N; i >= 0; --i) {
        for (int j = 0; j < grid.M+1; ++j) {
            f << u[i*(grid.M+1) +j] << " ";
        }
        f << std::endl;
    }
    std::cout << "Succesfully saved to './out.txt'" << std::endl;
}



__device__ int max_elem_idx;

__global__ void findMaxInColumn(double* matrix, int n, int col) {
  int max_idx = col;
  double max_val = std::fabs(matrix[col * n + col]);
  double curr_val;

  // Loop through the rows below the given row to find the maximum element
  for (int i = col + 1; i < n; i++) {
    curr_val = std::fabs(matrix[i * n + col]);
    if (curr_val > max_val) {
      max_idx = i;
      max_val = curr_val;
    }
  }

  // Store the index of the maximum element in a global variable
  max_elem_idx = max_idx;
}

__global__ void swapRows(double* matrix, double* identity, int n, int row) {
  // If the given row is already the one with the maximum element, no need to swap
  if (row == max_elem_idx) {
    return;
  }

  // Calculate the index and offset for each thread
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = gridDim.x * blockDim.x;

  double temp;
  // Loop through the columns and swap the elements of the two rows
  for (; idx < n; idx += offset) {
    temp = matrix[row * n + idx];
    matrix[row * n + idx] = matrix[max_elem_idx * n + idx];
    matrix[max_elem_idx * n + idx] = temp;

    temp = identity[row * n + idx];
    identity[row * n + idx] = identity[max_elem_idx * n + idx];
    identity[max_elem_idx * n + idx] = temp;
  }
}



// Gaussian elimination on the matrix to reduce it to row echelon form
// Subtract the values below the pivot in each column

__global__ void subtract_below(double* matrix, double* identity, int size, int pivot) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int offset_x = gridDim.x * blockDim.x;
  int offset_y = gridDim.y * blockDim.y;

  int i, j;
  double coefficient;
  // Loop through the rows below the pivot row
  for (i = pivot + 1 + x; i < size; i += offset_x) {
    // Calculate the coefficient to multiply the pivot row by
    coefficient = matrix[i * size + pivot] / matrix[pivot * size + pivot];
    // Loop through the columns to subtract the values
    for (j = pivot + 1 + y; j < size; j += offset_y) {
      matrix[i * size + j] -= coefficient * matrix[pivot * size + j];
    }
    // Loop through the columns of the identity matrix to subtract the values
    for (j = y; j < size; j += offset_y) {
      identity[i * size + j] -= coefficient * identity[pivot * size + j];
    }
  }
}

// Nullify the values below the pivot in each column
__global__ void nullify_below(double* matrix, int size, int pivot) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int offset_x = gridDim.x * blockDim.x;
  // Loop through the rows below the pivot row
  for (int i = pivot + 1 + x; i < size; i += offset_x) {
    // Set the value to 0
    matrix[i * size + pivot] = 0.0;
  }
}

// Subtract the values above the pivot in each column
__global__ void subtract_above(double* matrix, double* identity, int size, int pivot) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int offset_x = gridDim.x * blockDim.x;
  int offset_y = gridDim.y * blockDim.y;

  int i, j;
  double coefficient;
  // Loop through the rows above the pivot row
  for (i = pivot - 1 - x; i >= 0; i -= offset_x) {
    // Calculate the coefficient to multiply the pivot row by
    coefficient = matrix[i * size + pivot] / matrix[pivot * size + pivot];
    // Loop through the columns to subtract the values
    for (j = pivot - 1 - y; j >= 0; j -= offset_y) {
      matrix[i * size + j] -= coefficient * matrix[pivot * size + j];
    }
    // Loop through the columns of the identity matrix to subtract the values
    for (j = y; j < size; j += offset_y) {
      identity[i * size + j] -= coefficient * identity[pivot * size + j];
    }
  }
}



//Global function to set all elements above the diagonal to zero

__global__ void nullifyAboveDiagonal(double* matrix, int n, int x) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offset_x = gridDim.x  * blockDim.x;
  
  // Iterate over the rows above the current row x and set all elements to zero
  for (int i = x - idx - 1; i >= 0; i -= offset_x) {
    matrix[i * n + x] = 0.0;
  }
}

// Global function to divide each element of the identity matrix by the corresponding element of the matrix being transformed
__global__ void divideIdentity(double* matrix, double* identity, int n) {
  int idx = blockIdx.x  * blockDim.x + threadIdx.x;
  int idy = blockIdx.y  * blockDim.y + threadIdx.y;
  int offset_x = gridDim.x * blockDim.x;
  int offset_y = gridDim.y * blockDim.y;
  
  // Iterate over the rows and columns of the matrices and divide corresponding elements
  for (int i = idx; i < n; i += offset_x) {
    for (int j = idy; j < n; j += offset_y) {
      identity[i * n + j] /= matrix[i * n + i];
    }
  }
}

// Global function to set the diagonal elements to 1.0
__global__ void SetDiagonalToOne(double* matrix, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offset_x = gridDim.x * blockDim.x;
  
  // Iterate over the diagonal elements and set them to 1.0
  for (int i = idx; i < n; i += offset_x) {
    matrix[i*  n + i] = 1.0;
  }
}


void InverseGPU(double* matrix, double* identity, int n) {
// Define the dimensions of the CUDA blocks and threads.
dim3 BLOCKS_1D(8);
dim3 THREADS_1D(32);
dim3 BLOCKS_2D(8, 8);
dim3 THREADS_2D(32, 32);

// Allocate memory on the device for the input matrix and identity matrix.
double* dev_input_matrix;
double* dev_identity_matrix;
cudaMalloc(&dev_input_matrix, sizeof(double) * n * n);
cudaMalloc(&dev_identity_matrix, sizeof(double) * n * n);

// Copy the input matrix and identity matrix from the host to the device.
cudaMemcpy(dev_input_matrix, matrix, sizeof(double) * n * n,
           cudaMemcpyHostToDevice);
cudaMemcpy(dev_identity_matrix, identity, sizeof(double) * n * n,
           cudaMemcpyHostToDevice);

// Create CUDA events to measure the execution time of the function.
cudaEvent_t begin, end;
cudaEventCreate(&begin);
cudaEventCreate(&end);
cudaEventRecord(begin, 0);

// Forward pass: perform Gaussian elimination to transform the input matrix into an upper triangular matrix.
for (int i = 0; i < n; i++) {
  // Find the maximum element in the current column and swap its row with the current row.
  findMaxInColumn<<<1, 1>>>(dev_input_matrix, n, i);
  swapRows<<<BLOCKS_1D, THREADS_1D>>>(dev_input_matrix, dev_identity_matrix, n, i);

  // Subtract multiples of the current row from all rows below it to eliminate elements below the diagonal.
  subtract_below<<<BLOCKS_2D, THREADS_2D>>>(dev_input_matrix, dev_identity_matrix, n, i);

  // Set all elements below the diagonal in the current column to 0.
  nullify_below<<<BLOCKS_1D, THREADS_1D>>>(dev_input_matrix, n, i);
}

// Backward pass: perform back-substitution to transform the upper triangular matrix into a diagonal matrix.
for (int i = n - 1; i >= 0; i--) {
  // Subtract multiples of the current row from all rows above it to eliminate elements above the diagonal.
  subtract_above<<<BLOCKS_2D, THREADS_2D>>>(dev_input_matrix, dev_identity_matrix, n, i);

  // Set all elements above the diagonal in the current column to 0.
  nullifyAboveDiagonal<<<BLOCKS_1D, THREADS_1D>>>(dev_input_matrix, n, i);
}

// Divide each element in the identity matrix by the corresponding element in the input matrix.
divideIdentity<<<BLOCKS_2D, THREADS_2D>>>(dev_input_matrix, dev_identity_matrix, n);

// Divide each element in the input matrix by the corresponding diagonal element to set all diagonal elements to 1.
SetDiagonalToOne<<<BLOCKS_1D, THREADS_1D>>>(dev_input_matrix, n);

// Record the execution time and destroy the CUDA events.
cudaEventRecord(end, 0);
cudaEventSynchronize(end);
float t;
cudaEventElapsedTime(&t, begin, end);
std::cout <<"Time taken for inversion: " << t << std::endl;
cudaEventDestroy(begin);
cudaEventDestroy(end);

// Copy the inverted matrix and identity matrix from the device to the host.
cudaMemcpy(matrix, dev_input_matrix, sizeof(double) * n * n,
           cudaMemcpyDeviceToHost);
cudaMemcpy(identity, dev_identity_matrix, sizeof(double) * n * n,
           cudaMemcpyDeviceToHost);

// Free the memory allocated on the device.
cudaFree(dev_input_matrix);
cudaFree(dev_identity_matrix);
}


// Function to perform matrix-vector multiplication

__global__ void MatrixVectorMult(double *mat, double *vec, double *res, int size) {
    // Calculate the row index based on the thread and block dimensions
    int row_idx = threadIdx.x + blockDim.x * blockIdx.x;
    // Initialize the sum to 0
    double sum = 0;
   
    // If the row index is within the bounds of the matrix
    if(row_idx < size) {
        // Iterate over each column in the row
        for(int col_idx = 0; col_idx < size; col_idx++) {
            // Multiply the matrix value by the corresponding vector value and add it to the sum
            sum += mat[row_idx * size + col_idx] * vec[col_idx];
        }
    }
    
    // Store the sum in the corresponding index of the result array
    res[row_idx] = sum;
}


void MatrixVectorMultiplicationGPU(double* mat, double* vec, double* res, int n) {
  // determine the number of threads and blocks to use
  int THREADS_PER_BLOCK = std::min(n, 512);
  int BLOCKS_PER_GRID = std::ceil((1.0 * n) / THREADS_PER_BLOCK);
  dim3 BLOCKS_2D(BLOCKS_PER_GRID, BLOCKS_PER_GRID);
  dim3 THREADS_2D(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

  // allocate memory on the device and copy matrix and vector to the device
  double *d_matrix, *d_vec, *d_res;
  cudaMalloc(&d_matrix, sizeof(double) * n * n);
  cudaMalloc(&d_vec, sizeof(double) * n);
  cudaMalloc(&d_res, sizeof(double) * n);
  cudaMemcpy(d_matrix, mat, sizeof(double) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vec, vec, sizeof(double) * n, cudaMemcpyHostToDevice);

  // measure the time spent on the GPU
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  cudaEventRecord(begin, 0);

  // launch the kernel that performs the matrix-vector multiplication
  MatrixVectorMult<<<BLOCKS_2D, THREADS_2D>>>(d_matrix, d_vec, d_res, n);

  // measure the elapsed time and print it
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, begin, end);
  std::cout <<"Time taken for MatVec operation: " << elapsed_time << std::endl;

  // destroy the events and copy the result back to the host
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  cudaMemcpy(res, d_res, sizeof(double) * n, cudaMemcpyDeviceToHost);

  // free memory on the device
  cudaFree(d_matrix);
  cudaFree(d_vec);
  cudaFree(d_res);
}



int main() {

    struct Grid my_grid;
    double * Matrx_A = new double[my_grid.num_rows*my_grid.num_cols*my_grid.num_rows*my_grid.num_cols]();
    double * x = new double[my_grid.num_rows*my_grid.num_cols]();
    double * Vector_b = new double[my_grid.num_rows*my_grid.num_cols]();
/* (Grid& grid, double *matrix_A, double *vector_b)    
*/    
    create_system(my_grid, Matrx_A, Vector_b);

    double* I = new double[my_grid.num_rows*my_grid.num_cols*my_grid.num_rows*my_grid.num_cols]();  // identity matrix
    for (int i = 0; i < my_grid.num_rows*my_grid.num_cols; i++) I[i * my_grid.num_rows*my_grid.num_cols + i] = 1.0;


    InverseGPU(Matrx_A, I, my_grid.num_rows*my_grid.num_cols); // I = Matrx_A ^ (-1)

    MatrixVectorMultiplicationGPU(I, Vector_b, x, my_grid.num_rows*my_grid.num_cols);


    double * u = create_field(my_grid,x);
    //print_field(my_grid,u);

    save_to_file(my_grid,u);

    delete[] Matrx_A;
    delete[] Vector_b;
    delete[] x;
    delete[] u;
    delete[] I;

    return 0;
}
