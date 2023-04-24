
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <math.h>

using namespace std;

// Define the linear regression model
double linear_regression(double x, double a, double b) {
    return a*x + b;
}

// Define the sum of squared residuals (loss function)
double sum_squared_residuals(vector<double>& x, vector<double>& y, double a, double b) {
    double sum = 0.0;
    for (int i = 0; i < x.size(); i++) {
        double r = y[i] - linear_regression(x[i], a, b);
        sum += r * r;
    }
    return sum;
}

// Define the partial derivative of the loss function with respect to a
double d_da(vector<double>& x, vector<double>& y, double a, double b) {
    double sum = 0.0;
    for (int i = 0; i < x.size(); i++) {
        double r = y[i] - linear_regression(x[i], a, b);
        sum += r * (-x[i]);
    }
    return 2.0 * sum;
}

// Define the partial derivative of the loss function with respect to b
double d_db(vector<double>& x, vector<double>& y, double a, double b) {
    double sum = 0.0;
    for (int i = 0; i < x.size(); i++) {
        double r = y[i] - linear_regression(x[i], a, b);
        sum += r * (-1.0);
    }
    return 2.0 * sum;
}

// Define the gradient descent function
void gradient_descent(vector<double>& x, vector<double>& y, double& a, double& b, double learning_rate, int num_iterations) {
    for (int i = 0; i < num_iterations; i++) {
        double a_gradient = d_da(x, y, a, b);
        double b_gradient = d_db(x, y, a, b);
        a -= learning_rate * a_gradient;
        b -= learning_rate * b_gradient;
    }
}

// Define the mean squared error function
double mean_squared_error(vector<double>& x, vector<double>& y, double a, double b) {
    double sum = 0.0;
    for (int i = 0; i < x.size(); i++) {
        double r = y[i] - linear_regression(x[i], a, b);
        sum += r * r;
    }
    return sum / x.size();
}

int main() {
    // Set up the sample parameters
    double a = 2.0;
    double b = 1.0;
    int n = 100;
    double noise_stddev = 0.1;
    double learning_rate = 0.1;
    int num_iterations = 100;

    // Generate the sample data
    vector<double> x(n);
    vector<double> y(n);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    normal_distribution<double> distribution(0.0, noise_stddev);
    for (int i = 0; i < n; i++) {
        x[i] = (double)i / (double)n;
        y[i] = linear_regression(x[i], a, b) + distribution(generator);
    }

    // Perform the gradient descent optimization
    gradient_descent(x, y, a, b, learning_rate, num_iterations);

    // Compute the mean squared error of the optimized model
    double mse = mean_squared_error(x, y, a, b);

    // Print the results
    cout << "Sample parameters:" << endl;
    cout << "a = " << a << ", b = " << b << ", n = " << n << ", noise_stddev = " << noise_stddev << endl;
    cout << "Computed parameters:" << endl;
    cout << "a_hat = " << a << ", b_hat = " << b << endl;
    cout << "Mean squared error = " << mse << endl;

    return 0;
}