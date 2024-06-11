#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>  
using namespace std;
using namespace Eigen;

// Function to standardize data
MatrixXd standardize(const MatrixXd& data) {
    cout << "Original Data:\n" << data << endl;
    MatrixXd standardized = data;
    VectorXd mean = data.colwise().mean();
    VectorXd stddev = ((data.rowwise() - mean.transpose()).array().square().colwise().sum() / (data.rows() - 1)).sqrt();
    standardized = (data.rowwise() - mean.transpose()).array().rowwise() / stddev.transpose().array();
    cout << "Standardized Data:\n" << standardized << endl;
    return standardized;
}

// PCA Function
MatrixXd pca(const MatrixXd& data, int n_components) {
    // Step 1: Standardize the Data
    MatrixXd standardized_data = standardize(data);

    // Step 2: Compute the Covariance Matrix
    MatrixXd cov_matrix = (standardized_data.transpose() * standardized_data) / (standardized_data.rows() - 1);
    cout << "Covariance Matrix:\n" << cov_matrix << endl;

    // Step 3: Compute Eigenvectors and Eigenvalues
    SelfAdjointEigenSolver<MatrixXd> eig(cov_matrix);
    VectorXd eigenvalues = eig.eigenvalues();
    MatrixXd eigenvectors = eig.eigenvectors();
    cout << "Eigenvalues:\n" << eigenvalues << endl;
    cout << "Eigenvectors:\n" << eigenvectors << endl;

    // Step 4: Sort Eigenvalues and Eigenvectors
    vector<pair<double, VectorXd>> eigen_pairs;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        eigen_pairs.push_back(make_pair(eigenvalues(i), eigenvectors.col(i)));
    }
    sort(eigen_pairs.rbegin(), eigen_pairs.rend(), [](const pair<double, VectorXd>& a, const pair<double, VectorXd>& b) {
        return a.first > b.first;
    });

    MatrixXd sorted_eigenvectors(eigenvectors.rows(), n_components);
    for (int i = 0; i < n_components; ++i) {
        sorted_eigenvectors.col(i) = eigen_pairs[i].second;
    }
    cout << "Sorted Eigenvectors (Principal Components):\n" << sorted_eigenvectors << endl;

    // Step 5: Project Data
    MatrixXd transformed_data = standardized_data * sorted_eigenvectors;
    cout << "Transformed Data:\n" << transformed_data << endl;
    return transformed_data;
}

// Function to print a MatrixXd
void printMatrix(const MatrixXd& matrix) {
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            cout << matrix(i, j) << " ";
        }
        cout << endl;
    }
}

int main() {
    // Example data (10 samples, 2 features)
    MatrixXd data(10, 2);
    data << 2.5, 2.4,
            0.5, 0.7,
            2.2, 2.9,
            1.9, 2.2,
            3.1, 3.0,
            2.3, 2.7,
            2.0, 1.6,
            1.0, 1.1,
            1.5, 1.6,
            1.1, 0.9;

    int n_components = 1;  // Reduce to 1 principal component
    MatrixXd transformed_data = pca(data, n_components);

    cout << "Final Transformed Data:\n";
    printMatrix(transformed_data);

    return 0;
}