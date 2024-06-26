#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <omp.h>

using namespace std;
using namespace Eigen;

// Read data from CSV file
MatrixXd read_data(const string& filename, int rows, int cols) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file");
    }
    MatrixXd data(rows, cols);
    string line;
    int row = 0;
    while (getline(file, line) && row < rows) {
        stringstream lineStream(line);
        string cell;
        int col = 0;
        while (getline(lineStream, cell, ',') && col < cols) {
            data(row, col) = stod(cell);
            col++;
        }
        row++;
    }
    return data;
}

// Standardize data
MatrixXd standardize(const MatrixXd& data) {
    MatrixXd standardized = data;
    VectorXd mean = data.colwise().mean();
    VectorXd stddev = ((data.rowwise() - mean.transpose()).array().square().colwise().sum() / (data.rows() - 1)).sqrt();
    standardized = (data.rowwise() - mean.transpose()).array().rowwise() / stddev.transpose().array();
    return standardized;
}

// PCA Function with parallel covariance matrix calculation
MatrixXd pca_cov(const MatrixXd& data, int n_components) {
    MatrixXd standardized_data = standardize(data);
    MatrixXd cov_matrix(data.cols(), data.cols());

    // Parallel covariance matrix calculation
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < data.cols(); ++i) {
        for (int j = i; j < data.cols(); ++j) {
            cov_matrix(i, j) = (standardized_data.col(i).array() * standardized_data.col(j).array()).mean();
            cov_matrix(j, i) = cov_matrix(i, j);
        }
    }

    // Eigenvalue decomposition
    SelfAdjointEigenSolver<MatrixXd> eig(cov_matrix);
    VectorXd eigenvalues = eig.eigenvalues();
    MatrixXd eigenvectors = eig.eigenvectors();

    // Sort eigenvalues and eigenvectors
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

    return sorted_eigenvectors;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <datafile> <rows> <cols>" << endl;
        return 1;
    }

    string datafile = argv[1];
    int rows = stoi(argv[2]);
    int cols = stoi(argv[3]);

    MatrixXd data = read_data(datafile, rows, cols);
    int n_components = 1;

    // Measure time for parallel covariance version
    auto start_par = chrono::high_resolution_clock::now();
    MatrixXd transformed_data_par = pca_cov(data, n_components);
    auto end_par = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_par = end_par - start_par;
    cout << "PCA with parallel covariance calculation took " << duration_par.count() << " seconds." << endl;

    return 0;
}
