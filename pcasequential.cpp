#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <fstream>
#include <vector>

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

// PCA Function
MatrixXd pca(const MatrixXd& data, int n_components) {
    MatrixXd standardized_data = standardize(data);
    MatrixXd cov_matrix = (standardized_data.transpose() * standardized_data) / (standardized_data.rows() - 1);

    SelfAdjointEigenSolver<MatrixXd> eig(cov_matrix);
    VectorXd eigenvalues = eig.eigenvalues();
    MatrixXd eigenvectors = eig.eigenvectors();

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

    // Measure time for sequential version
    auto start_seq = chrono::high_resolution_clock::now();
    MatrixXd transformed_data_seq = pca(data, n_components);
    auto end_seq = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_seq = end_seq - start_seq;
    cout << "Sequential version took " << duration_seq.count() << " seconds." << endl;

    return 0;
}
