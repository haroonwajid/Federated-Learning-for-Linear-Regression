#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>

using namespace std;

struct DataPoint {
    float hours;
    int performance;
};

struct ModelParams {
    float w, b, mean, std_dev;
};

vector<DataPoint> read_dataset(const string& filename) {
    vector<DataPoint> dataset;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        float hours;
        int performance;
        if (iss >> hours >> performance) {
            dataset.push_back({hours, performance});
        }
    }
    return dataset;
}

ModelParams read_parameters(const string& filename) {
    ifstream file(filename);
    ModelParams params;
    file >> params.w >> params.b >> params.mean >> params.std_dev;
    return params;
}

double compute_rmse(const vector<DataPoint>& dataset, const ModelParams& params) {
    double sum_error = 0.0;
    for (const auto& data : dataset) {
        float normalized_hours = (data.hours - params.mean) / params.std_dev;
        float y_pred = params.w * normalized_hours + params.b;
        sum_error += pow(y_pred - data.performance, 2);
    }
    return sqrt(sum_error / dataset.size());
}

pair<float, float> train_model(const vector<DataPoint>& dataset, float learning_rate, int epochs) {
    float w = 0.0, b = 0.0;
    int n = dataset.size();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float dw = 0.0, db = 0.0;
        for (const auto& data : dataset) {
            float y_pred = w * data.hours + b;
            dw += (y_pred - data.performance) * data.hours;
            db += (y_pred - data.performance);
        }
        w -= learning_rate * dw / n;
        b -= learning_rate * db / n;
    }
    return make_pair(w, b);
}

int main() {
    vector<string> train_filenames = {
        "dataset/trainset_1.txt", "dataset/trainset_2.txt", "dataset/trainset_3.txt",
        "dataset/trainset_4.txt", "dataset/trainset_5.txt", "dataset/trainset_6.txt",
        "dataset/trainset_7.txt", "dataset/trainset_8.txt", "dataset/trainset_9.txt"
    };
    vector<string> client_files = {
        "client_0_params.txt", "client_1_params.txt", "client_2_params.txt",
        "client_3_params.txt", "client_4_params.txt", "client_5_params.txt",
        "client_6_params.txt", "client_7_params.txt", "client_8_params.txt"
    };
    string test_filename = "dataset/testset_10.txt";

    // Combine all training datasets
    vector<DataPoint> combined_dataset;
    for (const auto& filename : train_filenames) {
        vector<DataPoint> dataset = read_dataset(filename);
        combined_dataset.insert(combined_dataset.end(), dataset.begin(), dataset.end());
    }

    // Train centralized model
    float learning_rate = 0.01;
    int epochs = 1000;
    pair<float, float> centralized_params = train_model(combined_dataset, learning_rate, epochs);
    float w_centralized = centralized_params.first;
    float b_centralized = centralized_params.second;

    // Read and average client parameters for federated model
    vector<ModelParams> client_params;
    for (const auto& file : client_files) {
        client_params.push_back(read_parameters(file));
    }

    ModelParams global_params = {0, 0, 0, 0};
    for (const auto& params : client_params) {
        global_params.w += params.w;
        global_params.b += params.b;
        global_params.mean += params.mean;
        global_params.std_dev += params.std_dev;
    }
    global_params.w /= client_params.size();
    global_params.b /= client_params.size();
    global_params.mean /= client_params.size();
    global_params.std_dev /= client_params.size();

    // Read test dataset
    vector<DataPoint> testset = read_dataset(test_filename);

    // Compute RMSE for federated model
    double federated_rmse = compute_rmse(testset, global_params);
    cout << "Federated Learning Global Model RMSE: " << federated_rmse << endl;

    // Compute RMSE for centralized model
    ModelParams centralized_model_params = {w_centralized, b_centralized, 0, 1}; // No normalization for centralized model
    double centralized_rmse = compute_rmse(testset, centralized_model_params);
    cout << "Centralized Model RMSE: " << centralized_rmse << endl;

    return 0;
}