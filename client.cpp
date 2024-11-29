#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <random>

using namespace std;

struct DataPoint {
    float hours;
    int performance;
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

pair<vector<DataPoint>, vector<DataPoint>> split_dataset(vector<DataPoint>& dataset, double train_ratio = 0.8) {
    shuffle(dataset.begin(), dataset.end(), default_random_engine(42));
    int train_size = static_cast<int>(dataset.size() * train_ratio);
    return make_pair(vector<DataPoint>(dataset.begin(), dataset.begin() + train_size),
                     vector<DataPoint>(dataset.begin() + train_size, dataset.end()));
}

pair<float, float> normalize_data(vector<DataPoint>& dataset) {
    float sum_hours = 0, sum_squared_hours = 0;
    for (const auto& data : dataset) {
        sum_hours += data.hours;
        sum_squared_hours += data.hours * data.hours;
    }
    float mean = sum_hours / dataset.size();
    float std_dev = sqrt(sum_squared_hours / dataset.size() - mean * mean);
    
    for (auto& data : dataset) {
        data.hours = (data.hours - mean) / std_dev;
    }
    return make_pair(mean, std_dev);
}

double compute_rmse(const vector<DataPoint>& dataset, float w, float b) {
    double sum_error = 0.0;
    for (const auto& data : dataset) {
        float y_pred = w * data.hours + b;
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
        
        // Adaptive learning rate
        learning_rate *= 0.99;
    }
    return make_pair(w, b);
}

void write_parameters(const string& filename, float w, float b, float mean, float std_dev) {
    ofstream file(filename);
    file << w << " " << b << " " << mean << " " << std_dev << endl;
}

int main() {
    vector<string> train_filenames = {
        "dataset/trainset_1.txt", "dataset/trainset_2.txt", "dataset/trainset_3.txt",
        "dataset/trainset_4.txt", "dataset/trainset_5.txt", "dataset/trainset_6.txt",
        "dataset/trainset_7.txt", "dataset/trainset_8.txt", "dataset/trainset_9.txt"
    };

    for (int i = 0; i < train_filenames.size(); ++i) {
        string dataset_file = train_filenames[i];
        string output_file = "client_" + to_string(i) + "_params.txt";

        vector<DataPoint> dataset = read_dataset(dataset_file);
        pair<vector<DataPoint>, vector<DataPoint>> split_result = split_dataset(dataset);
        vector<DataPoint>& train_set = split_result.first;
        vector<DataPoint>& val_set = split_result.second;

        pair<float, float> norm_params = normalize_data(train_set);
        float mean = norm_params.first;
        float std_dev = norm_params.second;
        normalize_data(val_set);

        float learning_rate = 0.1;
        int epochs = 10000;
        pair<float, float> model_params = train_model(train_set, learning_rate, epochs);
        float w = model_params.first;
        float b = model_params.second;

        double train_rmse = compute_rmse(train_set, w, b);
        double val_rmse = compute_rmse(val_set, w, b);

        cout << "Client " << i << " - Train RMSE: " << train_rmse << ", Validation RMSE: " << val_rmse << endl;

        write_parameters(output_file, w, b, mean, std_dev);
    }

    return 0;
}