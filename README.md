# Federated Learning for Linear Regression

Federated Learning is a decentralized approach to machine learning, where data remains on edge devices, and only model parameters are shared with a central server. This preserves data privacy and allows collaborative learning across multiple devices, making it highly suitable for industries like healthcare, finance, and mobile technologies.

This project implements **Federated Learning for Linear Regression** with a single independent variable, using **C/C++ client-server processes**. The goal is to explore distributed machine learning, parameter exchange between clients and server, and global model aggregation.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Centralized Linear Regression](#centralized-linear-regression)
3. [Federated Linear Regression](#federated-linear-regression)
4. [Dataset Description](#dataset-description)
5. [Project Steps](#project-steps)
6. [Evaluation](#evaluation)
7. [How to Clone and Run](#how-to-clone-and-run)
8. [Contributing](#contributing)
9. [License](#license)

---
## How to Clone and Run

### Clone the Repository
To get started, clone the repository from GitHub:
```bash
git clone https://github.com/haroonwajid/Federated-Learning-for-Linear-Regression.git
cd Federated-Learning-for-Linear-Regression
```
## How to Run

### Requirements
- C/C++ compiler (e.g., GCC or Clang)

### Steps
1. Compile the client and server programs:
   ```bash
   gcc -o client client.cpp
   gcc -o server server.cpp

## Introduction

In federated learning:
- **Clients** train models locally on their datasets and send model parameters (weights and biases) to the central **server**.
- The **server** aggregates these parameters to create a global model, which is used for predictions.

This project includes:
1. A centralized implementation of linear regression.
2. A federated learning framework with multiple clients and a server.

---

## Centralized Linear Regression

### Objective
Train a single linear regression model on a combined dataset and evaluate its performance using the **Root Mean Squared Error (RMSE)**.

### Steps
1. Combine nine subsets of the dataset into a single training set.
2. Train the model using gradient descent:
   - **Model Form**:  
     \( y = w \cdot x + b \)  
     Where:
     - \( w \): weight  
     - \( b \): bias  
     - \( x \): study hours  
     - \( y \): predicted performance index  
3. Evaluate the model on the tenth subset (test set) using RMSE:
   \[
   RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
   \]
   Where:
   - \( y_i \): actual performance index  
   - \( \hat{y}_i \): predicted performance index  
   - \( N \): number of records  

4. Output the **RMSE** for the test set.

---

## Federated Linear Regression

### Objective
Implement a federated learning framework using multiple clients and a central server to collaboratively train a global linear regression model.

### Client Program
1. Train a local linear regression model on one subset using gradient descent.
2. Send the model parameters (weights and biases) to the server.

### Server Program
1. Receive model parameters from all nine clients.
2. Perform **weighted averaging** to aggregate parameters:
   \[
   w_{global} = \frac{1}{9} \sum_{i=1}^{9} w_i, \quad b_{global} = \frac{1}{9} \sum_{i=1}^{9} b_i
   \]
3. Use the global model to predict performance on the test set.
4. Evaluate the global model using **RMSE**.
5. Output the final **RMSE**.

---

## Dataset Description

The dataset contains 10,000 records of students' performance and study hours, split into 10 subsets:
- **Nine subsets**: Training data  
- **One subset**: Test data  

Each record consists of:
1. **Study Hours**: A floating-point value (e.g., 3.5).
2. **Performance Index**: An integer score (1-100).

### Example
| Study Hours | Performance Index |
|-------------|-------------------|
| 3.5         | 78                |
| 5.0         | 85                |
| 2.2         | 60                |
| 7.1         | 92                |
| 4.3         | 75                |

---

## Project Steps

1. **Centralized Linear Regression**
   - Merge training subsets.
   - Train and evaluate a linear regression model.

2. **Federated Learning Framework**
   - Implement client and server programs:
     - Clients train local models and send parameters.
     - Server aggregates parameters and evaluates the global model.

3. **Evaluation**
   - Test the global model on the test dataset using RMSE.

---

## Evaluation

### Metrics
- **Root Mean Squared Error (RMSE)**: Measures model performance on the test set.

Expected Outputs:
1. RMSE for the centralized model.
2. RMSE for the global federated model.

---
