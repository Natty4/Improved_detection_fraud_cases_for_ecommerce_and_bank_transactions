# Fraud Detection Project

This project implements a fraud detection system for e-commerce and bank credit card transactions using machine learning. It includes data preprocessing, feature engineering, model training (Logistic Regression and XGBoost), evaluation, and model explainability using SHAP.

## Project Structure

- `config/`: Configuration settings (e.g., file paths, model parameters).
- `src/`: Source code for data preprocessing, feature engineering, model training, and explainability.
- `scripts/`: Main script to run the fraud detection pipeline.
- `data/`: Directory for raw and processed data.
- `requirements.txt`: Python dependencies.

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd fraud-detection


2. Install dependencies:
   ```bash
    pip install -r requirements.txt
3. Place the datasets (fraud_data.csv, ipaddress_to_country.csv, creditcard.csv) in the data/raw/ directory.
4. Run the pipeline:
    ```
        python scripts/main.py
    ```
### Outputs

    Precision-Recall curves saved in outputs/.
    SHAP summary and force plots saved in outputs/.
    Model evaluation metrics printed to the console.

### Datasets

    fraud_data.csv: E-commerce transaction data.
    ipaddress_to_country.csv: IP address to country mapping.
    creditcard.csv: Credit card transaction data.

### Requirements

See requirements.txt for the full list of dependencies.
text
---


```
    install:
        pip install -r requirements.txt

    run:
        python scripts/main.py

    clean:
        rm -rf outputs/*

```