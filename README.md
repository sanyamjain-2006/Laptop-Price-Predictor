# Laptop Price Predictor 💻📉

An end-to-end Machine Learning project that processes raw laptop specifications, applies data cleaning and advanced feature engineering, and uses a gradient boosting model to predict laptop retail prices.

## 🚀 Project Overview

Predicting laptop prices can be complex due to the varying combinations of hardware specs like RAM, storage types (SSD/HDD), screen resolutions, and processor brands. This project takes raw scraped laptop data (`laptop_data.csv`), cleans structural anomalies, engineered highly correlated features, and trains an **XGBoost Regressor** wrapped in a structured Scikit-Learn `Pipeline` to estimate prices dynamically.

## 🛠️ Key Features & Data Engineering

The pipeline performs deep feature engineering to optimize model performance:
* **String Parsing:** Strips unwanted characters from variables (e.g., extracting numerical values from `Ram` and `Weight`).
* **Screen Specification Extraction:** Extracts `Touchscreen` and `IPS` panel flags, calculates Pixels Per Inch (**PPI**), and discards redundant raw resolution dimensions.
* **Hardware Categorization:** Groups diverse CPU and GPU lines into major brand blocks (Intel Core i3/i5/i7, AMD, Nvidia, etc.) and filters out low-frequency noise (e.g., ARM processors).
* **Storage Decomposition:** Leverages regex and text parsing to cleanly split compound storage strings into individual numerical representations for `HDD` and `SSD` capacities.
* **Target Transformation:** Applies a log transformation ($y = \log(\text{Price})$) to stabilize variance and normalize target price distributions.

## 🏗️ Model Architecture

The project builds a clean, deployment-ready Scikit-Learn `Pipeline` containing:
1.  **ColumnTransformer / OneHotEncoder:** Encodes categorical features seamlessly while robustly handling unknown test features using `handle_unknown='ignore'`.
2.  **XGBoost Regressor:** Utilizes tuned ensemble gradient boosting (`n_estimators=45`, `max_depth=5`) for low-error regression outputs.

## 📊 Performance Metrics

The model evaluates prediction accuracy using two core validation metrics:
* **R² Score:** Measures the proportion of variance explained by the model features.
* **Mean Absolute Error (MAE):** Identifies the average magnitude of prediction errors in the log-transformed space.

## 📂 Project Structure

```text
├── laptop_data.csv     # Raw dataset
├── notebook.ipynb       # Data Cleaning, Feature Engineering & Modeling
├── df.pkl               # Pickled pandas DataFrame for UI reference
├── Model.pkl            # Pickled Scikit-Learn trained Pipeline
└── README.md            # Project documentation
