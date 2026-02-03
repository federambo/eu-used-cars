# Used Car Price Prediction 

This repository contains a complete machine learning pipeline designed to predict the market price of used cars. The project covers everything from data ingestion to model deployment using Scikit-Learn pipelines.

## Project Overview
The goal of this project is to estimate vehicle prices based on technical features and usage history. This is achieved through:
1.  **Robust Cleaning:** Identifying and removing anomalies in mileage and pricing.
2.  **Feature Engineering:** Processing categorical variables (Brand, Model, Fuel Type) and scaling numerical data.
3.  **Machine Learning:** Training a Random Forest Regressor to capture non-linear relationships in the automotive market.

## Dataset Features
The model is trained on data including:
* **Numerical:** Mileage (km), Year, Engine Power, etc.
* **Categorical:** Make, Model, Fuel Type and Transmission.
* **Target:** `Price` (The valuation of the vehicle).

## Tech Stack
* **Python** (Pandas, NumPy)
* **Visualization:** Matplotlib & Seaborn
* **Machine Learning:** Scikit-Learn
* **Model Serialization:** Pickle

## Workflow

### 1. Data Cleaning & EDA (`eda_cleaning.ipynb`)
* **Outlier Detection:** Used the **Z-score** method to filter out unrealistic mileage values (e.g., dropping entries with over 9,999,999 km).
* **Data Integrity:** Handled missing values and ensured data consistency across categorical fields.
* **Storage:** The cleaned dataframe is shared/exported for the modeling phase.

### 2. Model Training & Evaluation (`model_training_and_evaluation.ipynb`)
* **Pre-processing Pipeline:** * `StandardScaler` for numerical features.
    * `OneHotEncoder` / `OrdinalEncoder` for categorical features via `ColumnTransformer`.
* **Algorithm:** **Random Forest Regressor** was selected for its high performance and ability to handle complex feature interactions.
* **Persistence:** The final trained model is saved using `pickle` for future inference.

## Results
The model shows a strong fit, as evidenced by the **Actual vs. Predicted** scatter plot included in the evaluation notebook. The regression line follows the data distribution closely, indicating high predictive accuracy for the majority of the price range.

## Project Structure
```bash
.
├── data
│   └── fullGas.csv
├── notebooks
│   └── eda_cleaning.ipynb
│   └── model_training_and_evaluation.ipynb
└── README.md

```

## How to use

Clone this repository.

Install dependencies: pip install pandas scikit-learn matplotlib seaborn.

Run eda_cleaning.ipynb to process the raw data.

Run model_training_and_evaluation.ipynb to train the model and see results.

## Data Source
The raw data used in this project was sourced from **Kaggle**: 
**[Cars Europe Dataset](https://www.kaggle.com/datasets/alemazz11/cars-europe)** by user *alemazz11*.

**Disclaimer:** This dataset is provided by a third party and is used here for educational and portfolio purposes. I do not claim ownership of the original data collection. All credits go to the original author for gathering and sharing this resource with the community.
                                 