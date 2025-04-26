
---

# Car Price Prediction and Analysis

This project focuses on the development of a machine learning model to predict used car selling prices based on historical data. It includes comprehensive data cleaning, feature engineering, exploratory data analysis, model building, and evaluation to derive actionable insights and improve price estimation accuracy.

---

## Project Structure

```
car-price-prediction/
│
├── car_prices.csv               # Raw dataset
├── sales_cleaned.csv             # Cleaned and processed dataset
├── car_price_prediction.ipynb    # Main project notebook
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```

---

## Objectives

- Conduct data preprocessing, including missing value treatment and outlier management.
- Engineer new features to enhance model performance.
- Perform exploratory data analysis (EDA) to identify key sales trends.
- Develop and evaluate machine learning models (Random Forest, XGBoost) for price prediction.
- Provide recommendations for future improvements.

---

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/car-price-prediction.git
   cd car-price-prediction
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(Alternatively, install manually: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost)*

3. Launch the Jupyter Notebook and execute `car_price_prediction.ipynb`.

---

## Dataset Overview

- **Source**: Kaggle car_prices dataset
- **Key Attributes**:
  - `year`, `make`, `model`, `trim`, `sellingprice`
  - `odometer`, `condition`, `color`, `state`, `saledate`
  
The dataset consists of detailed records of used car sales, capturing vehicle specifications, sale prices, and transaction dates across various U.S. states.

---

## Data Preprocessing

- **Missing Values**:
  - Imputed missing values in `transmission` using the mode.
  - Filled missing `odometer` readings using the median.
- **Feature Engineering**:
  - `age` = 2025 - `year`
  - `mileage_per_year` = `odometer` / `age`
  - `is_luxury` (binary indicator for luxury brands such as BMW, Audi, Lexus)
- **Outlier Treatment**:
  - Applied percentile capping for `odometer` and `sellingprice`.
  - Removed records with extremely unrealistic values (e.g., `odometer > 500,000 miles`, `sellingprice < $100`).

---

## Exploratory Data Analysis

- Analyzed total sales by U.S. state and car make.
- Visualized the relationship between odometer readings and selling price.
- Identified factors contributing most significantly to pricing variations.

---

## Machine Learning Approach

Two supervised learning models were developed and compared:

| Model              | R² Score | MAE  | RMSE |
|--------------------|----------|------|------|
| Random Forest Regressor | 0.702 | 3329.7  | 5236.08  |
| XGBoost Regressor       | 0.863 | 2225.05  | 3355.3  |

### Process:
- Label-encoded categorical features.
- Split data into training and testing subsets (80/20 split).
- Evaluated models using:
  - Coefficient of Determination (R²)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)

---

## Key Functions

| Function | Purpose |
|:---|:---|
| `feature_engineering(df)` | Performs feature engineering and handles outliers. |
| `prepare_data_for_modeling(df, categorical_features)` | Prepares and encodes the dataset for modeling. |
| `train_and_evaluate_model(X_train, y_train, X_test, y_test, model)` | Trains the model and evaluates its performance. |
| `main()` | Orchestrates the entire workflow. |

---

## Example Prediction

The trained XGBoost model was used to predict the selling price of a **2022 BMW X5** with **25,000 miles** and **automatic transmission**, demonstrating its practical application.

---

## Future Enhancements

- Conduct hyperparameter optimization to further improve model performance.
- Deploy the model via a web application interface (e.g., using Streamlit or Flask).
- Incorporate additional variables such as fuel type, engine displacement, and drivetrain information.
- Explore advanced ensemble models such as LightGBM and CatBoost.

---

## Acknowledgements

- Data sourced from Kaggle.
- This project utilizes open-source Python libraries, including Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and XGBoost.

---

## Contact

For further information or collaboration opportunities, please reach out:

- **Name:** Ujwala Padala  
- **Email:** ujwalapadala1001@gmail.com    

---

# End of Document

---

---


