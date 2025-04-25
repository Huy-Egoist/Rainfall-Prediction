# Weather-Prediction-using-Machine-Learning

Machine learning models like Logistic Regression and Decision Trees can effectively analyze atmospheric patterns to predict weather conditions. By training on historical meteorological data, these models learn complex relationships between weather features to forecast outcomes like precipitation probability with high accuracy.

We implement and compare Logistic Regression and Decision Tree classifiers to predict rain occurrence based on multi-year weather observation data.

![weather-prediction-header](https://i.pinimg.com/736x/a4/8c/3e/a48c3eddd75956b8cdf9f6b465df605d.jpg)

## The Dataset

This project uses the [Weather Dataset from Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) containing daily weather observations from multiple Australian locations. Key features include:

| Feature | Description |
|---|---|
| Date | Date of observation (YYYY-MM-DD) |
| Location | Geographic location name |
| MinTemp | Minimum temperature (°C) |
| MaxTemp | Maximum temperature (°C) |
| Rainfall | Precipitation amount (mm) |
| WindGustDir | Wind gust direction |
| WindSpeed | Wind speed (km/h) |
| Humidity | Relative humidity (%) |
| Pressure | Atmospheric pressure (hPa) |
| CloudCover | Cloud coverage (oktas) |
| Temp9am | 9 AM temperature (°C) |
| RainToday | Binary rain indicator (Yes/No) |
| RainTomorrow | Target variable: Next day rain prediction |

**Note:** The target variable `RainTomorrow` is binary (1 = Rain expected, 0 = No rain).

## Model Implementation

### Logistic Regression Approach
- Handles binary classification through sigmoid probability estimation
- Key parameters:
  - Penalty (L1/L2 regularization)
  - Solver optimization method
  - Class weight balancing

### Decision Tree Approach
- Non-linear modeling through recursive data partitioning
- Key parameters:
  - Max tree depth
  - Minimum samples per leaf
  - Splitting criterion (Gini/Entropy)

We implement both models with hyperparameter tuning and compare performance using:
- Accuracy scores
- ROC-AUC metrics
- Confusion matrices
- Training time efficiency

Key findings:
- Decision Trees achieve higher accuracy but require more training time
- Logistic Regression shows better computational efficiency
- Both models benefit from feature engineering of temporal patterns

## How to Use
1. Clone repository
2. Install requirements
3. Run analysis: .ipynb
