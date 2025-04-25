# Weather-Prediction-using-Machine-Learning

Machine learning models like Logistic Regression and Decision Trees can effectively analyze atmospheric patterns to predict weather conditions. By training on historical meteorological data, these models learn complex relationships between weather features to forecast outcomes like precipitation probability with high accuracy.

We implement and compare Logistic Regression and Decision Tree classifiers to predict rain occurrence based on multi-year weather observation data.

![weather-prediction-header](https://cdn.pixabay.com/photo/2016/05/24/16/48/mountains-1412683_1280.png)

## The Dataset

This project uses the [Weather Dataset from Kaggle](https://static.lag.vn/upload/news/22/11/11/anime-bocchi-the-rock-duoc-yeu-thich-toan-cau-2_WLIW.jpg?w=1200&h=800&crop=pad&scale=both&encoder=wic&subsampling=444) containing daily weather observations from multiple Australian locations. Key features include:

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
3. Run analysis: `.ipynb`
