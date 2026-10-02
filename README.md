# Titanic Survival Prediction

A machine learning classification project that predicts whether a passenger survived the Titanic disaster using passenger demographic and travel information.

## Project Overview

This project compares two supervised machine learning algorithms:

* Random Forest Classifier
* Logistic Regression

The models use preprocessing pipelines to handle numerical and categorical variables, missing values, feature scaling, and one-hot encoding.

Hyperparameter tuning is performed using GridSearchCV with stratified 5-fold cross-validation.

## Dataset

The project uses the Titanic dataset provided through Seaborn.

The features used include:

* Passenger class (`pclass`)
* Sex
* Age
* Number of siblings/spouses aboard (`sibsp`)
* Number of parents/children aboard (`parch`)
* Fare
* Passenger class category
* Passenger type (`who`)
* Adult male indicator
* Whether the passenger was travelling alone

The target variable is:

```text
survived
```

where:

* `0` = Did not survive
* `1` = Survived

## Machine Learning Workflow

The project follows this workflow:

1. Load the Titanic dataset
2. Examine the target class distribution
3. Split the dataset into training and testing sets
4. Identify numerical and categorical features
5. Handle missing numerical values using median imputation
6. Handle missing categorical values using most-frequent imputation
7. Standardize numerical features
8. One-hot encode categorical features
9. Build a preprocessing and classification pipeline
10. Perform hyperparameter tuning using GridSearchCV
11. Evaluate the model on unseen test data
12. Generate classification reports
13. Generate confusion matrices
14. Analyze feature importance
15. Train and evaluate Logistic Regression
16. Compare the two classification approaches

## Models

### Random Forest

The Random Forest model is tuned using:

* Number of estimators
* Maximum tree depth
* Minimum samples required for splitting

The model's feature importance scores are also analyzed to identify which variables contributed most to its predictions.

### Logistic Regression

Logistic Regression is tuned using:

* Solver
* Regularization penalty
* Class weighting

The absolute magnitude of the model coefficients is visualized to examine the relative contribution of the encoded features.

## Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

The test set is kept separate from model training and hyperparameter tuning.

## Project Structure

```text
titanic-survival-prediction/
│
├── README.md
├── requirements.txt
├── titanic_survival_prediction.py
├── outputs/
│   ├── random_forest_confusion_matrix.png
│   ├── random_forest_feature_importance.png
│   ├── logistic_regression_confusion_matrix.png
│   └── logistic_regression_coefficients.png
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/titanic-survival-prediction.git
cd titanic-survival-prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python titanic_survival_prediction.py
```

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

## Key Concepts Demonstrated

This project demonstrates practical use of:

* Data preprocessing
* Missing-value imputation
* Feature scaling
* One-hot encoding
* Train/test splitting
* Machine learning pipelines
* Cross-validation
* Hyperparameter tuning
* Random Forest classification
* Logistic Regression
* Model evaluation
* Feature importance analysis
* Confusion matrices
