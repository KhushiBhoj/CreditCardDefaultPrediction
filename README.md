# Credit Card Default Prediction: Logistic Regression vs Neural Network

## Project Overview

This project aims to predict whether a credit card holder will default on their payment next month using machine learning models. Two approaches are compared:
- Logistic Regression (LR) – a traditional, interpretable linear model
- Neural Network (NN) – a non-linear deep learning model

The focus is on evaluating accuracy, precision, recall, F1-score, ROC-AUC, and the ability to handle imbalanced data.

## Dataset

- The dataset contains credit card clients’ payment records including:
  - Demographic info: Age, Sex, Education, Marriage
  - Financial info: Credit limit, past bill amounts, past payment amounts, payment history
  - Target variable: default.payment.next.month (0 = non-default, 1 = default)
- Class distribution:
  - Non-default (0) → 3504 samples (~78%)
  - Default (1) → 996 samples (~22%)

Challenge: Imbalanced classes → models may bias towards majority class

Data Preprocessing

Train/Validation/Test Split: 70% train, 15% validation, 15% test

Scaling: StandardScaler applied on numerical features after splitting to avoid data leakage

Handling imbalance:

Logistic Regression trained on original data (baseline)

Neural Network trained with SMOTE oversampling to balance classes

Models
Logistic Regression

Scaled numerical features + one-hot categorical features

Evaluated on test set

Results:

Accuracy: 0.81

ROC-AUC: 0.72

Class 0 → Precision: 0.82, Recall: 0.97, F1-score: 0.89

Class 1 → Precision: 0.71, Recall: 0.24, F1-score: 0.35

Confusion Matrix:

[[3407   97]
 [ 761  235]]


Observation:

LR has high overall accuracy and identifies most non-defaulters correctly

Struggles to detect defaulters (low recall for class 1)

Shows the effect of class imbalance

Neural Network

Fully connected feedforward network

Input: scaled numerical + categorical features

SMOTE applied to training data to balance classes

ReLU activation, dropout regularization, sigmoid output

Results:

Accuracy: 0.62

ROC-AUC: 0.68

Class 0 → Precision: 0.90, Recall: 0.58, F1-score: 0.71

Class 1 → Precision: 0.34, Recall: 0.78, F1-score: 0.48

Confusion Matrix:

[[2030 1474]
 [ 220  776]]


Observation:

NN improves recall for defaulters significantly

Lower overall accuracy because many non-defaulters are misclassified

Better at detecting minority class due to SMOTE oversampling

Comparison & Inferences
Metric	Logistic Regression	Neural Network
Accuracy	0.81	0.62
ROC-AUC	0.72	0.68
Class 1 Recall	0.24	0.78
Class 1 Precision	0.71	0.34
Class 0 Recall	0.97	0.58

Key Insights:

Logistic Regression is better for overall accuracy and majority class prediction

Neural Network is better at detecting defaulters (minority class) due to SMOTE

Trade-off: NN sacrifices accuracy on non-defaulters to maximize recall for defaulters

Choice of model depends on business goal:

Minimize false negatives (catch defaulters) → Neural Network

Overall accuracy / simplicity / interpretability → Logistic Regression

Conclusion

In this experiment:

Logistic Regression provides a highly accurate and interpretable baseline, but fails to catch most defaulters due to class imbalance

Neural Network, trained with SMOTE, improves recall for defaulters significantly, though overall accuracy drops

For credit risk management, where catching defaulters is critical, the Neural Network approach is more effective

Future improvements:

Tune NN threshold to optimize precision-recall tradeoff

Try ensemble methods like XGBoost or Random Forest

Feature engineering (payment trends, credit utilization ratios)
