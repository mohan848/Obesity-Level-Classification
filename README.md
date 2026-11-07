
# Obesity Level Classification Using Machine Learning

## a. Problem Statement

Obesity is one of the major health concerns in recent years, affecting individuals across different age groups and populations. It can increase the risk of several health conditions, including cardiovascular diseases and other disorders. Obesity levels depend on demographic characteristics, eating habits, family history, and physical activity.
<p>Machine learning techniques such as multi-class classification models can be implemented and evaluated using appropriate metrics to classify individuals into one of the seven obesity-level categories by identifying patterns in their demographic characteristics, eating habits, family history, and physical activity.</p>

## b. Dataset Description

**Dataset Name:** Estimation of Obesity Levels Based on Eating Habits and Physical Condition

**Dataset Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

The dataset contains 2,111 instances and 17 attributes, including 16 input features and one target variable.

The input features include demographic information, eating habits, family history, and physical activity-related characteristics.

**Target Variable:** `NObeyesdad`

The target variable consists of seven obesity-level categories:

- Insufficient Weight
- Normal Weight
- Overweight Level I
- Overweight Level II
- Obesity Type I
- Obesity Type II
- Obesity Type III

## c. GitHub Repository Link

https://github.com/mohan848/Obesity-Level-Classification

## d. Models Used and Performance Comparison

Below classification models are used:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbours (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models were evaluated using Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8747 | 0.9834 | 0.8707 | 0.8724 | 0.8708 | 0.8539 |
| Decision Tree | 0.9149 | 0.9488 | 0.9157 | 0.9118 | 0.9133 | 0.9007 |
| KNN | 0.8251 | 0.9650 | 0.8235 | 0.8187 | 0.8131 | 0.7981 |
| Naive Bayes | 0.5083 | 0.8372 | 0.5207 | 0.5002 | 0.4511 | 0.4471 |
| Support Vector Machine | 0.9243 | 0.9929 | 0.9250 | 0.9220 | 0.9230 | 0.9118 |

*Note: Precision, Recall, F1-score, and AUC are macro-averaged metrics. AUC is calculated using the One-vs-Rest approach.*

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved good results, with an accuracy of 87.47% and an F1-score of 87.08%. Overall, it performed well but below SVM and Decision Tree. |
| Decision Tree | Achieved better results, with an accuracy of 91.49% and an F1-score of 91.33%. Its performance is very close to SVM classifier, showing that it is also an effective classifier for this dataset. |
| KNN | Achieved moderate performance, with 82.51% accuracy and an F1-score of 81.31%, showing lower classification effectiveness. |
| Naive Bayes | Recorded the lowest performance across all metrics when compared to models, with 50.83% accuracy and an F1-score of 45.11%, showing the difficulty in accurately classifying between the obesity-level categories. |
| Support Vector Machine | Achieved the best overall performance with an accuracy of 92.43% and F1-score of 92.30% and highest scores across all other evaluation metrics. This shows that SVM is the most effective classifier for this dataset. It also shows great balance between precision and recall scores. |

### Overall Winner for the Dataset

**Support Vector Machine (SVM)**

<p>Based on all evaluation metrics, Support Vector Machine is the overall winner for the obesity classification dataset. It achieved highest accuracy, AUC, precision, recall, F1 score and MCC among all five models showing that it's able to accurately classify between the obesity-level categories. F1-score of 92.30% shows a strong balance between precision and recall indicating that the model effectively minimizes both false positives and false negatives across the seven obesity categories.</p>

## e. Conclusion
Overall, five machine learning classification models are implemented to classify individuals into various obesity-level categories. The models were evaluated using appropriate performance metrics, with Support Vector Machine achieving the best overall performance. Then an interactive Streamlit application was also developed to evaluate model performance using trained models.
