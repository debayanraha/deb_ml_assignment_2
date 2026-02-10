# Mobile Price Classification using Machine Learning Models

## a. Problem Statement

The objective of this assignment is to design, implement, and evaluate multiple supervised machine learning classification models to predict the price range of mobile phones based on their technical specifications. The problem is formulated as a multi-class classification task, where the target variable represents different price categories. Multiple classical and ensemble-based machine learning algorithms are implemented and compared using standard evaluation metrics.

This classification helps retailers and manufacturers understand which hardware configurations justify specific price points in a competitive market.

This assignment demonstrates the importance of model selection, preprocessing, and evaluation in multi-class classification problems.

---

## b. Dataset Description

The dataset used for this assignment is the **Mobile Price Classification Dataset** sourced from Kaggle.
The dataset consists of 2,000 instances with 21 major features. 
**Key characteristics:**
- Each record represents a mobile phone.
- Input features include battery power, RAM, internal memory, screen resolution, number of cores, connectivity features, etc.
- All features are numerical.
- Target variable: `price_range` with four classes:
  - 0 – Low Cost
  - 1 – Medium Cost
  - 2 – High Cost
  - 3 – Very High Cost

---

## c. Models Statistics

### Evaluation Metrics Used
- Accuracy
- AUC (Weighted)
- Precision (Weighted)
- Recall (Weighted)
- F1 Score (Weighted)
- Matthews Correlation Coefficient (MCC)

### Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC | Observation |
|--------|----------|-----|-----------|--------|----|-----|------------|
| Logistic Regression | 0.950 | 0.9978 | 0.9503 | 0.9500 | 0.9497 | 0.9336 | **Excellent performance.** Strong linear separability with excellent performance after scaling |
| Decision Tree | 0.670 | 0.8444 | 0.6625 | 0.6700 | 0.6653 | 0.5606 | **Underperforming.** Moderate performance, prone to overfitting |
| kNN | 0.590 | 0.8149 | 0.6043 | 0.5900 | 0.5909 | 0.4560 | **Worst performer.** Sensitive to scaling and dimensionality. Distance-based logic is likely hindered by the varying scales of features. |
| Naive Bayes | 0.805 | 0.9376 | 0.8086 | 0.8050 | 0.8066 | 0.7401 | **Decent baseline.** Efficient and robust despite independence assumption. Reliable but slightly outmatched by more complex ensemble methods. |
| Random Forest | 0.920 | 0.9879 | 0.9235 | 0.9200 | 0.9210 | 0.8938 | **Very Strong.** Strong ensemble performance with reduced overfitting. Effectively handles feature importance, particularly RAM and Battery power. |
| XGBoost | 0.955 | 0.9979 | 0.9554 | 0.9550 | 0.9551 | 0.9400 | **Top Performer.** Best overall performance with superior generalization. Provides the most robust and accurate predictions across all metrics. |

---

## Conclusion

Ensemble models, especially **XGBoost**, achieved the highest predictive performance across all metrics. Logistic Regression also performed remarkably well, indicating strong linear patterns in the data. kNN showed comparatively weaker results due to sensitivity to feature space dimensionality.


### Final Observations
* **Best Model**: **XGBoost** achieved the highest accuracy (95.5%) and MCC (0.94), making it the most reliable model for production.
* **The "RAM" Effect**: Across all models, RAM was observed to be the most influential feature in determining the price category.
* **Efficiency**: While XGBoost is the most accurate, **Logistic Regression** offers a surprisingly competitive alternative with much lower computational overhead.



