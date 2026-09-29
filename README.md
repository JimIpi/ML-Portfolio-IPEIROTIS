# ML-Portfolio-IPEIROTIS
# Adult Income Prediction: Classical ML vs. Neural Networks

## Introduction
This portfolio project explores whether a complex neural network can outperform a simpler, classical machine learning model on a structured, tabular dataset. The objective is to predict whether an individual earns more than $50,000 a year based on demographic and employment survey data. 

## Data, Preprocessing, and Workflow
**Data:** We used the Adult (Census Income) dataset from OpenML. The dataset contains a mix of numeric features (e.g., age, hours worked per week) and categorical features (e.g., occupation, education).

**Preprocessing:** To handle the mixed data types safely and avoid data leakage, I utilized a scikit-learn `ColumnTransformer`:
*   **Numeric features:** Missing values were imputed using the median, followed by standard scaling.
*   **Categorical features:** Missing values were imputed using the most frequent category, followed by one-hot encoding (dropping unknown categories).

**Models:** 
1.  **Classical Model:** A `LogisticRegression` classifier embedded in a pipeline, evaluated initially using 5-fold cross-validation.
2.  **Neural Network:** A Keras `Sequential` model with two hidden dense layers (64 and 32 units, ReLU activation), a Dropout layer (0.3) to prevent overfitting, and a sigmoid output layer for binary classification. The network was trained using the Adam optimizer, binary crossentropy loss, and an EarlyStopping callback.

## Key Results
Both models were evaluated on the same held-out test set. Because the dataset is imbalanced (the majority earns <=50K), F1-score was used as the primary metric, supported by accuracy and confusion matrices.

*   **Logistic Regression:** F1 Score: ~0.656 | Accuracy: ~85.2%
*   **Neural Network:** F1 Score: ~0.665 | Accuracy: ~85.8%

**Conclusion:** The Neural Network achieved slightly better results, successfully identifying a few more high-income individuals (reducing false negatives). However, given the marginal performance boost, the added complexity and lack of transparency of the neural network might not be justified for this specific tabular dataset, where the simpler Logistic Regression model performed very competitively. *(Note: Please view the attached `.ipynb` notebook to see the detailed confusion matrices and comparison charts).*

## Reflection
*   **What worked well:** Using `Pipeline` and `ColumnTransformer` made the preprocessing workflow extremely clean and robust against data leakage. Additionally, implementing `EarlyStopping` for the neural network successfully prevented overfitting and saved training time.
*   **What was difficult:** Deciding how to evaluate the models fairly was challenging due to the class imbalance. Relying solely on accuracy would have been misleading, making the interpretation of the confusion matrices and F1 score essential.
*   **What could be improved:** In the future, I could experiment with tree-based ensemble models (like Random Forest or XGBoost), which typically excel on tabular data. Additionally, performing a grid search to fine-tune the neural network's hyperparameters (like dropout rate or learning rate) might yield a more significant performance gap.
