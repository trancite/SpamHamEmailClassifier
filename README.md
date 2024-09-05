# Email Classification: Spam vs. Ham using Machine Learning Models and Statistical Preprocessing Techniques

This project involves developing a machine learning model to classify emails as either Spam or Ham, utilizing various preprocessing techniques and machine learning models to find the optimal combination.

## Project Overview
- **Objective**: Develop a classification system for emails (Spam vs. Ham) by exploring and optimizing a variety of machine learning models and text preprocessing techniques to determine the most effective combination.
- **Datasets**: Two labeled email datasets (Spam and Ham) were utilized. One for training and validation, and the other reserved for testing the final model.

## Steps and Methodology

1. **Data Preparation**
    - Downloaded and uncompressed two datasets in CSV format. One dataset is used for training and validation, and the other for testing.
    - Removed duplicate instances and excluded overlapping data between the two datasets.

2. **Brief Statistical Analysis**
    - Performed a label distribution analysis to assess class imbalance and identify potential areas for performance enhancement.
    - Created a baseline "silly model" for performance comparison.

3. **Text Cleaning**
    - Developed a `TextCleaner` class to remove email subjects and normalize text.
    - Added a new column "mails_cleaned" in the dataset containing the processed text.

4. **Text Preprocessing Tools**
    - Implemented and tuned three preprocessing tools:
      - TF-IDF Vectorizer
      - Count Vectorizer
      - Hashing Vectorizer

5. **Clustering Analysis: Unsupervised Learning (Dataset 1)**
    - Applied KMeans clustering with various preprocessing tools to divide the dataset into clusters.
    - Determined the optimal number of clusters (ranging from 2 to 20) based on Silhouette scores.
    - Found that the Hashing Vectorizer provided the best clustering results for spam-ham classification, outperforming the baseline model. The Count Vectorizer also showed promise but was less effective.

6. **Model Tuning**
    - Selected and tuned several models using GridSearchCV with the TF-IDF Vectorizer:
      - SVC
      - Extra Trees
      - Logistic Regression
      - Random Forest
      - K-Neighbors
    - Achieved high scores with optimized parameters, indicating potential overfitting. Evaluated performance on a separate validation set to mitigate this risk.
    - Compared models with different preprocessing tools and selected the best-performing combination for each preprocessing technique.
    Combined the top three models using a Voting Classifier, resulting in nearly perfect validation scores.

7. **Final Testing: Using the Second Dataset**
    - Applied the same text cleaning process to the second dataset and tested the final model.
    - Although performance was slightly lower compared to the training set, it remained very strong.

## Requirements
- Python 3.11.5
- Pandas
- Scikit-learn
- NumPy
- re
- string
- unicodedata
- nltk
- matplotlib (optional)
- seaborn (optional)
- scipy
- jupyter



Note: This project was entirely developed by me without contributions from others. I would appreciate being cited if someone uses this work in their projects, but I don’t mind—everyone is free to use and reference this project. If anyone wants to collaborate or ask me a question, feel free to reach out.

## References
1. Kaggle Spam Email Dataset: [Dataset 1](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)
                              [Dataset 2](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset)
2. Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.

3. TensorFlow. (2023). *TensorFlow Documentation*. Retrieved from https://www.tensorflow.org
