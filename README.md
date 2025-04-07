Fake Job Prediction Model


Overview
This project is focused on detecting fraudulent job postings using natural language processing (NLP) and machine learning techniques. It uses the fake_job_postings.csv dataset, which contains various job-related attributes including job descriptions, company profiles, required education, and more. The objective is to classify job postings as either fraudulent or legitimate.

The project involves text preprocessing, TF-IDF feature extraction, dataset balancing using SMOTE, and training multiple classification models. These models are then evaluated using standard metrics to determine their performance in identifying fraudulent listings.


Libraries Used
The following Python libraries are used in this project:

Pandas: For data manipulation and analysis

NumPy: For numerical operations

Matplotlib and Seaborn: For data visualization and plotting

Scikit-learn: For machine learning model building, evaluation, and text vectorization (used for Multinomial Naive Bayes, Random Forest, and SVM)

NLTK: For natural language processing and stop word removal

Imbalanced-learn: For class balancing using SMOTE

XGBoost: For implementing the XGBoost classifier

WordCloud: For generating visual representations of the most frequent words


Dataset
The dataset used in this project is called fake_job_postings.csv and is available on Kaggle:

🔗 Real or Fake - Fake Job Posting Prediction Dataset

Preprocessing Steps
Missing Value Handling
Categorical columns: Filled with the string "Unknown"

Numerical columns: (if any) filled with median values

Text Data Processing
Several textual columns (e.g., title, location, company_profile, description) are concatenated into a single text column

Stop words are removed from the text data using NLTK

Feature Encoding
Categorical columns such as required_experience, required_education, and employment_type are label-encoded for model compatibility

Text Vectorization
The text column is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) to convert the raw text into numerical features

Class Balancing
The dataset is imbalanced, so SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the target classes


Model Training and Evaluation

Models Used

Multinomial Naive Bayes

Random Forest Classifier

XGBoost Classifier

Support Vector Machine (SVM)


Evaluation Metrics

Accuracy

Classification Report (Precision, Recall, F1-score)

Confusion Matrix

Each model is trained and evaluated using the same preprocessing pipeline, and their results are compared to determine the best-performing approach.

Visualizations

Confusion Matrices: Show the classification results for each model, including true positives, false positives, true negatives, and false negatives

ROC Curves: Receiver Operating Characteristic curves are plotted for each model to visualize their performance in terms of the trade-off between true positive rate and false positive rate


Files

fake_job_postings.csv: The dataset containing job posting information

Fake job prediction.ipynb: The Jupyter notebook containing the complete code for the project


Instructions to Run the Project
Clone this repository to your local machine

Install the required libraries (you can use pip install -r requirements.txt if available)

Download the dataset from Kaggle and place fake_job_postings.csv in the same directory as the notebook

Open and run the Jupyter notebook: Fake job prediction.ipynb


Conclusion
This project demonstrates an effective approach to detecting fraudulent job postings using NLP and machine learning. By combining TF-IDF vectorization, SMOTE, and multiple classification models, the solution achieves solid performance in identifying scams. Future improvements could involve hyperparameter tuning, deep learning methods, or using additional external features to further enhance prediction accuracy.
