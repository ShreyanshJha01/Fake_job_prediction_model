#Fake Job Prediction Model

Overview

This project is focused on detecting fraudulent job postings in a dataset using natural language processing (NLP) and machine learning techniques. It uses the fake_job_postings.csv dataset, which contains various job posting attributes including job descriptions, company profiles, salary ranges, and other features. The goal is to identify which job postings are fraudulent and which are legitimate.

The project utilizes techniques such as text preprocessing, feature extraction (CountVectorizer), and machine learning algorithms (Multinomial Naive Bayes) to train and evaluate a model that can classify job postings as fraudulent or not.

Libraries Used

The following Python libraries are used in this project:

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Matplotlib and Seaborn: For data visualization and plotting.

Scikit-learn: For machine learning model building, evaluation, and feature extraction.
NLTK: For natural language processing, specifically stop word removal.
WordCloud: For generating word clouds.
Dataset
The dataset used in this project is called fake_job_postings.csv and can be found on Kaggle:
Real or Fake - Fake Job Posting Prediction Dataset

The dataset contains the following columns:

job_id: Unique identifier for each job posting.
telecommuting: Whether the job allows telecommuting (1 or 0).
has_company_logo: Whether the job posting has a company logo (1 or 0).
has_questions: Whether the job posting includes questions (1 or 0).
fraudulent: Target column (1 = fraudulent, 0 = not fraudulent).
title: Job title.
location: Job location.
department: Department of the job.
salary_range: The salary range for the job.
company_profile: Description of the company.
description: Job description.
requirements: Requirements for the job.
benefits: Benefits offered by the company.
employment_type: Type of employment (e.g., full-time, part-time).
required_experience: Required experience for the job.
required_education: Required education for the job.
industry: Industry in which the job is offered.
function: Function or department of the job.
Preprocessing Steps
Missing Value Handling:
Numerical columns: Missing values are filled with the median of the respective column.
Categorical columns: Missing values are filled with the string "Unknown".
Text Data Processing:
Several textual columns (e.g., title, location, company_profile) are concatenated into a single column text.
Stop words are removed from the text data to enhance feature extraction.
Feature Encoding:
Categorical columns such as required_experience, required_education, and employment_type are encoded using LabelEncoder.
Text Vectorization:
The text data is vectorized using CountVectorizer to transform the text into a bag-of-words representation.
Model Training and Evaluation
Model: Multinomial Naive Bayes (MultinomialNB) is used for classification, which is well-suited for text classification problems.
Evaluation: The model's performance is evaluated using accuracy, classification report (precision, recall, F1-score), and a confusion matrix.
Visualizations
Word Cloud for All Jobs: A word cloud is generated showing the most frequent words across all job postings.
Word Cloud for Real Jobs: A separate word cloud for legitimate (non-fraudulent) job postings.
Word Cloud for Fraudulent Jobs: A word cloud for fraudulent job postings.
Confusion Matrix: A confusion matrix is displayed to evaluate the performance of the model in terms of true positives, true negatives, false positives, and false negatives.
Files
fake_job_postings.csv: The dataset containing job posting data.
fake_job_postings_detection.ipynb: The Jupyter notebook containing the code for the project.
Instructions to Run the Project
Clone this repository to your local machine.
Install the required dependencies.
Download the fake_job_postings.csv file and place it in the same directory as the notebook.
Run the Jupyter notebook fake_job_postings_detection.ipynb.
Conclusion
This project provides a robust approach for detecting fraudulent job postings using machine learning and NLP techniques. The results from the model can be further improved with more advanced techniques such as deep learning or by fine-tuning hyperparameters. This project serves as a valuable step toward automating the detection of fraudulent job postings, which can protect job seekers and employers from scams.
