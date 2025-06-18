📧 Spam Detection using Machine Learning
This project aims to detect spam messages using machine learning models. The dataset contains SMS messages labeled as spam or ham (not spam). Three models are implemented and compared: Naive Bayes, Decision Tree (ID3), and Support Vector Machine (SVM).

📂 Dataset
The dataset is a CSV file (spam_data.csv) containing two columns:

Category: Label indicating if a message is "spam" or "ham".
Message: The actual SMS text.
🧹 Preprocessing
Lowercased all messages
Removed digits and punctuation
Removed English stopwords using NLTK
📊 Feature Extraction
Used TF-IDF Vectorization to convert text into numerical features suitable for machine learning models.

🧠 Models Used
1. Naive Bayes (MultinomialNB)
Baseline and GridSearchCV-tuned version
Achieved accuracy: 96.65%
Tuned version achieved: 97.97%
2. Decision Tree (ID3 algorithm)
Used entropy as the criterion
Base model accuracy: 95.75%
Tuned version achieved: 95.63%
3. Support Vector Machine (SVM)
Used a linear kernel
Best overall performance
Accuracy: 98.62%
Precision: 97.63%
Recall: 91.96%
F1 Score: 94.71%
📈 Evaluation Metrics
Metrics computed for each model:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Classification Report
📊 Visualizations
Confusion Matrices (using seaborn heatmaps)
Model Accuracy Comparison (bar chart)
Precision, Recall, F1-Score Comparisons (bar charts)
Word Cloud of most frequent words
📦 Requirements
pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud
Also, download NLTK stopwords:

import nltk
nltk.download('stopwords')
🚀 Running the Project
Load the dataset and preprocess text.
Train and evaluate the models.
Compare performance visually and statistically.
Identify the best performing model — SVM in this case.
📌 Conclusion
The SVM model outperformed Naive Bayes and Decision Tree in terms of accuracy and other evaluation metrics. It's recommended as the most robust model for spam classification on this dataset.

📁 File Structure
├── spam_data.csv
├── spam_classifier.ipynb / .py
├── README.md
🔒 License
This project is for educational purposes. Feel free to use and modify it for learning or development.

