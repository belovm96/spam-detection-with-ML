# Spam Detection with Machine Learning
In this project, I explore email spam and non-spam data and perform ML modeling for spam - not spam text classification. I was able to achieve 96% spam detection accuracy with SVM algorithm.

## Project Recap
* Created a data preprocessing script - utilized stemming and lemmatization methods for Enron1 dataset normalization.
* Performed EDA on the preprocessed data and uncovered some insights that led to further dataset cleaning.
* Utilized TF-IDF & Bag-of-words approaches to:
  1. encode the email data such that it can be used for ML modeling
  2. come up with top 5 most common topics in spam and non-spam emails via Latent Dirichlet Allocation modeling and Non-negative Matrix Factorization algorithm
* Built SVM, Logistic Regression, Random Forest, and Naive Bayes classifiers for spam detection and used grid search to optimize these classifiers' performance.
* Analyzed the performance of each classifier using confusion matrix, precision and recall metrics, and provided some discussion on the final classification results for each machine learning model.

## Visualization Highlights
![Spam Emails WordCloud](https://github.com/belovm96/spam-detection-with-ML/blob/master/images/spamcloud.png)

## Libraries & Tools
**Programming Language:** Python 3.7

**Libraries:** Pandas, NLTK, Scikit-learn, pyLDAvis, Matplotlib, Seaborn

### Credit to:
Kamil Mysiak and his Towards Data Science [article](https://towardsdatascience.com/nlp-part-3-exploratory-data-analysis-of-text-data-1caa8ab3f79d) that I used as a baseline for my EDA.
