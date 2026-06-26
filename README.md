# Yelp Review Sentiment Classification

A Natural Language Processing project that classifies Yelp restaurant reviews as positive or negative using Bag-of-Words features and a Multinomial Naive Bayes classifier.

## Project Goal

The goal is to build a clean binary sentiment classifier using Yelp reviews. The project focuses on 1-star reviews as negative examples and 5-star reviews as positive examples, creating a clear supervised learning problem.

## Dataset

The project uses Yelp review data containing review text and star ratings. For this binary classification task:

- 1-star reviews are labeled as negative.
- 5-star reviews are labeled as positive.
- 2, 3, and 4-star reviews are excluded to reduce label ambiguity.

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Matplotlib, Seaborn
- Jupyter Notebook

## Workflow

1. Load and inspect the Yelp review dataset.
2. Explore review length and rating distributions.
3. Filter the dataset to 1-star and 5-star reviews.
4. Clean text by removing punctuation, lowercasing, tokenizing, and removing stopwords.
5. Vectorize text with `CountVectorizer`.
6. Train a Multinomial Naive Bayes classifier.
7. Evaluate performance using a classification report and confusion matrix.

## Key Finding

A TF-IDF experiment reduced model performance in this notebook. For this dataset and Naive Bayes setup, raw word-count features performed better because repeated sentiment words such as "great", "love", "bad", and "terrible" carry strong predictive signal.

## How to Run

```bash
git clone https://github.com/marcelngoyi90/yelp_review_classification.git
cd yelp_review_classification
pip install pandas numpy scikit-learn nltk matplotlib seaborn jupyter
jupyter notebook
```

Open the notebook and run the cells in order.

## Future Improvements

- Add logistic regression and linear SVM baselines.
- Compare CountVectorizer, TF-IDF, and transformer embeddings.
- Add cross-validation and hyperparameter tuning.
- Convert the notebook into a reusable training script.
