# yelp_review_classification
This project builds a sentiment classification model using Yelp restaurant reviews.   The goal is to automatically determine whether a review expresses **positive** or **negative** sentiment.   We focus specifically on **1-star** (negative) and **5-star** (positive) reviews to create a clean binary classification problem.

## Dataset

The dataset used in this project is the standard **Yelp reviews dataset**, which includes:

- The full text of each review  
- A star rating from **1 to 5**  

Although the dataset contains all star values (1–5), this project focuses on a **binary classification** problem by selecting only:

- **1-star reviews** → considered **negative**
- **5-star reviews** → considered **positive**

## Project Objectives

The notebook walks through the complete ML pipeline:

### **1. Data Loading & Exploration**
- Inspect dataset structure  
- View class distribution  
- Visualize review length distribution globally and per star rating  

### **2. Feature Engineering**
- Create a simple numerical feature: `length` of each review  
- Plot histograms and facet plots  

### **3. Text Cleaning**
A custom cleaning function is used to:
- Remove punctuation  
- Convert to lowercase  
- Tokenize  
- Remove English stopwords  

### **4. Text Vectorization (Bag-of-Words)**
- Convert text into word-count vectors using `CountVectorizer`  
- Apply the cleaned tokens as the analyzer  

### **5. Model Training**
Train a **Multinomial Naive Bayes** classifier using:
- 80% training data  
- 20% testing data  

### **6. Evaluation**
- Classification report (precision, recall, f1-score)  
- Confusion matrix visualization

## Important Finding — TF-IDF **reduced** performance

The notebook also includes an optional experiment using **TF-IDF (TfidfTransformer)** on top of the word counts.

 **Result:** TF-IDF significantly **worsened** the model’s performance.

This is expected in some Naive Bayes cases because:

- TF-IDF down-weights frequent words too aggressively  
- Naive Bayes performs best with **raw frequency counts**, not normalized features  
- Yelp reviews contain many repeated opinion words (“good”, “bad”, “love”, “terrible”) which NB handles well with raw counts  

For this project, the **CountVectorizer (Bag-of-Words)** model provides **much higher accuracy**.

---
