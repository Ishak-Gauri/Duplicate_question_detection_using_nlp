# Duplicate Question Detection in Q&A Communities

This repository contains the complete implementation and analysis of the project **"Identifying Duplicate Questions in Q&A Communities Using Machine Learning: An Empirical Study with NLP and Deep Learning Techniques"**.

## 🧠 Objective

The goal of this project is to automatically detect whether two questions posted on a Q&A platform (like Quora) are semantically equivalent — i.e., duplicates — using both traditional Machine Learning techniques and advanced Deep Learning models.

## 📂 Dataset

- **Source**: Quora Question Pairs (available on [Kaggle](https://www.kaggle.com/c/quora-question-pairs))
- **Size**: 400,000+ question pairs
- Each row includes: `question1`, `question2`, and a binary label `is_duplicate`.

## ⚙️ Features & Preprocessing

- Text Cleaning:
  - Lowercasing
  - Decontraction (e.g., `can't` → `cannot`)
  - Removal of HTML, special characters, and extra spaces using BeautifulSoup and regex
- Tokenization and stopword removal
- Feature Engineering:
  - Length and word count (`q1_len`, `q2_len`, `q1_num_words`, `q2_num_words`)
  - Common and total word count
  - Word share ratio
  - Token-based features (e.g., common non-stopwords, stopwords, etc.)

## 📊 Models Used

### Traditional Machine Learning
- Logistic Regression
- Random Forest
- Gradient Boosting (XGBoost)

### Deep Learning
- **Siamese LSTM** network using Keras:
  - Embedding layer with GloVe vectors
  - Shared BiLSTM layers
  - Manhattan distance + dense layer for prediction

## 🧪 Evaluation Metrics

- Accuracy
- F1-Score
- Precision & Recall
- ROC-AUC Score

## 📌 Results

| Model             | Accuracy | F1-Score |
|------------------|----------|----------|
| Logistic Regression | ~76%     | ~0.70    |
| Random Forest       | ~78%     | ~0.72    |
| Siamese LSTM        | ~82%     | ~0.77    |

## 🧰 Technologies Used

- Python
- Pandas, NumPy, scikit-learn
- NLTK, BeautifulSoup
- TensorFlow / Keras
- Streamlit (for interactive app)
- Matplotlib / Seaborn for visualizations

## 🚀 Working Demo

You can run the interactive **Streamlit** web app locally.

### 🔧 Command to Run

```bash
streamlit run app.py


After running, open your browser and visit:
http://localhost:8501
