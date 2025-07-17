# 📩 SMS Spam Detection using Naive Bayes and LSTM

This project implements and compares different machine learning models to classify SMS messages as **spam** or **ham (non-spam)**. The goal is to evaluate whether a traditional model like **Naive Bayes** or a deep learning model like **LSTM** performs better in accurately detecting spam messages.

---

## 📚 Dataset

- **Source**: [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- The dataset contains 5,572 SMS messages labeled as either `ham` or `spam`.

---

## 🧠 Models Implemented

| Model           | Framework  | Description |
|----------------|------------|-------------|
| Naive Bayes     | Scikit-learn | TF-IDF-based classifier with Laplace smoothing |
| Keras LSTM      | TensorFlow/Keras | LSTM neural network trained on tokenized SMS text |
| PyTorch LSTM (Experimental) | PyTorch | Implemented for comparison, but excluded due to underperformance |

---

## 🧪 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

---

## 📊 Final Results (Test Set)

| Model        | Accuracy | Precision (Spam) | Recall (Spam) | F1 Score (Spam) |
|--------------|----------|------------------|----------------|-----------------|
| Naive Bayes  | 0.97     | 1.00             | 0.78           | 0.88            |
| Keras LSTM   | 0.99     | 0.97             | 0.94           | 0.95            |

> 🔍 **Conclusion**: Keras LSTM outperforms Naive Bayes in recall and F1-score, making it more robust in detecting spam.

---

## 🚫 PyTorch LSTM Note

A PyTorch-based LSTM was implemented but consistently underperformed (87% accuracy, stagnant learning). It was therefore excluded from the final evaluation.

---

## 🔧 Project Structure
├── spam.csv # Dataset (from Kaggle)
├── ICT303_Assignment2.ipynb # Main Jupyter notebook
├── README.md # Project readme
├── logs/ # TensorBoard logs (Keras)
└── lstm_model.png # Visualized LSTM model architecture


---

## 🛠️ Libraries Used

- **Python 3.10+**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Scikit-learn**
- **NLTK** (for text cleaning and stopword removal)
- **TensorFlow / Keras**
- **PyTorch** (for experimental model)
- **torchview** (for model visualization)

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/sms-spam-detector.git
cd sms-spam-detector

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook ICT303_Assignment2.ipynb
