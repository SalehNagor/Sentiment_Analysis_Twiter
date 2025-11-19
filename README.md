# IMDB Sentiment Analysis Pipeline 🎬

This project implements a modular Machine Learning pipeline for sentiment analysis on the IMDB movie reviews dataset. It utilizes **DistilBERT** for sequence classification, fine-tuned to achieve high accuracy in detecting positive and negative sentiments.

## 🚀 Project Features
- **Modular Architecture:** Code is organized into distinct modules for preprocessing, training, and evaluation.
- **End-to-End Pipeline:** A single entry point (`main.py`) manages the entire workflow from data ingestion to evaluation.
- **State-of-the-Art Model:** Uses Hugging Face's `DistilBertForSequenceClassification`.
- **Reproducibility:** Includes environment requirements and seed setting for consistent results.

## 📂 Project Structure

```text
project_root/
│
├── data/                   # Contains the dataset (IMDB_dataset.csv)
├── models/                 # Stores the fine-tuned model and checkpoints
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── preprocessing.py    # Data cleaning, splitting, and tokenization
│   ├── train.py            # Model initialization and training loop
│   └── evaluate.py         # Evaluation metrics and classification report
│
├── main.py                 # Main script to execute the pipeline
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation

🛠️ Installation
Clone the repository:

git clone [https://github.com/SalehNagor/sentiment_analysis.git](https://github.com/SalehNagor/sentiment_analysis.git)
cd sentiment_analysis

Create a virtual environment (Optional but recommended):

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt
(Note: Ensure you have accelerate installed for the Trainer API)

⚙️ Usage
To run the complete pipeline (Data Ingestion → Training → Evaluation), simply execute the main.py script:

python main.py

📊 Results
The model is evaluated on a held-out test set (15% of the data).

Metric: Accuracy

Expected Performance: ~88% Accuracy on the test set.

📝 Requirements
Python 3.8+

Transformers

PyTorch

Scikit-learn

Pandas