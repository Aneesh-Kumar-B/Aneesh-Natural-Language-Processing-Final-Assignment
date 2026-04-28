# Financial Decision Making

## Overview

This project predicts financial decisions — BUY, HOLD, or SELL — using financial news data.  
The model learns the relationship between company-specific news and future price movement.

--------------------------------------------------

## How to Run the Code:

1. Install dependencies:

    pip install datasets transformers scikit-learn torch pandas numpy matplotlib

2. Open the notebook (.ipynb) in Jupyter Notebook or Google Colab.

3. Run all cells in order:
   - Load dataset
   - Preprocess data
   - Create labels
   - Train baseline model
   - Train transformer model
   - Evaluate results

4. Run prediction:

    predict("Company reports strong earnings growth", "AAPL")

--------------------------------------------------

## Report:

### Dataset

Dataset used:
https://huggingface.co/datasets/TheFinAI/daily_news

The dataset contains:
- Financial news (list of headlines)
- Asset names (AAPL, TSLA, BTC, etc.)
- Future price differences

--------------------------------------------------

### Data Processing

- Combined all assets into a single dataset
- Converted list of news into a single text string
- Removed missing values from target column
- Cleaned text data

--------------------------------------------------

### Label Creation

The continuous target (future_price_diff) was converted into 3 classes using quantile-based binning:

0 → SELL  
1 → HOLD  
2 → BUY  

Final class distribution:

SELL  = 1076  
HOLD  = 1076  
BUY   = 1076  

--------------------------------------------------

### Feature Engineering

Example Final input format:

- Financial news about AAPL. Company reports strong earnings growth.

- Only company name and news text are used as input.

--------------------------------------------------

### Models Used

Baseline Model:
- TF-IDF vectorization
- Logistic Regression

Transformer Model:
- BERT (bert-base-uncased)
- Fine-tuned for classification

--------------------------------------------------

### Experimental Setup

- Train/Test split: 80% / 20% (time-based)
- Max sequence length: 256
- Batch size: 4
- Epochs: 2

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score

--------------------------------------------------

### Results

Baseline Model Accuracy: 38%

SELL  → Precision: 0.36 | Recall: 0.34 | F1: 0.35  
HOLD  → Precision: 0.43 | Recall: 0.50 | F1: 0.46  
BUY   → Precision: 0.36 | Recall: 0.33 | F1: 0.34  

--------------------------------------------------

Transformer Model (BERT) Accuracy: 45%

SELL  → Precision: 0.49 | Recall: 0.18 | F1: 0.26  
HOLD  → Precision: 0.44 | Recall: 0.72 | F1: 0.54  
BUY   → Precision: 0.45 | Recall: 0.46 | F1: 0.46  

<img width="570" height="462" alt="image" src="https://github.com/user-attachments/assets/96165e9b-2081-4627-9f5d-bc00448a7ab2" />

--------------------------------------------------

### Observations

- Transformer model performs better than baseline
- HOLD class has highest recall
- SELL class is hardest to predict
- Model tends to predict HOLD when uncertain

--------------------------------------------------

### Sample Prediction

Input:
Company reports strong earnings growth  
Company: AAPL  

Output:
HOLD
