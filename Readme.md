# AI Social Media Sentiment Analyzer

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-success)

An **AI-powered web application** that analyzes the sentiment of social media posts, customer reviews, or any textual content using **Natural Language Processing (NLP)**.

The system combines **VADER Sentiment Analysis** with **Transformer-based deep learning models** to provide accurate **Positive, Negative, and Neutral sentiment classification**.

---

# Live Demo

*(Add your Render deployment link here)*

---

# Key Features

* AI-powered sentiment detection
* Hybrid **VADER + Transformer ensemble model**
* Analyze multiple texts at once
* Upload **CSV or TXT files** for batch analysis
* Interactive **Chart.js sentiment visualization**
* Export results as **CSV**
* Animated modern UI
* Real-time sentiment statistics

---

# Application Architecture

```
User (Browser)
      │
      ▼
Frontend (HTML + CSS + JavaScript)
      │
      │  Fetch API Request
      ▼
Flask Backend (Python API)
      │
      ▼
Text Preprocessing
      │
      ▼
Sentiment Analysis Engine
(VADER + Transformer Model)
      │
      ▼
Sentiment Classification
      │
      ▼
JSON Response
      │
      ▼
Frontend Visualization
(Chart.js + Results UI)
```

---

# Tech Stack

## Frontend

* HTML5
* CSS3
* JavaScript
* Chart.js
* Animate.css

## Backend

* Python
* Flask

## AI / NLP

* NLTK VADER Sentiment Analyzer
* HuggingFace Transformers
* PyTorch

## Data Processing

* Pandas
* NumPy
* Matplotlib
* Seaborn

---

# Project Structure

```
sentiment-analyzer/
│
├── app.py
├── sentiment_model.py
│
├── templates/
│   └── index.html
│
├── requirements.txt
└── README.md
```

---

# How the System Works

### 1 User Input

User enters text or uploads a dataset.

Example input:

```
Amazing product! I love it.
Worst service ever.
The experience was okay.
```

---

### 2 Text Processing

The backend cleans the text by removing:

* URLs
* hashtags
* mentions
* unnecessary spaces

---

### 3 Sentiment Detection

The system uses an **ensemble model**:

**Model 1 — VADER**

Rule-based sentiment analysis optimized for social media text.

**Model 2 — Transformer**

Deep learning model from HuggingFace for contextual sentiment analysis.

---

### 4 Ensemble Decision Logic

```
If VADER score is near zero
        → Neutral

Else
        → Positive or Negative

If Transformer confidence > threshold
        → Override prediction
```

---

### 5 Results Generation

The backend returns structured results:

```
{
  "text": "Amazing product",
  "sentiment": "Positive",
  "score": 0.82
}
```

---

### 6 Visualization

The frontend displays:

* Sentiment labels
* Confidence score
* Summary statistics
* Interactive sentiment chart

---

# Installation

## 1 Clone Repository

```
git clone https://github.com/yourusername/sentiment-analyzer.git
cd sentiment-analyzer
```

---

## 2 Create Virtual Environment

```
python -m venv venv
```

Activate it

### Windows

```
venv\Scripts\activate
```

### Mac/Linux

```
source venv/bin/activate
```

---

## 3 Install Dependencies

```
pip install -r requirements.txt
```

---

## 4 Download NLTK Dataset

```
python
```

```
import nltk
nltk.download('vader_lexicon')
```

---

## 5 Run the Application

```
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

# Deployment on Render

### Build Command

```
pip install -r requirements.txt
```

### Start Command

```
gunicorn app:app
```

---

# Example Output

| Text                    | Sentiment | Score |
| ----------------------- | --------- | ----- |
| Amazing product         | Positive  | 0.82  |
| Worst service ever      | Negative  | -0.76 |
| The experience was okay | Neutral   | 0.01  |

---

# Future Improvements

* Twitter API integration
* Real-time sentiment monitoring
* Multi-language support
* Fine-tuned transformer models
* Dashboard analytics
* Large-scale social media scraping

---

# Author

Mohd Shahbaz Khan
B.Tech Computer Science Engineering

---

# License

MIT License

Feel free to use and improve this project.
