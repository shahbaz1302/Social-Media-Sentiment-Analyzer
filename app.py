from flask import Flask, request, jsonify, render_template, send_file
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import io
import base64
import pandas as pd
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import csv
from collections import Counter
import re
import os

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

# Load transformer model
try:
    transformer_pipeline = pipeline("sentiment-analysis")
except:
    transformer_pipeline = None
    print("Warning: Transformers not available, using VADER only")

def clean_text(text):
    """Clean text for better analysis"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    return text.strip()

def ensemble_sentiment(text):
    """FIXED: Proper neutral detection with debug output"""
    print(f"Analyzing: '{text[:50]}...'")  # Debug
    
    # VADER analysis (primary)
    vader_scores = sia.polarity_scores(text)
    vader_compound = vader_scores['compound']
    print(f"VADER compound: {vader_compound:.3f}")
    
    # Transformer analysis (secondary, only if available)
    trans_label = None
    trans_score = 0
    if transformer_pipeline:
        try:
            trans_result = transformer_pipeline(text)[0]
            trans_label = trans_result['label']
            trans_score = trans_result['score'] if trans_result['label'] == 'POSITIVE' else -trans_result['score']
            print(f"Transformer: {trans_label} ({trans_score:.3f})")
        except Exception as e:
            print(f"Transformer error: {e}")
    
    # FIXED LOGIC: Prioritize VADER for neutral detection
    if abs(vader_compound) < 0.10:  # Wider neutral range: -0.10 to +0.10
        sentiment = "Neutral"
        score = round(vader_compound, 3)
        print(f"✅ NEUTRAL (VADER: {vader_compound:.3f})")
    else:
        # Use ensemble only for strong positive/negative
        if vader_compound >= 0.10:
            sentiment = "Positive"
        else:
            sentiment = "Negative"
        
        # Transformer boost only for extremes
        if transformer_pipeline and abs(trans_score) > 0.7:
            sentiment = "Positive" if trans_score > 0 else "Negative"
        
        score = round((vader_compound * 0.7 + trans_score * 0.3) if transformer_pipeline else vader_compound, 3)
        print(f"Final: {sentiment} ({score:.3f})")
    
    return sentiment, score

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    texts = data.get("texts", [])
    results = []
    
    for text in texts:
        sentiment, score = ensemble_sentiment(clean_text(text))
        results.append({
            "text": text,
            "sentiment": sentiment,
            "score": score
        })
    
    return jsonify({
        "results": results,
        "summary": get_sentiment_summary(results)
    })

@app.route("/analyze_file", methods=["POST"])
def analyze_file():
    print("📁 File upload request received")  # Debug log
    
    # Check if file exists in request
    if 'file' not in request.files:
        print("❌ No file in request.files")
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    print(f"📄 File received: {file.filename}")  # Debug log
    
    # Check if file has content
    if file.filename == '':
        print("❌ Empty filename")
        return jsonify({"error": "No file selected"}), 400
    
    # Check file size (max 10MB)
    file.seek(0, 2)  # Move to end
    file_size = file.tell()
    file.seek(0)  # Reset to start
    print(f"📏 File size: {file_size} bytes")
    
    if file_size > 10 * 1024 * 1024:  # 10MB limit
        return jsonify({"error": "File too large (max 10MB)"}), 400
    
    texts = []
    try:
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(file, nrows=100)  # First 100 rows
            # Try common text columns
            text_col = None
            for col in df.columns:
                if 'text' in col.lower() or 'comment' in col.lower() or 'post' in col.lower():
                    text_col = col
                    break
            if text_col is None:
                text_col = df.columns[0]  # First column as fallback
            texts = df[text_col].astype(str).tolist()
            
        elif file.filename.lower().endswith('.txt'):
            content = file.read().decode('utf-8', errors='ignore')
            texts = [line.strip() for line in content.split('\n') if line.strip()]
            texts = texts[:100]
            
        print(f"✅ Extracted {len(texts)} texts from file")
        
    except Exception as e:
        print(f"❌ File processing error: {str(e)}")
        return jsonify({"error": f"File processing failed: {str(e)}"}), 400
    
    if not texts:
        return jsonify({"error": "No text data found in file"}), 400
    
    # Process texts
    results = []
    for text in texts:
        sentiment, score = ensemble_sentiment(clean_text(text))
        results.append({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "sentiment": sentiment,
            "score": score
        })
    
    return jsonify({
        "results": results,
        "summary": get_sentiment_summary(results),
        "total_processed": len(results),
        "filename": file.filename
    })


@app.route("/export_csv", methods=["POST"])
def export_csv():
    data = request.json
    results = data.get("results", [])
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Text", "Sentiment", "Score"])
    
    for item in results:
        writer.writerow([item["text"][:100], item["sentiment"], item["score"]])
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'sentiment_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

def get_sentiment_summary(results):
    sentiments = [r["sentiment"] for r in results]
    counts = Counter(sentiments)
    total = len(results)
    
    return {
        "positive": round(counts.get("Positive", 0) / total * 100, 1),
        "negative": round(counts.get("Negative", 0) / total * 100, 1),
        "neutral": round(counts.get("Neutral", 0) / total * 100, 1),
        "dominant": max(counts, key=counts.get) if counts else "Neutral",
        "total": total
    }

@app.route("/chart", methods=["POST"])
def generate_chart():
    data = request.json
    results = data.get("results", [])
    
    sentiments = [r["sentiment"] for r in results]
    counts = Counter(sentiments)
    
    plt.figure(figsize=(8, 6))
    colors = {'Positive': '#4CAF50', 'Negative': '#F44336', 'Neutral': '#FFC107'}
    plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', colors=[colors.get(k, '#888') for k in counts.keys()])
    plt.title("Sentiment Distribution")
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({"chart": f"data:image/png;base64,{plot_url}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

