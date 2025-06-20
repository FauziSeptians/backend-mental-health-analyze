from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd  # Ini yang benar
import torch
import re
import random
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from model_utils import load_model_for_inference
from huggingface_hub import login



login(token='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')
negative_sentiment_words = set()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model, tokenizer = load_model_for_inference()

def highlight_negative_words(text, important_words):
    highlighted = text
    for word in important_words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted = pattern.sub(f"**{word}**", highlighted)
    return highlighted

def load_training_data(file_path):
    print(f"Loading training data from {file_path}...")
    df = pd.read_csv(file_path)
    data = df
    print(f"Loaded {len(data)} samples for analysis")
    return data["post_text"].tolist(), data["Sentiment_encoded"].tolist()

def analyze_mental_health_comment(text, model, tokenizer):
    original_text = text

    processed_text = text

    encoded = tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            prediction = torch.sigmoid(logits).item() if logits.numel() == 1 else torch.softmax(logits, dim=1)[0, 1].item()
        else:
            prediction = torch.sigmoid(outputs[0]).item() if outputs[0].numel() == 1 else torch.softmax(outputs[0], dim=1)[0, 1].item()

        print("res : pred", prediction)

    sentiment = "POSITIVE" if prediction >= 0.5 else "NEGATIVE"
    confidence = prediction if sentiment == "POSITIVE" else 1 - prediction

    words = processed_text.split()
    word_importance = {}

    for i, word in enumerate(words):
        if len(word.strip()) == 0:
            continue

        modified_words = words.copy()
        modified_words[i] = "[UNK]"
        modified_text = " ".join(modified_words)

        mod_encoded = tokenizer(modified_text, return_tensors="pt", truncation=True, max_length=128)
        mod_input_ids = mod_encoded["input_ids"].to(device)
        mod_attention_mask = mod_encoded["attention_mask"].to(device)

        with torch.no_grad():
            mod_outputs = model(mod_input_ids, mod_attention_mask)
            if hasattr(mod_outputs, "logits"):
                mod_logits = mod_outputs.logits
                mod_prediction = torch.sigmoid(mod_logits).item() if mod_logits.numel() == 1 else torch.softmax(mod_logits, dim=1)[0, 1].item()
            else:
                mod_prediction = torch.sigmoid(mod_outputs[0]).item() if mod_outputs[0].numel() == 1 else torch.softmax(mod_outputs[0], dim=1)[0, 1].item()

        importance = abs(prediction - mod_prediction)
        word_importance[word] = importance

    word_importance_sorted = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)

    num_important = max(3, min(int(len(words) * 0.3), 10))
    most_important_words = [word for word, _ in word_importance_sorted[:num_important]]

    print(f"Analyzing text: {original_text}")
    print(f"Processed text: {processed_text}")
    print(f"Detected sentiment: {sentiment} (Confidence: {confidence:.2f})")
    print(f"Most important words: {most_important_words}")

    highlighted_text = highlight_negative_words(original_text, most_important_words)

    if word_importance_sorted:
        max_score = max(score for _, score in word_importance_sorted)
        if max_score > 0:
            normalized_word_importance = [(word, round((score/max_score) * 100, 2))
                                         for word, score in word_importance_sorted]
        else:
            normalized_word_importance = [(word, 0) for word, _ in word_importance_sorted]
    else:
        normalized_word_importance = []

    return {
        "original_text": original_text,
        "processed_text": processed_text,
        "highlighted_text": highlighted_text,
        "sentiment": sentiment,
        "confidence": confidence,
        "prediction": prediction,
        "mental_health_context": "Negative mental health sentiment detected." if sentiment == "NEGATIVE" else "Positive mental health sentiment.",
        "negative_words_detected": most_important_words,
        "word_importances": normalized_word_importance
    }

def generate_wordcloud_from_importance(word_importances, max_words=100, normalize=True):
    """
    Generate wordcloud based on actual word importance scores from NLP analysis
    
    Args:
        word_importances: List of [word, importance_score] pairs
        max_words: Maximum number of words to display
        normalize: Whether to normalize scores (recommended for better visualization)
    """
    print(f"Generating scientifically valid wordcloud with {len(word_importances)} words...")
    
    # Convert to frequency dictionary
    if normalize:
        # Normalize scores to 1-100 range for better wordcloud scaling
        scores = [score for word, score in word_importances]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            word_freq = {word: 50 for word, score in word_importances}
        else:
            word_freq = {
                word: 1 + 99 * (score - min_score) / (max_score - min_score)
                for word, score in word_importances
            }
    else:
        # Use raw importance scores
        word_freq = {word: max(score, 0.1) for word, score in word_importances}  # Ensure positive values
    
    print("Word frequencies for wordcloud:")
    for word, freq in list(word_freq.items())[:5]:  # Show top 5
        original_score = next(score for w, score in word_importances if w == word)
        print(f"  '{word}': {freq:.2f} (original importance: {original_score})")
    
    # Generate wordcloud using importance-based frequencies
    wordcloud = WordCloud(
        width=1200, 
        height=600, 
        background_color='white',
        max_words=max_words, 
        contour_width=3, 
        contour_color='steelblue',
        colormap='viridis',  # Better color scheme for importance visualization
        relative_scaling=0.5,  # Better size distribution
        min_font_size=10
    ).generate_from_frequencies(word_freq)

    # Create the plot
    img = io.BytesIO()
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Importance Visualization", fontsize=16, pad=20)
    plt.tight_layout(pad=0)
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    img.seek(0)
    
    print("NLP-valid wordcloud generated successfully!")
    return img

def create_app(model, tokenizer):
    print("Creating Flask application...")
    app = Flask(__name__)

    CORS(app, resources={r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "ngrok-skip-browser-warning"]
    }})

    @app.route('/analyze-sentiment', methods=['POST'])
    def analyze_sentiment():
        data = request.json
        if 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        print(f"Analyzing sentiment for text: '{data['text'][:50]}...'")
        result = analyze_mental_health_comment(data['text'], model, tokenizer)

        if result["sentiment"] == "NEGATIVE":
            new_negative_words = result['negative_words_detected']
            negative_sentiment_words.update(new_negative_words)

        print(f"Sentiment analysis complete. Result: {result['sentiment']}")
        return jsonify({"data": result})

    @app.route('/analyze-sentiment-bulk', methods=['POST'])
    def analyze_sentiment_bulk():
        data = request.json
        if 'texts' not in data or not isinstance(data['texts'], list):
            return jsonify({"error": "No texts array provided or invalid format"}), 400

        results = []
        for text in data['texts']:
            print(f"Analyzing sentiment for text: '{text[:50]}...'")
            result = analyze_mental_health_comment(text, model, tokenizer)

            if result["sentiment"] == "NEGATIVE":
                new_negative_words = result['negative_words_detected']
                negative_sentiment_words.update(new_negative_words)

            results.append(result)

        print(f"Bulk sentiment analysis complete. Processed {len(results)} texts.")
        return jsonify({"data": results})

    @app.route('/generate-wordcloud', methods=['POST'])
    def create_wordcloud():
        data = request.json
        if 'array' not in data:
            print(data['array'])
            return jsonify({"error": "No data provided"}), 400
        wordCloud = data['array']
        print(wordCloud)
        img = generate_wordcloud_from_importance(wordCloud, max_words=data.get('max_words', 100))
        print("Wordcloud generated and ready to send")
        return send_file(img, mimetype='image/png')

    @app.route('/negative-sentiment-list', methods=['GET'])
    def get_negative_words():
        print(f"Returning list of {len(negative_sentiment_words)} negative sentiment words")
        return jsonify({"negative_sentiment_words": list(negative_sentiment_words), "count": len(negative_sentiment_words)})

    @app.route('/', methods=['GET'])
    def home():
        print("Home endpoint accessed")
        return jsonify({
            "message": "Mental Health Sentiment Analysis API",
            "model": "mental/mental-bert-base-uncased",
            "negative_words_count": len(negative_sentiment_words)
        })

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With,ngrok-skip-browser-warning')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response

    print("Flask application created successfully with all endpoints configured!")
    return app

app = create_app(model, tokenizer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000)