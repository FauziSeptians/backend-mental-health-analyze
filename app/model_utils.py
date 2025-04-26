import os
import json
import torch
import requests
import zipfile
from transformers import AutoTokenizer, AutoModel
from model import MentalBERTClassifier

# URLs for model, tokenizer, and config
MODEL_URL = "https://www.dropbox.com/scl/fi/a697fkln3gubbee0giwll/model.pt?rlkey=r6pdz755f8uh8ut51t8538dzf&st=sk51e6tq&dl=1"
MODEL_CONFIG_URL = "https://www.dropbox.com/scl/fi/6txcchbtse8r207x3bceq/model_config.json?rlkey=d5qqkmjt3hyiac3l5uxqzkryj&st=gsdtcl8f&dl=1"
TOKENIZER_URL = "https://www.dropbox.com/scl/fi/gyuxo6div4fdhf6wr9jux/tokenizer.zip?rlkey=hvd3p8tvw6mn5q4k6krzsprbv&st=90j8thrk&dl=1"

def download_file(url, dest):
    """Download file if not exist."""
    if os.path.exists(dest):
        print(f"📂 {dest} already exists, skipping download.")
        return
    print(f"⬇️ Downloading {dest}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def load_model_for_inference(path='mental_health_model', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure path exists
    os.makedirs(path, exist_ok=True)

    # Define full paths
    model_path = os.path.join(path, "model.pt")
    config_path = os.path.join(path, "model_config.json")
    tokenizer_zip_path = os.path.join(path, "tokenizer.zip")
    tokenizer_dir = os.path.join(path, "tokenizer")

    # Download necessary files
    download_file(MODEL_URL, model_path)
    download_file(MODEL_CONFIG_URL, config_path)
    download_file(TOKENIZER_URL, tokenizer_zip_path)

    # Extract tokenizer if not already extracted
    if not os.path.exists(tokenizer_dir):
        print("📦 Extracting tokenizer...")
        with zipfile.ZipFile(tokenizer_zip_path, 'r') as zip_ref:
            zip_ref.extractall(tokenizer_dir)

    # Load model config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Model config file not found at {config_path}")

    with open(config_path, 'r') as f:
        model_config = json.load(f)

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(path + '/tokenizer/tokenizer')
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load tokenizer: {str(e)}")

    # Load base BERT model
    try:
        bert_model = AutoModel.from_pretrained(model_config['bert_model_name'])
    except KeyError:
        raise KeyError("'bert_model_name' not found in model_config.json")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load base BERT model: {str(e)}")

    # Instantiate classifier
    try:
        model = MentalBERTClassifier(bert_model, dropout_rate=model_config.get('dropout_rate', 0.3))
    except NameError:
        raise NameError("MentalBERTClassifier class is not defined properly.")
    
    # Load weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model weights not found at {model_path}")

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model weights: {str(e)}")

    model = model.to(device)
    model.eval()

    print(f"✅ Model, config, and tokenizer loaded successfully from {path}")
    print(f"🖥️  Model loaded to: {device}")

    return model, tokenizer
