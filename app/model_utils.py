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

# Local file paths
MODEL_PATH = "model.pt"
MODEL_CONFIG_PATH = "model_config.json"
TOKENIZER_ZIP = "tokenizer.zip"
TOKENIZER_DIR = "tokenizer"

def download_file(url, dest):
    if os.path.exists(dest):
        return
    with requests.get(url, stream=True) as r:
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def load_model_for_inference(path='mental_health_model', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Download model, config, and tokenizer from Dropbox
    download_file(MODEL_URL, MODEL_PATH)
    download_file(MODEL_CONFIG_URL, MODEL_CONFIG_PATH)
    download_file(TOKENIZER_URL, TOKENIZER_ZIP)

    # Extract tokenizer
    if not os.path.exists(TOKENIZER_DIR):
        with zipfile.ZipFile(TOKENIZER_ZIP, 'r') as zip_ref:
            zip_ref.extractall(TOKENIZER_DIR)

    # Load model config
    if not os.path.exists(MODEL_CONFIG_PATH):
        raise FileNotFoundError(f"Model config file not found at {MODEL_CONFIG_PATH}")

    with open(MODEL_CONFIG_PATH, 'r') as f:
        model_config = json.load(f)

    # Load tokenizer
    tokenizer_path = TOKENIZER_DIR
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {str(e)}")

    # Load base BERT model
    try:
        bert_model = AutoModel.from_pretrained(model_config['bert_model_name'])
    except KeyError:
        raise KeyError("'bert_model_name' not found in model_config.json")
    except Exception as e:
        raise RuntimeError(f"Failed to load base BERT model: {str(e)}")

    # Assuming you have a custom classifier (MentalBERTClassifier)
    try:
        model = MentalBERTClassifier(bert_model, dropout_rate=model_config.get('dropout_rate', 0.3))
    except NameError:
        raise NameError("MentalBERTClassifier class is not defined. Make sure to import or define it before calling this function.")

    # Load model weights
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {str(e)}")

    model = model.to(device)
    model.eval()

    print(f"✅ Model, config, and tokenizer loaded successfully from {path}")
    print(f"   Model loaded to: {device}")

    return model, tokenizer
