import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import re
import requests
import zipfile
import io

MODEL_RELEASE_URL = "https://github.com/YourUsername/nlp-assignment-app/releases/download/v1.0/models.zip"
MODEL_DIR = "models" 

@st.cache_resource
def setup_models(url):
    if not os.path.exists(MODEL_DIR):
        st.info(f"Downloading models from release... (this happens once)")
        
        try:
            r = requests.get(url)
            r.raise_for_status() 
            
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(".") 
            st.success("Models downloaded and unzipped successfully!")
            
        except Exception as e:
            st.error(f"Error downloading models: {e}")
            return None
    
    return MODEL_DIR

MODEL_DIR = setup_models(MODEL_RELEASE_URL)

class WordPredictorMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, block_size, dropout_rate, padding_idx, activation='relu'):
        super(WordPredictorMLP, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.fc1 = nn.Linear(block_size * embedding_dim, hidden_dim)
        
        if activation == 'tanh':
            self.activation1 = nn.Tanh()
        else:
            self.activation1 = nn.ReLU()
            
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if activation == 'tanh':
            self.activation2 = nn.Tanh()
        else:
            self.activation2 = nn.ReLU()
            
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.activation1(self.fc1(x))
        x = self.dropout1(x)
        x = self.activation2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def generate_text(model, seed_text, num_words_to_gen, block_size, word_to_idx, idx_to_word, temperature=1.0, device='cpu'):
    
    model.eval()
    
    padding_idx = 0 
    unknown_idx = 1 
    
    words = seed_text.lower().split()
    generated_text = words[:]
    
    context = words[-block_size:]
    
    if len(context) < block_size:
        pad_indices = [padding_idx] * (block_size - len(context))
        context_idx = pad_indices + [word_to_idx.get(w, unknown_idx) for w in context]
    else:
        context_idx = [word_to_idx.get(w, unknown_idx) for w in context]

    with torch.no_grad():
        for _ in range(num_words_to_gen):
            input_tensor = torch.tensor([context_idx], dtype=torch.long).to(device)
            logits = model(input_tensor)
            
            logits_scaled = logits / temperature
            probs = F.softmax(logits_scaled, dim=1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            
            next_word = idx_to_word.get(next_idx, '<UNK>') 
            
            generated_text.append(next_word)
            context_idx = context_idx[1:] + [next_idx]

    return ' '.join(generated_text)

@st.cache_data
def load_vocab(dataset_prefix):
    vocab_path = os.path.join(MODEL_DIR, f"{dataset_prefix}_vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        vocab_data['idx_to_word'] = {int(k): v for k, v in vocab_data['idx_to_word'].items()}
        return vocab_data['word_to_idx'], vocab_data['idx_to_word']
    return None, None

@st.cache_resource
def load_model(model_filename, model_config):
    device = torch.device('cpu')
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    if os.path.exists(model_path):
        model = WordPredictorMLP(**model_config).to(device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading {model_filename}: {e}")
            return None
    else:
        st.error(f"File not found: {model_filename}")
        return None

st.set_page_config(layout="wide")
st.title("Next-Word Predictor MLP (Section 1.4)")

st.sidebar.header("Generation Controls")

dataset_map = {
    "Category I: Natural Language": "nl",
    "Category II: Structured Code": "code"
}
dataset_choice = st.sidebar.selectbox("Select Dataset:", list(dataset_map.keys()))
dataset_prefix = dataset_map[dataset_choice]

st.sidebar.subheader("1. Model Architecture")

if dataset_prefix == "nl":
    emb_dim = st.sidebar.selectbox("Embedding Dim:", (32, 64))
    block_size = st.sidebar.selectbox("Context Length:", (5, 10))
    hid_dim = 1024 
    activation = "relu" 
else:
    st.sidebar.markdown("*(Fixed Architecture: Emb=64, Ctx=5)*")
    emb_dim = 64
    block_size = 5
    hid_dim = 1024
    activation = "relu"

st.sidebar.subheader("2. Model Checkpoint")
model_variant = st.sidebar.selectbox(
    "Choose Model State:",
    ("good_fit", "overfit", "underfit"),
    format_func=lambda x: x.replace('_', ' ').title()
)

st.sidebar.subheader("3. Generation Parameters")
temperature = st.sidebar.slider(
    "Temperature (Randomness):",
    min_value=0.1, max_value=2.0, value=0.8, step=0.1
)
num_words = st.sidebar.slider(
    "Words to Generate (k):",
    min_value=10, max_value=200, value=50, step=10
)

if MODEL_DIR: 
    word_to_idx, idx_to_word = load_vocab(dataset_prefix)

    if not word_to_idx:
        st.error(f"Could not load '{dataset_prefix}_vocab.json' from the '{MODEL_DIR}' folder. Check file names.")
    else:
        vocab_size = len(word_to_idx)
        
        if dataset_prefix == "nl":
            model_filename = f"model_emb{emb_dim}_hid{hid_dim}_act-{activation}_ctx{block_size}_{model_variant}.pth"
        else:
            model_filename = f"code_{model_variant}.pth"

        model_config = {
            'vocab_size': vocab_size,
            'embedding_dim': emb_dim,
            'hidden_dim': hid_dim,
            'block_size': block_size,
            'dropout_rate': 0.4, 
            'padding_idx': 0,
            'activation': activation
        }
        
        st.header(f"Input and Output ({dataset_choice})")
        
        default_text = "sherlock holmes was a" if dataset_prefix == "nl" else "if ( x > 0 ) {"
        seed_text = st.text_area("Enter your seed text:", default_text)
        
        if st.button("Generate Text"):
            
            model_to_use = load_model(model_filename, model_config)
            
            if model_to_use:
                with st.spinner(f"Generating text using '{model_filename}'..."):
                    output_text = generate_text(
                        model=model_to_use,
                        seed_text=seed_text,
                        num_words_to_gen=num_words,
                        block_size=block_size,
                        word_to_idx=word_to_idx,
                        idx_to_word=idx_to_word,
                        temperature=temperature,
                        device='cpu'
                    )
                    
                    st.subheader("Generated Sequence")
                    st.markdown(f"**{seed_text}**{output_text[len(seed_text):]}")
                    st.info(f"FYI: Any word in your seed text not in the {vocab_size}-word vocabulary was mapped to the '<UNK>' token.")
else:
    st.error("Model directory not found. The download may have failed.")