#from transformers import AutoTokenizer, AutoModel
from transformers import GPT2Tokenizer, GPT2Model

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_text_encoder():
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2Model.from_pretrained("distilgpt2")
    model.config.output_hidden_states = True
    model.to(device)
    model.eval()
    return tokenizer, model

def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Usamos el último estado oculto del último token como embedding representativo
    hidden_states = outputs.hidden_states       #UNA TUPLA DE CAPAS
    if hidden_states is None:
        raise ValueError("hidden_states sigue siendo None. Algo está mal con el modelo.")
    last_hidden = hidden_states[-1]             #ÚLTIMA CAPA
    text_embedding = last_hidden[:, -1, :]      #ÚLTIMO TOKEN
    return text_embedding
