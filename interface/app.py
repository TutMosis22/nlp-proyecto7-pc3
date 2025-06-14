import streamlit as st
from PIL import Image
import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vit_encoder import load_vit_model, get_image_embedding, extract_image_features
from src.text_encoder import load_text_encoder, encode_text
from src.caption_blip import generate_caption_blip
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="VisionTalk: Generador de Captions", page_icon="📸")
st.title("VisionTalk 📷")
st.markdown("""
**Sube una imagen y elige un modelo para generar un caption.**

Puedes comparar resultados entre **GPT2** (modelo no entrenado multimodal) y **BLIP** (modelo entrenado conjuntamente con texto e imagen).
""")

# --- CARGAR MODELOS ---
@st.cache_resource

def load_models():
    # ViT
    vit_processor, vit_model = load_vit_model()
    # GPT2
    gpt2_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    gpt2_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    gpt2_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    gpt2_model.eval()
    return vit_processor, vit_model, gpt2_tokenizer, gpt2_model

vit_processor, vit_model, gpt2_tokenizer, gpt2_model = load_models()

# --- SUBIDA DE IMAGEN ---
image_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

model_choice = st.selectbox("Selecciona el modelo de captioning", ["GPT2 (no entrenado multimodal)", "BLIP (preentrenado)"])

# --- BOTÓN DE PROCESAMIENTO ---
if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    if st.button("Generar caption"):
        with st.spinner("Generando caption..."):
            if model_choice.startswith("GPT2"):
                # Extraer embedding visual
                image_embedding = extract_image_features(image, vit_processor, vit_model)
                # Texto base
                prompt = "a photo of"
                inputs = gpt2_tokenizer(prompt, return_tensors="pt").to(gpt2_model.device)
                outputs = gpt2_model.generate(**inputs, max_length=20, pad_token_id=gpt2_tokenizer.eos_token_id)
                caption = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
                st.markdown(f"**Caption (GPT2):** {caption}")

            elif model_choice.startswith("BLIP"):
                caption = generate_caption_blip(image)
                st.markdown(f"**Caption (BLIP):** {caption}")

# --- PIE DE PÁGINA ---
st.markdown("""
---
Hecho con ♥ para el curso de Procesamiento del Lenguaje Natural
""")