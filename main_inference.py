import argparse
import torch
from PIL import Image

from src.vit_encoder import load_vit_model, get_image_embedding
#from src.text_encoder import get_text_embedding
from src.multimodal_decoder import load_multimodal_decoder, generate_caption
from src.text_encoder import encode_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Ruta a la imagen de entrada")
    parser.add_argument("--text", type=str, default="Describe la imagen", help="Prompt de texto base")
    args = parser.parse_args()

    print("📦 Cargando modelos...")
    vit_model, vit_processor = load_vit_model()
    tokenizer, decoder_model = load_multimodal_decoder()

    print("🖼️ Cargando imagen y extrayendo embedding visual...")
    image = Image.open(args.image).convert("RGB")
    image_embedding = get_image_embedding(args.image, vit_model, vit_processor)

    print("✍️ Procesando texto base y extrayendo embedding textual...")
    text_embedding = encode_text(args.text, tokenizer, decoder_model)

    print("🧠 Generando caption...")
    caption = generate_caption(image_embedding, text_embedding, decoder_model, tokenizer)

    print("\n📷 Caption generado:")
    print(">>>", caption)

if __name__ == "__main__":
    main()
