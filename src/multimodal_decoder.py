from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_multimodal_decoder():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    model.eval()
    return tokenizer, model

def generate_caption(image_embedding, text_embedding, decoder_model, tokenizer, max_length=30):
    # Sumamos embeddings como forma simple de fusión multimodal
    combined_embedding = image_embedding + text_embedding  # Ambos deben ser [1, 768]

    # Creamos un token de inicio (bos_token o cualquier string como "Imagen:")
    input_ids = tokenizer("Imagen:", return_tensors="pt").input_ids.to(device)

    # Usamos el embedding combinado como past_key_values (truco rápido y no exacto)
    # Esto es un truco: forzamos que el modelo imagine que "Imagen:" activa cierto estado
    with torch.no_grad():
        outputs = decoder_model.generate(
            input_ids,
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption