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
    # SUMAMOS EMBEDDINGS COMO FORMA SIMPLE DE FUNCIÓN MULTIMODAL
    combined_embedding = image_embedding + text_embedding  # AMBOS DEBEN SER [1, 768]

    # CREAMOS UN TOKEN DE INICIO (bos_token O CUALQUIER STRING COMO "Imagen:")
    input_ids = tokenizer("Imagen:", return_tensors="pt").input_ids.to(device)

    #USAMOS EL EMBEDDING COMBINADO COMO past_key_values (UN TRUQUITO RÁPIDO PERO QUE NO ES EXACTO)
    # ESTO ES UN TRUQUITO QUE ENCONTRÉ POR INTERNET: FORZAMOS QUE EL MODELO IMAGINE QUE "Imagen:" ACTIVA CIERTO ESTADO
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