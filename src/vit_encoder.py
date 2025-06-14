from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vit_model():
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    model.to(device)
    model.eval()
    return processor, model

def get_image_embedding(image_path, processor, model):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Extraemos el embedding del [CLS] token (posición 0)
    image_embedding = outputs.last_hidden_state[:, 0, :]
    return image_embedding


## ESTA PARTE ES PARA LA FUNCIÓN DE app.py YA QUE LA INTERFAZ Streamlit 
## TRABAJA CON UNA IMAGEN CARGADA DESDE EL NAVEGADOR, NO DESDE UNA RUTA.

def extract_image_features(image, processor, model):
    """
    Procesa una imagen PIL para extraer sus embeddings usando ViT.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Se esperaba una imagen PIL.Image")

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Usamos el embedding del token [CLS]
    image_embedding = outputs.last_hidden_state[:, 0, :]
    return image_embedding
