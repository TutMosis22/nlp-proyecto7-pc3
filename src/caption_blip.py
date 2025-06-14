from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Detectar si hay GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el procesador y modelo BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Función para generar un caption con BLIP
#def generate_caption_blip(image_path):
def generate_caption_blip(image):
    image = image.convert("RGB")
    #image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption