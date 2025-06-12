import torch
import torch.nn as nn
from transformers import ViTModel, ViTFeatureExtractor

class ViTEncoder(nn.Module):
    """
    Extrae embeddings de imágenes usando un Vision Transformer (ViT) preentrenado.
    """
    def __init__(self, model_name='google/vit-base-patch16-224-in21k'):
        super(ViTEncoder, self).__init__()
        
        # CARGAMOS EL MODELO PRE ENTRENADO Y EL EXTRACTOR DE CARACTERÍSTICAS
        self.vit = ViTModel.from_pretrained(model_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

        #CONGELAMOS SUS PESOS PARA QUE NO SE ENTRENEN DE NUEVO
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, images):
        """
        Toma imágenes (tensor de forma [B, 3, 224, 224]) y devuelve embeddings [B, num_patches, dim].
        """
        outputs = self.vit(pixel_values=images)
        return outputs.last_hidden_state  # [batch_size, num_patches+1, hidden_size]
