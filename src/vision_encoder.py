## ESTE CÓDIGO CONTIENE UNA VERSIÓN SENCILLA DEL VISION TRANSFORMER (ViT)
## PARA EXTRAER REPRESENTACIONES (EMBEDDINGS) DE IMÁGENES

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

class VisionEncoder(nn.Module):
    """
    Codificador visual basado en Vision Transformer (ViT).
    Convierte una imagen en una secuencia de embeddings.
    """
    def __init__(self, device='cpu'):
        super(VisionEncoder, self).__init__()

        # CARGAMOS UN MODELO ViT BASE PREENTRENADO (vit_b_16 con pesos preentrenados en ImageNet)
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.model = vit_b_16(weights=weights)

        #ELIMINAMOS LA CAPA DE CLASIFICACIÓN (HEAD) YA QUE NO VAMOS A PREDECIR CLASES
        self.model.heads = nn.Identity()

        #CONGELAMOS LOS PARÁMETROS (PODEMOS DESCONGELARLO SI DESEAMOS fine-tuning)
        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device
        self.to(device)

    def forward(self, images):
        """
        Recibe un batch de imágenes (tensor) y devuelve los embeddings visuales.
        Parámetros:
            images (torch.Tensor): Tensor de tamaño (B, 3, H, W)
        Retorna:
            embeddings (torch.Tensor): Tensor de tamaño (B, D)
        """
        return self.model(images)