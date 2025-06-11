import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

class TextEncoder(nn.Module):
    """
    Codificador de texto basado en DistilBERT.
    Convierte una oración en una secuencia de embeddings.
    """
    def __init__(self, device='cpu'):
        super(TextEncoder, self).__init__()

        # Cargamos el tokenizer y modelo preentrenado DistilBERT
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Congelamos los pesos del modelo para usarlo como extractor de características
        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device
        self.to(device)

    def forward(self, texts):
        """
        Recibe una lista de oraciones y devuelve sus embeddings.
        Parámetros:
            texts (List[str]): Lista de strings (oraciones)
        Retorna:
            embeddings (torch.Tensor): Tensor de tamaño (B, L, D)
        """
        # Tokenizamos el texto
        encoding = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=64  # puedes ajustar el máximo
        ).to(self.device)

        # Pasamos por el modelo y obtenemos los embeddings
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # outputs.last_hidden_state: (B, L, D)
        return outputs.last_hidden_state
