# src/multimodal_decoder.py

import torch
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    """
    Una capa de atención cruzada: permite que queries (texto) accedan a la información visual (imagen).
    """
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key_value):
        # query = texto, key_value = imagen
        attn_output, _ = self.multihead_attn(query=query, key=key_value, value=key_value)
        # Residual + LayerNorm
        return self.norm(query + self.dropout(attn_output))


class MultimodalDecoder(nn.Module):
    """
    Decoder que genera captions a partir de embeddings de imagen y texto.
    """
    def __init__(self, embed_dim=768, num_heads=8, vocab_size=30522, max_len=64):
        super(MultimodalDecoder, self).__init__()

        self.cross_attn1 = CrossAttentionBlock(embed_dim, num_heads)
        self.cross_attn2 = CrossAttentionBlock(embed_dim, num_heads)

        self.ln = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, vocab_size)  # SALIDA: logits sobre el vocabulario
        )

    def forward(self, text_embeds, image_embeds):
        """
        Aplica atención cruzada dos veces y proyecta a vocabulario.
        """
        x = self.cross_attn1(text_embeds, image_embeds)
        x = self.cross_attn2(x, image_embeds)
        x = self.ln(x)
        logits = self.mlp(x)
        return logits