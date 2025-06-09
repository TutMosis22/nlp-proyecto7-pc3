# Proyecto 7: Transformer Multimodal Texto-Imágenes (Vision-Language)
Este proyecto implementa un pipeline basado en Transformers para procesar texto e imágenes simultáneamente. El objetivo es generar descripciones automáticas (captions) para imágenes combinando modelos de lenguaje con modelos de visión, utilizando mecanismos de atención cruzada (cross-attention).

## Descripción del Proyecto
Los modelos multimodales combinan información de diferentes fuentes, como texto e imágenes. Aquí se explora un enfoque común en visión-lenguaje:

- Extraer embeddings de imágenes usando un Vision Transformer (ViT), dividiendo la imagen en patches.

- Extraer embeddings de texto usando un Transformer y un tokenizer.

- Fusionar ambos mediante cross-attention, donde el texto puede “preguntar” sobre la imagen para generar descripciones coherentes.

## Objetivos Específicos
- Implementar un pipeline que procese texto e imágenes como entrada.

- Utilizar embeddings preentrenados para texto e imagen.

- Construir un decoder que combine ambos tipos de embeddings para generar image captions.

## Requisitos de instalación
1. Clona el repositorio
2. Crea y activa un entorno virtual
3. Instala las dependencias:

    pip install -r requirements.txt



## Autor: Andre C.

Universidad Nacional de Ingeniería