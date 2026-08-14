# config.py
# Adaptado para português brasileiro com BERTimbau
# Original: Alcalde-Llergo et al. (2025)

START = "startcap"
STOP  = "endcap"

# BERTimbau — substitui SBW-vectors-300 (espanhol) e wiki-news-300d (inglês)
# Modelo BERT treinado em português brasileiro pelo NLP-USP
EMBEDDING_SIZE = 768
EMBEDDING_NAME = "neuralmind/bert-base-portuguese-cased"

"""VALUES FOR INCEPTION MODEL"""
INCEPTION_WIDTH      = 299
INCEPTION_HEIGHT     = 299
INCEPTION_OUTPUT_DIM = 2048

"""VALUES FOR VGG16 MODEL"""
VGG16_WIDTH      = 224
VGG16_HEIGHT     = 224
VGG16_OUTPUT_DIM = 4096

"""VALUES FOR MOBILENET MODEL"""
MOBILENET_WIDTH      = 224
MOBILENET_HEIGHT     = 224
MOBILENET_OUTPUT_DIM = 50176

"""VALUES FOR RNN MODEL"""
LOSS          = "categorical_crossentropy"
LEARNING_RATE = 1e-3
OPTIMIZER     = "adam"
METRICS       = ["accuracy"]