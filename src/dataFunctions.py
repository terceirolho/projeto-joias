# dataFunctions.py
# Adaptado para português brasileiro com BERTimbau
# Original: Alcalde-Llergo et al. (2025)

import os
import numpy as np
import config
import torch
from transformers import BertTokenizer, BertModel


def getData(filename):
    """Lê o captions.txt e retorna dicionário {imagem: legenda} e tamanho máximo."""
    data       = dict()
    max_length = 0

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().split('\n'):
            line = line.split()
            if len(line) < 2:
                continue
            image      = line[0]
            caption    = line[1:]
            max_length = max(max_length, len(caption))
            if image not in data:
                data[image] = ' '.join(caption)

    max_length += 2  # espaço para tokens START e STOP
    return data, max_length


def getLexicon(data):
    """Extrai vocabulário único de todas as legendas."""
    lex = set()
    for key in data:
        [lex.update(d.split()) for d in data[key].split()]
    lex.update(config.START.split())
    lex.update(config.STOP.split())
    return lex


def getDataArrays(data, train_list):
    """Lê lista de imagens e retorna arrays de imagens e legendas com START/STOP."""
    images_array   = []
    captions_array = []

    with open(train_list, 'r', encoding='utf-8') as f:
        for image in f.read().split('\n'):
            image = image.strip()
            if not image or image not in data:
                continue
            images_array.append(image)
            captions_array.append(
                f'{config.START} {data[image]} {config.STOP}'
            )

    return images_array, captions_array


def getTokenizers(lex):
    """Cria mapeamentos palavra → índice e índice → palavra."""
    idxtoword = {}
    wordtoidx = {}
    idx = 1
    for word in lex:
        wordtoidx[word] = idx
        idxtoword[idx]  = word
        idx += 1
    return idxtoword, wordtoidx


def getTokensArrays(captions_array, wordtoidx):
    """Converte legendas em sequências de índices numéricos."""
    token_captions_array = []
    for caption in captions_array:
        tokens = []
        for word in caption.split():
            if word in wordtoidx:
                tokens.append(wordtoidx[word])
        token_captions_array.append(tokens)
    return token_captions_array


def getBERTimbauEmbeddings(lex):
    """
    Gera embeddings para cada palavra do vocabulário usando BERTimbau.

    BERTimbau é um modelo BERT treinado em português brasileiro
    pelo NLP-USP (neuralmind/bert-base-portuguese-cased).

    Substitui:
      - SBW-vectors-300-min5.txt (espanhol, 300 dims)
      - wiki-news-300d-1M.vec    (inglês,   300 dims)

    Retorna dicionário {palavra: vetor numpy 768 dims}
    """
    print(f'Carregando BERTimbau: {config.EMBEDDING_NAME}')

    # Carrega tokenizador e modelo BERTimbau do HuggingFace
    tokenizer = BertTokenizer.from_pretrained(config.EMBEDDING_NAME)
    model     = BertModel.from_pretrained(config.EMBEDDING_NAME)
    model.eval()  # modo inferência — desativa dropout

    embeddings = {}
    palavras   = list(lex)

    print(f'Gerando embeddings para {len(palavras)} palavras...')

    with torch.no_grad():  # sem gradiente — economiza memória
        for palavra in palavras:
            # Tokeniza a palavra individualmente
            inputs = tokenizer(
                palavra,
                return_tensors='pt',  # tensores PyTorch
                truncation=True,
                max_length=10
            )

            # Gera embedding via BERTimbau
            outputs = model(**inputs)

            # Média dos tokens como representação da palavra
            # last_hidden_state: (1, n_tokens, 768)
            vetor = outputs.last_hidden_state.mean(dim=1).squeeze()
            embeddings[palavra] = vetor.numpy()

    print(f'BERTimbau: {len(embeddings)} embeddings gerados (768 dims)')
    return embeddings


def getEmbeddingMatrix(embeddings, vocab_size, wordtoidx):
    """
    Monta a matriz de embeddings para inicializar a camada Embedding.
    Dimensão: (vocab_size x EMBEDDING_SIZE)
    """
    embedding_matrix = np.zeros((vocab_size, config.EMBEDDING_SIZE))

    encontrados = 0
    for word, i in wordtoidx.items():
        vetor = embeddings.get(word)
        if vetor is not None:
            embedding_matrix[i] = vetor
            encontrados += 1

    print(f'Embedding matrix: {encontrados}/{len(wordtoidx)} palavras mapeadas')
    return embedding_matrix


def getEmbeddings():
    """
    Método original do autor — mantido para compatibilidade.
    Lê arquivo de embeddings estáticos no formato word2vec.
    """
    embeddings      = {}
    embeddings_path = os.path.join('data', config.EMBEDDING_NAME)

    with open(embeddings_path, 'rb') as f:
        for line in f:
            values = line.split()
            word   = values[0]
            coefs  = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coefs

    print(f'Encontrados {len(embeddings)} word vectors')
    return embeddings