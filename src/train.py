# train.py — Experimento 4
# Adaptado para português brasileiro com BERTimbau
# Original: Alcalde-Llergo et al. (2025)
# Modificações Exp 4:
#   - Seed fixa para random, NumPy e TensorFlow
#   - Class weights corrigido (cobre todos os índices do vocabulário)
#   - EarlyStopping patience=10
#   - ReduceLROnPlateau mantido
#   - Figuras salvas em models/train_logs/

import tensorflow.keras.preprocessing.image
from modelFunctions import CNNModel, RNNModel
import dataFunctions
import config
import numpy as np
import tensorflow as tf
import platform
import argparse
import pickle
import json
from matplotlib import pyplot as plt
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def plot_and_save(data1, data2, title, save_path):
    plt.figure()
    plt.plot(data1, label='train')
    plt.plot(data2, label='validation')
    plt.title(title)
    plt.ylabel(title)
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura salva em: {save_path}")


def carregar_class_weights(train_path, wordtoidx, vocab_size):
    cw_path = os.path.join(train_path, "class_weights.json")
    if not os.path.exists(cw_path):
        print(f"  ⚠️  class_weights.json não encontrado. Treinando sem class weights.")
        return None
    with open(cw_path, "r", encoding="utf-8") as f:
        cw_raw = json.load(f)

    # Cobre todos os índices do vocabulário com peso neutro 1.0
    cw_idx = {i: 1.0 for i in range(vocab_size)}

    # Sobrescreve apenas os tokens de classe com os pesos calculados
    encontrados = 0
    for classe, peso in cw_raw.items():
        if classe in wordtoidx:
            cw_idx[wordtoidx[classe]] = peso
            encontrados += 1

    print(f"  Class weights carregados: {cw_raw}")
    print(f"  Tokens mapeados: {encontrados}/{len(cw_raw)}")
    return cw_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to train an Image Captioning Model — Exp 4")

    parser.add_argument("--train_path", dest="train_path", type=str, required=True)
    parser.add_argument("--model_path", dest="model_path", type=str, default="models/")
    parser.add_argument("--cnn", dest="cnn_type", type=str, default="vgg16")
    parser.add_argument("--rnn", dest="rnn_type", type=str, default="gru")
    parser.add_argument("--use_embedding",
                        dest="use_embedding",
                        type=lambda x: str(x).lower() in ("true", "1", "yes", "y", "sim"),
                        default=False)
    parser.add_argument("--epochs", dest="epochs", type=int, default=50)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--neurons", dest="neurons", type=int, default=256)
    parser.add_argument("--rw_images", dest="rewrite_images", type=bool, default=False)
    parser.add_argument("--rw_model", dest="rewrite_model", type=bool, default=False)
    parser.add_argument("--pretrained_model", dest="pretrained_model", type=str, default=None)
    parser.add_argument("--use_class_weights",
                        dest="use_class_weights",
                        type=lambda x: str(x).lower() in ("true", "1", "yes", "y", "sim"),
                        default=True)

    args = parser.parse_args()

    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.join("models", "train_logs"), exist_ok=True)

    print("Getting data...")
    train_images_path = os.path.join(args.train_path, "train")
    val_images_path   = os.path.join(args.train_path, "validation")
    captions_path     = os.path.join(args.train_path, "captions.txt")
    train_list        = os.path.join(args.train_path, "train.txt")
    val_list          = os.path.join(args.train_path, "validation.txt")

    data, max_length = dataFunctions.getData(captions_path)
    lex = dataFunctions.getLexicon(data)

    train_images_array, train_captions_array = dataFunctions.getDataArrays(data, train_list)

    print("Embaralhando dados de treino...")
    train_combined = list(zip(train_images_array, train_captions_array))
    random.shuffle(train_combined)
    train_images_array, train_captions_array = zip(*train_combined)
    train_images_array   = list(train_images_array)
    train_captions_array = list(train_captions_array)

    val_images_array, val_captions_array = dataFunctions.getDataArrays(data, val_list)

    print("Tokenizing captions...")
    idxtoword, wordtoidx = dataFunctions.getTokenizers(lex)
    vocab_size = len(idxtoword) + 1

    train_token_captions_array = dataFunctions.getTokensArrays(train_captions_array, wordtoidx)
    val_token_captions_array   = dataFunctions.getTokensArrays(val_captions_array, wordtoidx)

    print("Creating CNN model...")
    cnn_model = CNNModel(args.cnn_type)

    if platform.system() == "Linux":
        name = args.train_path.split('/')[-1]
    else:
        name = args.train_path.split('\\')[-1]

    train_images_save = os.path.join("data", f'train_images_{name}_{cnn_model.get_output_dim()}.pk1')
    if not os.path.exists(train_images_save) or args.rewrite_images:
        print(f"Encoding images to {train_images_save}...")
        shape = (len(train_images_array), cnn_model.get_output_dim())
        train_encoded_images = np.zeros(shape=shape, dtype=np.float16)
        for i, img in enumerate(train_images_array):
            image_path = os.path.join(train_images_path, img)
            img_loaded = tensorflow.keras.preprocessing.image.load_img(
                image_path, target_size=(cnn_model.get_height(), cnn_model.get_width()))
            train_encoded_images[i] = cnn_model.encode_image(img_loaded)
        with open(train_images_save, 'wb') as f:
            pickle.dump(train_encoded_images, f)
        print("Saved encoded train images to disk")
    else:
        print(f"Loading images from {train_images_save}...")
        with open(train_images_save, 'rb') as f:
            train_encoded_images = pickle.load(f)

    val_images_save = os.path.join("data", f'val_images_{name}_{cnn_model.get_output_dim()}.pk1')
    if not os.path.exists(val_images_save) or args.rewrite_images:
        print(f"Encoding images to {val_images_save}...")
        shape = (len(val_images_array), cnn_model.get_output_dim())
        val_encoded_images = np.zeros(shape=shape, dtype=np.float16)
        for i, img in enumerate(val_images_array):
            image_path = os.path.join(val_images_path, img)
            img_loaded = tensorflow.keras.preprocessing.image.load_img(
                image_path, target_size=(cnn_model.get_height(), cnn_model.get_width()))
            val_encoded_images[i] = cnn_model.encode_image(img_loaded)
        with open(val_images_save, 'wb') as f:
            pickle.dump(val_encoded_images, f)
        print("Saved encoded validation images to disk")
    else:
        print(f"Loading images from {val_images_save}...")
        with open(val_images_save, 'rb') as f:
            val_encoded_images = pickle.load(f)

    embedding_matrix = None
    if args.use_embedding:
        print("Carregando BERTimbau (português brasileiro)...")
        embeddings = dataFunctions.getBERTimbauEmbeddings(lex)
        embedding_matrix = dataFunctions.getEmbeddingMatrix(embeddings, vocab_size, wordtoidx)

    print("Building model...")
    rnn_model = RNNModel(args.rnn_type, args.neurons, vocab_size, max_length,
                         cnn_model.get_output_dim(), config.OPTIMIZER, embedding_matrix)
    rnn_model.build_model()

    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"Carregando pesos do modelo pré-treinado: {args.pretrained_model}")
        pretrained   = tensorflow.keras.models.load_model(args.pretrained_model)
        transferidas = 0
        for layer in rnn_model.get_model().layers:
            try:
                pretrained_layer = pretrained.get_layer(layer.name)
                layer.set_weights(pretrained_layer.get_weights())
                transferidas += 1
                print(f"  ✅ {layer.name}")
            except:
                print(f"  ⚠️  {layer.name} → não transferida (dimensão diferente)")
        print(f"Pesos transferidos: {transferidas} camadas")

    rnn_model.compile_model()

    # Class weights — corrigido: cobre todos os índices do vocabulário
    class_weight = None
    if args.use_class_weights:
        class_weight = carregar_class_weights(args.train_path, wordtoidx, vocab_size)

    print("Training model...")

    dataset_name = args.train_path.rstrip('/').split('/')[-1]
    run_id       = f'{dataset_name}_{args.cnn_type}_{args.rnn_type}_{args.use_embedding}_{args.epochs}_{args.neurons}_{args.batch_size}'
    model_save   = os.path.join(args.model_path, f'model_{run_id}.hdf5')

    if not os.path.exists(model_save) or args.rewrite_model:
        train_generator = rnn_model.create_generator(
            train_encoded_images, train_token_captions_array, args.batch_size)
        val_generator = rnn_model.create_generator(
            val_encoded_images, val_token_captions_array, len(val_encoded_images))

        early_stopping = tensorflow.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1, mode='min')

        reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=2,
            mode='min', verbose=1, min_lr=1e-5, cooldown=0, min_delta=1e-5)

        history = rnn_model.get_model().fit(
            train_generator,
            epochs=args.epochs,
            steps_per_epoch=(len(train_encoded_images) // args.batch_size),
            validation_data=val_generator,
            validation_steps=(len(val_encoded_images) // args.batch_size),
            verbose=2,
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weight
        )

        log_dir = os.path.join("models", "train_logs")
        plot_and_save(
            history.history['accuracy'],
            history.history['val_accuracy'],
            f"Accuracy — {run_id}",
            os.path.join(log_dir, f"accuracy_{run_id}.png")
        )
        plot_and_save(
            history.history['loss'],
            history.history['val_loss'],
            f"Loss — {run_id}",
            os.path.join(log_dir, f"loss_{run_id}.png")
        )

        rnn_model.get_model().save(model_save)
        print(f'Saved model to {model_save}')

        with open(os.path.join(args.model_path, f'idxtoword_{run_id}.pk1'), 'wb') as f:
            pickle.dump(idxtoword, f)
            print("Saved idxtoword to disk")

        with open(os.path.join(args.model_path, f'wordtoidx_{run_id}.pk1'), 'wb') as f:
            pickle.dump(wordtoidx, f)
            print("Saved wordtoidx to disk")
    else:
        print(f'The model already exists at {model_save}')