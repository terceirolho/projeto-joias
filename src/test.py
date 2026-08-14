# test.py — Experimento 3
# Adaptado para português brasileiro com BERTimbau
# Original: Alcalde-Llergo et al. (2025)
# Modificações Exp 3:
#   - Suporte a datasets: generico, normal, completo
#   - Matriz de confusão por categoria (generico, completo)
#   - Matriz de confusão por cor (normal)
#   - BLEU score para dataset completo
#   - Figuras salvas em models/test_logs/
#   - Suporte a modelos _v2

import argparse
import numpy as np
import pickle
import platform
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.preprocessing.image
from tensorflow import keras
from modelFunctions import CNNModel, RNNModel
import dataFunctions
import os
import unicodedata
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def str2bool(v):
    return str(v).lower() in ("true", "1", "yes", "y", "sim")


def searchCaption(caption, data):
    predicted_image = False
    img = None
    for i, cap in data.items():
        if caption == cap:
            predicted_image = True
            img = i
            break
    return predicted_image, img


def obtain_CCR(ccr, total):
    if total == 0:
        return 0
    return (ccr / total) * 100


def normalize_caption(caption):
    caption = caption.lower()
    caption = unicodedata.normalize("NFKD", caption)
    caption = "".join(ch for ch in caption if not unicodedata.combining(ch))
    caption = caption.replace("-", " ")
    return caption


# ── Detecta categoria ──
def detect_tipo_joia(caption, filename=None):
    texto = caption
    if filename:
        texto = f"{caption} {filename}"
    c = normalize_caption(texto)

    if any(x in c for x in ["anel", "aneis", "alianca", "falange", "solitario"]):
        return "anel"
    if any(x in c for x in ["colar", "colares", "gargantilha", "choker",
                              "corrente", "cordao", "pendulo"]):
        return "colar"
    if any(x in c for x in ["brinco", "brincos", "argola", "ear cuff",
                              "earhook", "cuff", "earjacket", "piercing"]):
        return "brinco"
    return "outro"


# ── Detecta cor ──
def detect_cor(caption):
    c = normalize_caption(caption)
    c = c.replace("_", " ")  # normaliza Dourado_e_Prata → dourado e prata
    # Ordem importa — "dourado com prata" e "dourado e prata" antes de "dourado"
    if "dourado com prata" in c or "dourado e prata" in c:
        return "Dourado com Prata"
    if "dourado" in c or "rose" in c or "rosê" in c:
        return "Dourado"
    if "prata" in c:
        return "Prata"
    return "outro"


def calcular_bleu(expected_array, obtained_array):
    smoother = SmoothingFunction().method1
    scores = {1: [], 2: [], 3: [], 4: []}
    for expected, obtained in zip(expected_array, obtained_array):
        ref = [expected.split()]
        hyp = obtained.split()
        for n in range(1, 5):
            weights = tuple([1/n] * n + [0] * (4 - n))
            score = sentence_bleu(ref, hyp, weights=weights, smoothing_function=smoother)
            scores[n].append(score)
    return {f"BLEU-{n}": round(np.mean(scores[n]), 4) for n in range(1, 5)}


def salvar_figura_bleu(bleu_scores, run_id, log_dir):
    labels = list(bleu_scores.keys())
    values = list(bleu_scores.values())
    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
    plt.ylim(0, 1)
    plt.title(f"BLEU Scores — {run_id}")
    plt.ylabel("Score")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha='center', va='bottom', fontsize=10)
    path = os.path.join(log_dir, f"bleu_{run_id}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura BLEU salva em: {path}")


def salvar_matriz(matrix, labels, run_id, titulo, log_dir):
    df_cm = pd.DataFrame(matrix, index=labels, columns=labels)
    sn.set(font_scale=1.2)
    plt.figure(figsize=(8, 6))
    sn.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 14})
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title(f"{titulo} — {run_id}")
    fig_path = os.path.join(log_dir, f"matriz_confusao_{run_id}.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Matriz salva em: {fig_path}")


def calcular_metricas(matrix, labels, log_dir, run_id):
    total    = matrix.sum()
    acertos  = matrix.diagonal().sum()
    acc_ger  = (acertos / total) * 100 if total else 0

    print(f"Acurácia geral: {round(acc_ger, 2)}%")

    rows = []
    for idx, label in enumerate(labels):
        verd  = matrix[idx, :].sum()
        prev  = matrix[:, idx].sum()
        acert = matrix[idx, idx]
        rec   = (acert / verd)  * 100 if verd else 0
        prec  = (acert / prev)  * 100 if prev else 0
        f1    = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0

        rows.append({
            "categoria":      label,
            "total_real":     int(verd),
            "total_previsto": int(prev),
            "acertos":        int(acert),
            "recall_ccr_%":   round(rec, 2),
            "precisao_%":     round(prec, 2),
            "f1_%":           round(f1, 2),
        })
        print(f"\n{label}: recall={round(rec,2)}% | precisão={round(prec,2)}% | F1={round(f1,2)}%")

    df = pd.DataFrame(rows)
    path = os.path.join(log_dir, f"metricas_{run_id}.csv")
    df.to_csv(path, index=False)
    print(f"\nMétricas salvas em: {path}")
    return acc_ger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to test Image Captioning models')
    parser.add_argument('--model',     dest="model_name",    type=str, required=True)
    parser.add_argument('--test_path', dest="test_path",     type=str, required=True)
    parser.add_argument("--rw_images", dest="rewrite_images",type=str2bool, default=False)
    args = parser.parse_args()

    log_dir = os.path.join("models", "test_logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("Getting data...")
    images_path   = os.path.join(args.test_path, "test")
    captions_path = os.path.join(args.test_path, "captions.txt")
    test_list     = os.path.join(args.test_path, "test.txt")

    data, max_length = dataFunctions.getData(captions_path)
    lex = dataFunctions.getLexicon(data)
    images_array, captions_array = dataFunctions.getDataArrays(data, test_list)

    print("Tokenizing captions...")
    idxtoword, wordtoidx = dataFunctions.getTokenizers(lex)
    vocab_size = len(idxtoword) + 1

    sep      = '/' if platform.system() == "Linux" else '\\'
    name     = args.test_path.rstrip('/').rstrip('\\').split(sep)[-1]
    filename = os.path.basename(args.model_name)
    filename = filename.replace('model_', '', 1).replace('.hdf5', '')
    if filename.startswith(f'{name}_'):
        filename = filename[len(f'{name}_'):]

    parts         = filename.split('_')
    cnn_type      = parts[0]
    rnn_type      = parts[1]
    use_embedding = parts[2]
    epochs        = parts[3]
    neurons       = parts[4]
    batch_size    = parts[5]
    suffix        = f"_{parts[6]}" if len(parts) > 6 else ""

    run_id = f"{name}_{cnn_type}_{rnn_type}_{use_embedding}_{epochs}_{neurons}_{batch_size}{suffix}"

    print("Creating CNN model...")
    cnn_model = CNNModel(cnn_type)

    images_save = os.path.join("data", f'test_images_{name}_{cnn_model.get_output_dim()}.pk1')
    if not os.path.exists(images_save) or args.rewrite_images:
        print(f"Encoding images to {images_save}...")
        shape = (len(images_array), cnn_model.get_output_dim())
        encoded_images = np.zeros(shape=shape, dtype=np.float16)
        for i, img in enumerate(images_array):
            image_path = os.path.join(images_path, img)
            img_loaded = tensorflow.keras.preprocessing.image.load_img(
                image_path, target_size=(cnn_model.get_height(), cnn_model.get_width()))
            encoded_images[i] = cnn_model.encode_image(img_loaded)
        with open(images_save, 'wb') as f:
            pickle.dump(encoded_images, f)
        print("Saved encoded test images to disk")
    else:
        print(f"Loading images from {images_save}...")
        with open(images_save, 'rb') as f:
            encoded_images = pickle.load(f)

    print("Loading model...")
    rnn_model = RNNModel(max_length=max_length)
    rnn_model.set_model(keras.models.load_model(args.model_name))
    print("Model loaded")

    model_dir = os.path.dirname(args.model_name)
    pk_base   = f'{name}_{cnn_type}_{rnn_type}_{use_embedding}_{epochs}_{neurons}_{batch_size}{suffix}'

    with open(os.path.join(model_dir, f'idxtoword_{pk_base}.pk1'), 'rb') as f:
        idxtoword = pickle.load(f)
        print("Loaded idxtoword from disk")

    with open(os.path.join(model_dir, f'wordtoidx_{pk_base}.pk1'), 'rb') as f:
        wordtoidx = pickle.load(f)
        print("Loaded wordtoidx from disk")
        print("================================")

    ccr = 0
    expected_array      = []
    obtained_array      = []
    expected_tipo_array = []
    obtained_tipo_array = []
    expected_cor_array  = []
    obtained_cor_array  = []

    for i in range(len(images_array)):
        image         = images_array[i]
        image_encoded = encoded_images[i]

        print("Expected image: ",  image)
        print("Expected caption:", data[image])
        caption = rnn_model.generate_caption(image_encoded, wordtoidx, idxtoword)
        print("Obtained caption:", caption)

        expected_caption = data[image]
        obtained_caption = caption

        expected_array.append(expected_caption)
        obtained_array.append(obtained_caption)

        # Categoria
        expected_tipo = detect_tipo_joia(expected_caption, image)
        obtained_tipo = detect_tipo_joia(obtained_caption)
        expected_tipo_array.append(expected_tipo)
        obtained_tipo_array.append(obtained_tipo)

        # Cor
        expected_cor = detect_cor(expected_caption)
        obtained_cor = detect_cor(obtained_caption)
        expected_cor_array.append(expected_cor)
        obtained_cor_array.append(obtained_cor)

        predicted_image, img = searchCaption(obtained_caption, data)
        if obtained_caption == expected_caption:
            ccr += 1
        elif not predicted_image:
            print("The obtained caption does not exist in the dataset")

        print("================================")

    total   = len(images_array)
    ccr_pct = obtain_CCR(ccr, total)
    print(f"\nCCR exato = {round(ccr_pct, 2)}%")
    print("---------------------")

    # ── Genérico(Categoria) e Completo(Legenda Estruturada) — matriz por categoria ──
    if name in ["generico", "completo", "completo-2"]:
        print("CONFUSION MATRIX - CATEGORIA")
        labels = ["anel", "colar", "brinco"]
        matrix = confusion_matrix(expected_tipo_array, obtained_tipo_array, labels=labels)
        print(matrix)
        calcular_metricas(matrix, labels, log_dir, run_id)
        salvar_matriz(matrix, labels, run_id, "Matriz de Confusão — Categoria", log_dir)

   # ── Normal(Cor) — matriz por cor ──
    if name == "normal":
        print("CONFUSION MATRIX - COR")
        labels = ["Dourado", "Prata", "Dourado com Prata"]
        matrix = confusion_matrix(expected_cor_array, obtained_cor_array, labels=labels)
        print(matrix)
        calcular_metricas(matrix, labels, log_dir, run_id)
        salvar_matriz(matrix, labels, run_id, "Matriz de Confusão — Cor", log_dir)

    # ── Design — matriz por design ──
    if name == "design":
        print("CONFUSION MATRIX - DESIGN")
        labels = ["Escultural", "Figurativo", "Geométrico", "Letter", "Maximalista", "Minimalista", "Orgânico"]
        matrix = confusion_matrix(expected_array, obtained_array, labels=labels)
        print(matrix)
        calcular_metricas(matrix, labels, log_dir, run_id)
        salvar_matriz(matrix, labels, run_id, "Matriz de Confusão — Design", log_dir)
        df_legendas = pd.DataFrame({
            "imagem":   images_array,
            "esperada": expected_array,
            "obtida":   obtained_array,
        })
        legendas_path = os.path.join(log_dir, f"legendas_{run_id}.csv")

    # ── BLEU — apenas Completo (Legenda Estruturada) ──
    if name in ["completo", "completo-2"]:
        print("\n--- BLEU SCORES ---")
        bleu = calcular_bleu(expected_array, obtained_array)
        for k, v in bleu.items():
            print(f"  {k}: {v}")

        df_legendas = pd.DataFrame({
            "imagem":   images_array,
            "esperada": expected_array,
            "obtida":   obtained_array,
        })
        legendas_path = os.path.join(log_dir, f"legendas_{run_id}.csv")
        df_legendas.to_csv(legendas_path, index=False, encoding="utf-8-sig")
        print(f"Legendas salvas em: {legendas_path}")

        salvar_figura_bleu(bleu, run_id, log_dir)

        bleu_df   = pd.DataFrame([bleu])
        bleu_path = os.path.join(log_dir, f"bleu_{run_id}.csv")
        bleu_df.to_csv(bleu_path, index=False)
        print(f"BLEU CSV salvo em: {bleu_path}")