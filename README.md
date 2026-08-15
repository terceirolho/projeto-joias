# Projeto Joias

Este projeto utiliza da visão computacional e do aprendizado profundo para realizar o reconhecimento automático de atributos visuais em joias de uma marca brasileira. O modelo foi treinado com o intuito de ser capaz descrever atributos visuais de joias a partir de imagens, e gerar legendas estruturadas no formato: [CATEGORIA] [COR] com design [DESIGN]. ou [CATEGORIA] [COR] com design [DESIGN] e com [PEDRA] quando existente.

O treinamento segue uma estratégia de **transfer learning em cascata**:
Os atributos reconhecidos são:
*   **Categoria** (*Anel, Brinco, Colar*): **87,35%** de acurácia.
*   **Cor** (*Dourado, Prata, Dourado e Prata*): **89,10%** de acurácia.
*   **Design** (*Orgânico, Escultural, Minimalista, Figurativo, Geométrico, Letter, Maximalista*): **26,50%** de acurácia.
*   **Pedra** (*15 tipos distintos, incluindo Pérola, Zircônia, Quartzo, Pedra Natural, etc.*): **0,00%** de acurácia.

A arquitetura foi inspirada no estudo de *Alcalde-Llergo et al., 2025* ([jewelry_linguistics](https://github.com/jewelryling/jewelry_linguistics)). 

Dataset [CATEGORIA] → Dataset [COR] → Dataset [DESIGN] → Dataset [LEGENDA ESTRUTURADA]

---

### Exemplos de saída
 
```
Brinco dourado com design orgânico.
Colar prata com design escultural e com pérola.
Anel dourado com design figurativo.
```
 
---

## 🚀 Como Funciona o Fluxo do Projeto

O pipeline do projeto conecta o processamento de texto e imagem em uma arquitetura híbrida **CNN (Visão Computacional) + RNN (Processamento de Linguagem Natural)**:

````mermaid
flowchart TD
    A["Pré-processamento\nNormalização · Splits · Captions"] --> B["dataFunctions.py\nVocabulário e Tokenização"]
    B --> D["modelFunctions.py\nVGG16 + GRU"]
    C["config.py"] --> D
    D --> E["train.py\nTransfer Learning em Cascata\nDataset [CATEGORIA] → Dataset [COR] → Dataset [DESIGN] → Dataset [LEGENDA]"]
    E --> F["Modelos salvos\n.hdf5 · .pk1"]
    F --> G["test.py\nInferência e Avaliação"]
    G --> H["Métricas\nAcurácia · F1 · CCR · BLEU"]
````


## 🛠️ Pré-processamento e Dados - Foi complexo e precisou de uma análise profunda

### Normalização de Imagens
Antes de alimentar o modelo, as imagens físicas devem passar por uma padronização e ser usado no treinamento do modelo.
*   **Script:** `normalizacao_img.py`
*   **Ação:** Redimensiona as imagens para **900×900 pixels**, converte para o espaço de cores **RGB**, aplica preenchimento com **fundo branco** para manter a proporção sem distorcer a joia, e salva o resultado em formato **JPEG**.

### Aplicação de WeakLabeling 
**Análise visual com GPT-4o mini:** Novas colunas `Tipo_Imagem`, `Classes mistas`,  `Cor_Normalizada`, `Design_GPT`, `Confianca_Design`, `Justificativa_Design`,  e `Qtd_pecas`
*   **Script:** `splits.py`

### Splits 
Após tratar os dados disponiveis foi feito o split adequado para cada tarefa.
*   **Script:** `splits.py`

### Extrair_captions
Após splits foi extraido os captions de uma coluna nova `legendas` do csv para txt de cada dataset.
*   **Script:** `extrair.captions.py`

### Tratamento de Dataset Desbalanceado
Caso o seu conjunto de dados apresente uma disparidade muito grande na quantidade de imagens por classe, o pipeline oferece como alternativa o uso de Class Weights no treino:
*   **Ponderação de Perda** Utiliza o arquivo `class_weights.json` gerado no split para penalizar mais severamente os erros nas classes minoritárias durante o cálculo da *Loss Function*.
 
---
## Datasets
 
| Dataset | Tarefa | Vocabulário | Treino | Val | Teste |
|---|---|---|---|---|---|
| CATEGORIA | Classificar categoria | 3 tokens | 1.388 | 180 | 166 |
| COR | Classificar cor | 3 tokens | 859 | 279 | 266 |
| DESIGN | Classificar design | 7 tokens | 1.388 | 180 | 166 |
| LEGENDA ESTRUTURADA | Gerar legenda | 37 tokens | 1.388 | 180 | 166 |

 ### Divisão dos dados (SPLITs)
 
- **Proporção:** 60/20/20 por handle (não por imagem)
- **805 handles únicos:** 483 treino / 161 validação / 161 teste
- Dataset COR usa **só imagens still** (treino + val + teste)
- Demais datasets usam **still + lookbook no treino** e **só lookbook** em val/teste
---

## Rótulos por Tarefa
 
**CATEGORIA:** `Anel` · `Brinco` · `Colar`
 
**COR:** `Dourado` · `Prata` · `Dourado_e_Prata`
 
**DESIGN:** `Orgânico` · `Escultural` · `Minimalista` · `Figurativo` · `Geométrico` · `Letter` · `Maximalista`
 
**LEGENDA ESTRUTURADA** — template:
```
[CATEGORIA] [COR] com design [DESIGN].
[CATEGORIA] [COR] com design [DESIGN] e com [PEDRA].
```
---

## Arquitetura
| Componente | Especificação |
|---|---|
| Encoder | VGG16 pré-treinada na ImageNet |
| Saída do encoder | Vetor de 4.096 dimensões |
| Decoder | GRU com 256 neurônios |
| Fusão | Soma das representações visual e textual |
| Normalização de entrada | 900×900 pixels com padding branco → 224×224 |
| Marcadores de sequência | `startcap` / `endcap` |


## Configuração de Treinamento
 
| Parâmetro | Valor |
|---|---|
| Otimizador | Adam |
| Learning rate | 1×10⁻³ |
| Batch size | 32 |
| Épocas máximas | 50 |
| EarlyStopping patience | 10 |
| ReduceLROnPlateau | factor=0.2, patience=2, min_lr=1×10⁻⁵ |
| Dropout | 0.5 (duas camadas) |
| Seed | 42 |
| Embaralhamento | A cada época |
| Class weights | Método `balanced` do scikit-learn, cap em 5.0 |

---

 ## Resultados
 
| Tarefa | Métrica | Valor |
|---|---|---|
| CATEGORIA | Acurácia | **87,35%** |
| COR | Acurácia | **89,10%** |
| DESIGN | Acurácia | **25,30%** |
| LEGENDA ESTRUTURADA | CCR exato | **12,65%** |
| LEGENDA ESTRUTURADA | BLEU-4 | **0,4517** |
| LEGENDA ESTRUTURADA | Acurácia categoria | 86,75% |
| LEGENDA ESTRUTURADA | Acurácia cor | 68,07% |
| LEGENDA ESTRUTURADA | Acurácia design | 28,31% |
| LEGENDA ESTRUTURADA | Acurácia pedra | 0,00% |

---

## Trabalhos Futuros
 
- Classificação multirrótulo de design e treinar a partir do dataset somente com imagens still (produto em fundo branco)
- Dataset dedicado de pedra
- Ampliar dados das classes raras por coleta dirigida
- Testar EfficientNet, ResNet, Vision Transformer e CLIP
---


## Limitações
 
- **Pedra:** nunca prevista — 77,7% das imagens de treino sem pedra
- **Design:** fronteiras visuais subjetivas entre Escultural, Orgânico e Figurativo
- **Classes raras:** Dourado_e_Prata (31 imgs treino), Maximalista (56), Letter (65)
- **BLEU inflado:** tokens estruturais fixos respondem por 54% do BLEU-4


## 📁 Estrutura de Arquivos e Diretórios

```text
├── data/
│   ├── train_4096.pk1          # Features extraídas das imagens de treino (VGG16)
│   └── val_4096.pk1            # Features extraídas das imagens de validação (VGG16)
│
├── datasets/
│   ├── captions.txt            # Mapeamento global: imagem ──> legenda/classe
│   ├── train.txt               # Lista com os nomes das imagens de treino
│   ├── validation.txt          # Lista com os nomes das imagens de validação
│   ├── test.txt                # Lista com os nomes das imagens de teste
│   ├── class_weights.json      # Pesos calculados para compensar o desbalanceamento
│   ├── train/                  # Diretório com as imagens físicas de treino
│   ├── validation/             # Diretório com as imagens físicas de validação
│   └── test/                   # Diretório com as imagens físicas de teste
│
└── src/
    ├── config.py               # Configurações globais e hiperparâmetros
    ├── dataFunctions.py        # Funções para manipulação e tokenização de dados
    ├── modelFunctions.py       # Definição e arquitetura da rede (CNN + RNN)
    ├── train.py                # Script principal de treinamento do modelo
    └── test.py                 # Script de avaliação e geração de métricas

## 📝 Descrição dos Módulos (`src/`)

### ⚙️ `config.py`
Centraliza as constantes de configuração compartilhadas por todo o ecossistema do projeto:
*   **Hiperparâmetros:** `LEARNING_RATE`, `EMBEDDING_SIZE`, `OPTIMIZER` (Adam).
*   **NLP:** `EMBEDDING_NAME` (Embora há presença do BERTimbau, não foi usado pois foi feito um experimento anterior e visto que não obteve melhoras), tokens especiais `<startcap>` e `<endcap>`.
*   **Configurações do Modelo:** Dimensões de entrada de imagem padrão da VGG16 (224×224), funções de perda (`LOSS`) e métricas de monitoramento.

### 📊 `dataFunctions.py`
Responsável por estruturar a entrada de dados textuais.
*   `getData()`: Lê o arquivo `captions.txt` e retorna um dicionário estruturado `{imagem: legenda}`.
*   `getLexicon()`: Filtra e extrai o vocabulário único de todas as legendas.
*   `getDataArrays()`: Cruza os dicionários com as listas de splits (`train.txt`/`val.txt`).
*   `getTokenizers()`: Constrói os mapeamentos numéricos index-para-palavra (`idxtoword`) e palavra-para-index (`wordtoidx`).
*   `getTokensArrays()`: Transforma as legendas de texto puro em sequências numéricas de índices.
*   `getBERTimbauEmbeddings()`: Conecta ao modelo linguístico para extrair vetores semânticos de 768 dimensões para cada palavra do vocabulário. **Mas não foi usado 
*   `getEmbeddingMatrix()`: Estrutura a matriz final de pesos de embedding que será injetada na camada inicial da rede recorrente.

### 🧠 `modelFunctions.py`
Gerencia a arquitetura integrada do modelo.
*   **Módulo Vision (CNN):** Suporta o carregamento de backbones pré-treinados como VGG16, Inception ou MobileNet. A função `encode_image()` extrai o vetor de características denso de 4096 posições.
*   **Módulo Linguístico (RNN):** Constrói a rede recorrente usando células GRU (`build_gru_model()`) *Há LSTM, mas não foi usada (`build_lstm_model()`).
*   `compile_model()`: Compila o modelo com o otimizador e a função de perda.
*   `create_generator()`: Pipeline de dados customizado que entrega lotes (*batches*) de dados com embaralhamento ativo a cada nova época de treino.
*   `generate_caption()`: Recebe imagem e gera a legenda preditiva palavra por palavra até encontrar o token de encerramento.

### 🏋️‍♂️ `train.py`
Executa o fluxo completo de aprendizado da rede:
1. Carrega as referências de dados estruturados.
2. Extrai e gera o cache das características visuais na pasta `data/*.pk1` (via técnica de *Transfer Learning*).
3. Constrói o vocabulário a partir dos captions de cada dataset. *** BERTimbau não foi usado.
4. Aplica class weights carregados do class_weights.json para compensar o desbalanceamento entre classes.
5. Inicializa o treinamento aplicando callbacks de resiliência: `EarlyStopping` (interrompe o treino se o modelo parar de evoluir) e `ReduceLROnPlateau` (reduz a taxa de aprendizado ao encontrar um platô de perda).
6. Entrega o modelo final treinado no formato `.hdf5` e os tokenizadores correspondentes.

### 🧪 `test.py`
Avalia  o desempenho do modelo no conjunto de testes (`test/`):
1. Faz a inferência de novas imagens usando o método autoregressivo `generate_caption()`.
2. Avalia a exatidão através da métrica **CCR (Correct Classification Rate)** para verificar se a legenda gerada é 100% idêntica à esperada.
3. Mede a acurácia isolada por categorias (anel, brinco, colar) cor (dourado, prata e dourado e prata), design (organico, escultural, minimalista, figurativo, geométrico, maximalista, letter).
4. Plota matrizes de confusão e calcula métricas clássicas de classificação: Precisão, Recall e F1-Score.
5. Calcula a métrica **BLEU**, padrão internacional para avaliar a qualidade de textos gerados. **Apenas para a TAREFA 4 [GERAR LEGENDAS]
6. Consolida e exporta todos os relatórios estruturados em arquivos CSV dentro de `models/test_logs/` em especial o test da tarefa de gerar legendas.




## Como Usar

```bash
source /home/seu-ambiente/jewelry_env/bin/activate
cd /home/seu-ambiente/projetos/jewelry 
pip install -r requirements.txt
```

### 1. Preparar seu dataset

Adicione o arquivo `SEU_DATASET.csv` na raiz do projeto.  
*Espero que seu dataset esteja organizado e perfeito. Caso contrário, forças e boa sorte!
*(Não incluído no repositório por conter dados proprietários.)*

### 2. Treinamento dos modelos

```bash
# Dataset CATEGORIA (do zero)
python src/train.py \
  --train_path datasets/generico \
  --cnn vgg16 --rnn gru \
  --use_embedding false \
  --epochs 50 --neurons 256 --batch_size 32 \
  --use_class_weights true

# Dataset COR (transfer do CATEGORIA)
python src/train.py \
  --train_path datasets/normal \
  --use_class_weights true \
  --pretrained_model models/model_generico_vgg16_gru_False_50_256_32.hdf5

# Dataset DESIGN (transfer do COR)
python src/train.py \
  --train_path datasets/design \
  --use_class_weights true \
  --pretrained_model models/model_normal_vgg16_gru_False_50_256_32.hdf5

# Dataset LEGENDA ESTRUTURADA (transfer do DESIGN)
python src/train.py \
  --train_path datasets/completo-2 \
  --use_class_weights true \
  --pretrained_model models/model_design_vgg16_gru_False_50_256_32.hdf5
```

### 3. Avaliar os modelos

```bash
python src/test.py \
  --model models/model_DATASET_vgg16_gru_False_50_256_32.hdf5 \
  --test_path datasets/DATASET
```

### 4. Interface

A interface está disponível em:  
🔗 [https://letgonc-joias.hf.space](https://letgonc-joias.hf.space)

> O código da interface (`app.py`) não está incluído neste repositório.  
> Para executar localmente, acesse o repositório do Hugging Face Space:  
> [https://huggingface.co/spaces/letgonc/joias](https://huggingface.co/spaces/letgonc/joias)
