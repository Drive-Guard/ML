# DriveGuard ML

## 🚗 O que é este projeto

DriveGuard ML é um repositório Python para deteção de sonolência do condutor usando extração de características faciais a partir de vídeo e webcam. O projeto também traz um esqueleto para um modelo de detecção de acidentes, mas o foco principal é a extração de features, treinamento de modelo e teste em tempo real.

---

## 🔍 O que ele faz

- Extrai características faciais de imagens ou vídeos usando o MediaPipe Face Landmarker.
- Calcula métricas relacionadas à sonolência:
  - `EAR` (Eye Aspect Ratio)
  - `MAR` (Mouth Aspect Ratio)
  - ângulos de pose da cabeça (`pitch`, `yaw`, `roll`)
  - `PERCLOS` (percentual de fechamento dos olhos em uma janela)
- Treina um classificador Random Forest com as features extraídas.
- Executa inferência em tempo real via webcam para detectar se o condutor está com sono.

---

## 📁 Estrutura do repositório

- `main.py` - script principal que carrega variáveis de ambiente e define o comportamento.
- `requirements.txt` - dependências Python.
- `data/` - local de armazenamento dos CSVs e arquivos temporários.
  - `temp/` - logs e arquivos temporários.
- `interfaces/` - definição da interface base de modelo.
  - `BaseModelo.py`
- `models/` - implementação dos modelos de machine learning.
  - `DrowsinessDetection.py` - treina, testa e carrega o modelo de sonolência.
  - `AccidentDetection.py` - pipeline de detecção de acidentes.
- `utilitaries/` - helpers, engenharia de features e utilitários.
  - `FeatureEngineering.py` - extrai as features faciais de imagem/vídeo/webcam.
  - `Utils.py` - carregamento de dados, limpeza, relatórios e funções matemáticas.
  - `model_assets/face_landmarker.task` - modelo MediaPipe para detecção facial.

---

## ⚙️ Como funciona

### Fluxo de detecção de sonolência

1. `main.py` carrega as variáveis do arquivo `.env`.
2. Se `MODEL_BEHAVIOR = EXTRACT_FEATURES`, o projeto extrai features de imagem ou vídeo.
3. Se `MODEL_BEHAVIOR = TRAIN`, o projeto carrega o CSV de features, treina o modelo, avalia e salva o resultado.
4. Se `MODEL_BEHAVIOR = TEST`, o projeto carrega um modelo treinado e executa inferência pela webcam.

### Extração de features

- Imagens: `FaceLandmarker` processa fotos categorizadas em pastas `active/` e `fatigue/`.
- Vídeos: processa frames, calcula métricas e agrupa valores em janelas de tamanho fixo.
- Webcam: captura frames em tempo real e aplica o modelo treinado para detectar sono.

---

## 🧩 Variáveis de ambiente (`.env`)

Crie um arquivo `.env` na raiz do repositório a partir do `.env.template`.

### Exemplo de `.env`

```env
MODEL_FILEPATH="data/saved_model.joblib"
FILE_TO_TRAIN_MODEL="data/features_data_video_trusted.csv"
MODEL_TRAIN_IMAGE_FILEPATH="data/train_images"
MODEL_TEST_IMAGE_FILEPATH="data/test_images"
MODEL_TRAIN_VIDEO_FILEPATH="data/train_videos"
MODEL_BEHAVIOR="TRAIN"
FEATURE_EXTRACTION_TYPE="VIDEO"
```

### Variáveis obrigatórias

- `MODEL_BEHAVIOR` - define o modo de execução:
  - `TEST` → testa o modelo na webcam.
  - `TRAIN` → treina o modelo a partir das features.
  - `EXTRACT_FEATURES` → extrai features de vídeo ou imagem.
- `MODEL_FILEPATH` - caminho para salvar ou carregar o modelo treinado.

### Variáveis para extração de features

- `FEATURE_EXTRACTION_TYPE` - `VIDEO` ou `IMAGE` (usado quando `MODEL_BEHAVIOR=EXTRACT_FEATURES`).
- `MODEL_TRAIN_VIDEO_FILEPATH` - caminho da pasta ou arquivo de vídeo para extração.
- `MODEL_TRAIN_IMAGE_FILEPATH` - caminho da pasta de imagens para extração.

### Variável para treino

- `FILE_TO_TRAIN_MODEL` - caminho do CSV com as features já extraídas para treinar o modelo.

---

## ▶️ Como executar

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Configure o arquivo `.env` com o comportamento desejado.

3. Execute:

```bash
python main.py
```

---

## 🧪 Fluxos típicos de uso

### 1) Extrair features de vídeo

- `MODEL_BEHAVIOR=EXTRACT_FEATURES`
- `FEATURE_EXTRACTION_TYPE=VIDEO`
- `MODEL_TRAIN_VIDEO_FILEPATH` = pasta/arquivo de vídeos

### 2) Extrair features de imagem

- `MODEL_BEHAVIOR=EXTRACT_FEATURES`
- `FEATURE_EXTRACTION_TYPE=IMAGE`
- `MODEL_TRAIN_IMAGE_FILEPATH` = pasta de imagens

### 3) Treinar o modelo

- `MODEL_BEHAVIOR=TRAIN`
- `FILE_TO_TRAIN_MODEL` = CSV de features
- `MODEL_FILEPATH` = local para salvar o modelo (`.joblib`)

### 4) Testar com webcam

- `MODEL_BEHAVIOR=TEST`
- `MODEL_FILEPATH` = caminho para um modelo já treinado

---

## 💡 Observações importantes

- `main.py` usa `python-dotenv` para carregar as variáveis do `.env`.
- É necessário ter o arquivo de modelo MediaPipe em `utilitaries/model_assets/face_landmarker.task`.
- O `AccidentDetection.py` está presente no repositório, mas não é chamado diretamente por `main.py`.