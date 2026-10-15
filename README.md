# DriveGuard ML

Aplicação Python para extração de características faciais de vídeos e treinamento de classificadores de sonolência. O ponto de entrada é `main.py`, que seleciona uma operação por variáveis de ambiente.

## Funcionalidades atuais

- Extração de `EAR`, `MAR`, pose da cabeça (`pitch`, `yaw`, `roll`) e `PERCLOS` com MediaPipe Face Landmarker.
- Treinamento de um classificador de votação (Random Forest + XGBoost) ou de uma rede neural Keras.
- Uso de webcam para utilização de modelo salvo.
- Gravação de CSVs de extração e logs em `data/`.

## Estrutura

- `main.py`: carrega `.env` e encaminha a execução.
- `interfaces/BaseModelo.py`: pipeline comum, leitura dos dados e divisão por participante.
- `models/DrowsinessDetectionVoting.py`: classificador Voting com Random Forest e XGBoost.
- `models/DrowsinessDetectionNeuralNetwork.py`: classificador Keras via SciKeras.
- `utilitaries/Orchestrator.py`: seleciona e executa modelo/operação.
- `utilitaries/FeatureEngineering.py`: extração de features de vídeo e webcam.
- `utilitaries/Utils.py`: operações com CSV, modelos e métricas.
- `utilitaries/model_assets/face_landmarker.task`: arquivo do MediaPipe usado na detecção facial.
- `data/`: datasets CSV, vídeos/imagens do conjunto UTA-RLDD e logs em `temp/`.

## Instalação

Use uma versão de Python compatível com as dependências e instale os pacotes declarados:

```bash
pip install -r requirements.txt
```

O arquivo `utilitaries/model_assets/face_landmarker.task` precisa estar presente. Execute os comandos a partir da raiz do repositório, pois o caminho do asset é relativo ao diretório atual.

## Configuração e execução

Crie `.env` na raiz com as variáveis correspondentes à operação. `.env.template` lista possíveis chaves, mas os caminhos estão vazios e algumas chaves listadas não são usadas pelo código.

| Variável | Uso |
| --- | --- |
| `MODEL_BEHAVIOR` | Obrigatória. Valores aceitos: `EXTRACT`, `TRAIN` ou `TEST`. |
| `MODEL_FILEPATH` | Obrigatória. Caminho do modelo a salvar/carregar; também escolhe o classificador pelo nome do arquivo. |
| `FILE_TO_TRAIN_MODEL` | CSV informado no modo `TRAIN`. |
| `MODEL_TRAIN_VIDEO_FILEPATH` | Diretório de vídeos informado no modo `EXTRACT`. |

O nome em `MODEL_FILEPATH` deve conter `voting` ou `neural_network` para selecionar o classificador correspondente. Por exemplo, `data/voting_model.joblib` seleciona Voting. Se nenhum dos termos aparecer, o orquestrador instancia os dois modelos, mas o fluxo de treino/teste usa o primeiro (Voting).

Configure o modo desejado e execute:

```bash
python main.py
```

### Extrair features de vídeo

```env
MODEL_BEHAVIOR=EXTRACT
MODEL_TRAIN_VIDEO_FILEPATH= Local dos arquivos usados para realizar a extração de features
```

O diretório informado deve conter as subpastas `active/` e `fatigue/`, com os vídeos dentro delas. O rótulo do CSV é calculado pelas regras de EAR/PERCLOS, MAR e pitch; o nome da subpasta não é usado para definir o rótulo.

### Treinar

```env
MODEL_BEHAVIOR=TRAIN
MODEL_FILEPATH= CSV de features
FILE_TO_TRAIN_MODEL= Local selecionado para salvar o modelo
```

O treino lê o CSV somente quando o nome do arquivo é exatamente `features_video_train.csv`. O CSV precisa conter `participant_id`, `label`, `ear_behavior`, `pitch_behavior` e `roll_behavior`. Outros caminhos recebidos nesse modo são tratados como entrada de vídeo para extração, não como CSV de treino. Como a extração gera `features_video_enhanced_train.csv`, esse arquivo não é consumido diretamente pelo fluxo de treino atual sem adequar o nome/fluxo.

### Testar pela webcam

```env
MODEL_BEHAVIOR=TEST
MODEL_FILEPATH= Local do modelo já treinado
```

Esse modo requer webcam e um modelo previamente salvo que é utilizado para capturar os valores do rosto e classificar em alerta ou sonolento.

## Integração com a AWS

O modelo pode enviar as leituras para a nuvem do DriveGuard, provisionada no repositório [`Drive-Guard/Infra`](https://github.com/Drive-Guard/Infra) (API Gateway → S3 → RDS → dashboard). A integração é opcional: sem `DRIVEGUARD_API_URL` no `.env`, todos os modos funcionam como antes.

### Variáveis

| Variável | Uso |
| --- | --- |
| `DRIVEGUARD_API_URL` | URL da API (`terraform output -raw api_invoke_url`). Liga o envio. |
| `DRIVEGUARD_API_KEY` | Chave da API (`terraform output -raw api_key_value`). |
| `DRIVEGUARD_MOTORISTA_ID` | Identificador do motorista. É convertido em SHA-256 antes do envio. |
| `DRIVEGUARD_PLACA` | Opcional. Placa do veículo, também enviada só como hash. |
| `DRIVEGUARD_DEVICE_ID` | Identificador do dispositivo. Padrão `dg-edge-0001`. |
| `DRIVEGUARD_ARTIFACTS_BUCKET` | Bucket de artefatos (`terraform output -raw artifacts_bucket`), usado no modo `PUBLISH`. |
| `DRIVEGUARD_PUBLISH_STEP` | No modo `PUBLISH`, envia uma linha a cada N do CSV. Padrão 30 (≈ 1 leitura por segundo de vídeo). |

### Modo `TEST` com envio

Com as variáveis preenchidas, o modo `TEST` envia uma leitura por segundo enquanto a webcam roda, mais um alerta toda vez que o modelo passa de acordado para sono. A classificação e o aviso na tela não dependem da rede: o envio acontece numa thread separada e, se falhar, o lote fica em `data/temp/buffer_offline.jsonl` e é reenviado ao fim da sessão.

### Modo `PUBLISH`

```env
MODEL_BEHAVIOR=PUBLISH
MODEL_FILEPATH= modelo treinado
FILE_TO_TRAIN_MODEL= CSV gerado pelo modo EXTRACT (opcional)
```

Envia o modelo para o S3 e, se um CSV for informado, classifica as linhas com o modelo e envia as leituras para a API. Cada `participant_id` vira um motorista no dashboard. Como os vídeos não têm horário real, o horário de cada leitura é reconstruído a partir de `minute` e `frame`, mantendo o intervalo real entre elas.

### O que é enviado

| Campo | Origem |
| --- | --- |
| `estado` | `predict()` do modelo: 0 → `alerta`, 1 → `sonolento` |
| `score_fadiga` | `100 × P(sono)` |
| `ear`, `mar`, `perclos`, `head_pitch`, `head_yaw`, `head_roll` | Valores suavizados pelas mesmas janelas usadas na predição |

`P(sono)` vem de `predict_proba`. Em modelos com `voting="hard"`, que não têm `predict_proba`, ela é a média das probabilidades dos membros do ensemble (RF e XGBoost). A classe enviada continua sendo a do `predict()`; só o score vem dessa média.

Nenhuma imagem sai da máquina, só os números acima.

