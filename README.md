# DriveGuard — ML

Modelos de machine learning do **DriveGuard**, sistema de detecção de sonolência
em motoristas de caminhões e ônibus (TCC 2026).

Dois modelos convivem aqui:

| Modelo | Arquivo | Papel |
|---|---|---|
| **Detecção de sonolência** | `models/DrowsinessDetection.py` | Ensemble de votação embarcado no veículo. É o modelo de produção. |
| Detecção de risco de acidente | `models/AccidentDetection.py` | Random Forest sobre os dados abertos da PRF, para a análise de negócio. |

---

## 1. O modelo de sonolência: ensemble por votação

O `VotingClassifier` substituiu a avaliação de modelos isolados (Random Forest e
AdaBoost medidos separadamente). São três estimadores votando em **soft voting**:

| Estimador | Peso | Por que está no ensemble |
|---|---|---|
| `RandomForestClassifier` | 2 | Bagging: variância baixa, resiste a um landmark mal detectado num frame isolado |
| `GradientBoostingClassifier` | 2 | Boosting: captura a interação entre features (PERCLOS alto *junto com* pitch negativo) que a floresta dilui |
| `LogisticRegression` | 1 | Modelo linear calibrado; regulariza os dois anteriores quando eles concordam por overfitting |

### Por que soft voting

O produto não precisa só da classe — precisa de um **score contínuo de 0 a 100**
para emitir alertas progressivos *antes* do estado crítico. A votação soft devolve
a distribuição de probabilidade, e o score é a esperança do risco sob ela:

```
score = 100 × Σ P(classe) × peso_risco(classe)

peso_risco:  alerta = 0,0   fadiga = 0,5   sonolento = 1,0
```

Isso resolve a pendência apontada na dissertação (seção 8, *"definir a fórmula
utilizada para calcular o score"*) sem inventar pesos arbitrários sobre as
features. O score fica:

- **contínuo** — o alerta pode ser progressivo em vez de binário;
- **monotônico** — mais massa de probabilidade em `sonolento` sempre aumenta o score;
- **sensível à incerteza** — um caso ambíguo entre `alerta` e `sonolento` cai no
  meio da escala, não num extremo.

A escala casa com a do dashboard (`riskFromScore` em `mockData.ts`):
`<35` baixo · `<60` médio · `<80` alto · `>=80` crítico.

### SMOTE dentro do pipeline

O SMOTE agora é um passo do `imblearn.pipeline.Pipeline`, não uma etapa manual
antes do split. A diferença importa: aplicado antes do split, ele sintetiza
amostras a partir de exemplos que depois caem no teste — o modelo é avaliado com
dados derivados do que já viu e a métrica sai otimista. Dentro do pipeline, o
imblearn garante que ele só rode na porção de treino de cada fold da validação
cruzada.

O treino também reporta a **acurácia de cada estimador isolado**, para quantificar
o ganho do ensemble — número direto para o comparativo do TCC.

---

## 2. Conexão com a AWS

O repositório [`Drive-Guard/Infra`](https://github.com/Drive-Guard/Infra)
provisiona a nuvem. Este repositório conversa com ela em dois pontos:

```
┌──────────────── VEÍCULO ────────────────┐
│ Câmera → MediaPipe Face Mesh            │
│   → EAR, MAR, head pose                 │
│   → PERCLOS + blink rate (janela 60s)   │
│   → VotingClassifier → score 0-100      │
│   → ALERTA LOCAL (funciona offline)     │
│   → DriveGuardClient (lote + buffer)    │
└────────────────────┬────────────────────┘
                     │ POST /v1/eventos (x-api-key)
                     ▼
        API Gateway → Lambda → S3 Bronze
                     → RDS silver → gold → Dashboard
```

E o modelo treinado sobe para o bucket de artefatos:

```
DrowsinessDetection.publicar_modelo_no_s3()
    → s3://<artifacts>/modelos/voting/<versão>/modelo_voting.joblib
    → s3://<artifacts>/modelos/voting/<versão>/metricas.json
```

### Garantias de projeto

1. **O alerta nunca depende da rede.** O classificador roda no veículo e o alerta
   é local. O envio à nuvem acontece depois, em lote. Rede caiu, motorista segue
   protegido.
2. **Envio em lote com buffer em disco.** Um caminhão passa horas sem sinal. As
   leituras se acumulam e sobem juntas; se o POST falhar, o lote vai para
   `data/buffer_offline.jsonl` e é reenviado por `drenar_buffer_offline()`.
3. **Nenhum dado pessoal sai do veículo.** `motorista_id` e `placa` viram SHA-256
   antes de qualquer envio, e nenhuma imagem facial é transmitida — o frame é
   processado e descartado na mesma iteração (LGPD / privacy by design).

---

## 3. Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Configuração

Os valores saem do `terraform output` do repositório Infra:

```bash
export DRIVEGUARD_API_URL=$(terraform output -raw api_invoke_url)
export DRIVEGUARD_API_KEY=$(terraform output -raw api_key_value)
export DRIVEGUARD_ARTIFACTS_BUCKET=$(terraform output -raw artifacts_bucket)
export DRIVEGUARD_DEVICE_ID=dg-edge-0001
export DRIVEGUARD_MOTORISTA_ID=12345678901     # vira hash antes de sair
export DRIVEGUARD_PLACA=ABC1D23                # idem
```

No PowerShell:

```bash
$env:DRIVEGUARD_API_URL = terraform output -raw api_invoke_url
```

Sem `DRIVEGUARD_API_URL` e `DRIVEGUARD_API_KEY`, o cliente roda em modo offline e
grava tudo no buffer local — útil para testar sem a nuvem no ar.

## 5. Uso

```bash
python main.py treinar      # treina o ensemble e salva data/modelo_voting.joblib
python main.py publicar     # envia o modelo ao S3 e as leituras à API
python main.py monitorar    # inferência ao vivo na câmera
python main.py acidentes    # treina o modelo de acidentes (PRF)
```

Opções úteis:

```bash
python main.py treinar --sem-cv --sem-grafico
python main.py monitorar --fonte video.mp4 --duracao 120 --sem-video
python main.py publicar --limite 500
```

### Dados de treino

`main.py treinar` espera `data/features_data_trusted.csv`. Se o arquivo não
existir, o `FeatureEngineering` extrai as features de `data/train/`, que deve
seguir a estrutura:

```
data/train/
├── active/     # motorista alerta
└── fatigue/    # motorista em fadiga
```

---

## 6. Estrutura

```
ML/
├── main.py                            # CLI: treinar | publicar | monitorar | acidentes
├── requirements.txt
├── interfaces/BaseModelo.py
├── models/
│   ├── DrowsinessDetection.py         # ensemble de votação + score + publicação
│   ├── DrowsinessStream.py            # inferência ao vivo (câmera → alerta → AWS)
│   └── AccidentDetection.py           # Random Forest sobre dados da PRF
└── utilitaries/
    ├── DriveGuardClient.py            # cliente da API de ingestão (lote + retry + buffer)
    ├── FeatureEngineering.py          # extração de EAR, MAR e head pose
    ├── Utils.py                       # métricas, split, relatórios
    └── model_assets/face_landmarker.task
```

---

## 7. Métricas temporais no `DrowsinessStream`

Três métricas não saem de um frame isolado — precisam de janela:

| Métrica | Como é calculada |
|---|---|
| **PERCLOS** | % de frames com EAR abaixo do limiar, em janela de 60s |
| **Blink rate** | transições aberto→fechado→aberto, extrapoladas para piscadas/min |
| **Duração do fechamento** | média do tempo entre fechar e reabrir os olhos |

O limiar de "olho fechado" **não é fixo**: cada motorista tem uma abertura ocular
própria. Nos primeiros 10 segundos o sistema coleta amostras e usa a **mediana**
(não a média, que seria puxada pelas piscadas durante a calibração) como baseline;
o limiar passa a ser 70% desse valor. Enquanto não calibra, usa o valor da
literatura.

---

## 8. Limitações conhecidas

- **O `FeatureEngineering` rotula duas classes** (`active`/`fatigue`), mas o
  produto trabalha com três (`alerta`/`fadiga`/`sonolento`). O ensemble suporta as
  três — falta o dataset intermediário rotulado.
- **PERCLOS e blink rate não entram no treino.** Hoje o modelo usa
  `ear, mar, pitch, yaw, roll`, porque são as features que o dataset de imagens
  estáticas oferece. Elas já são calculadas e enviadas à nuvem no `DrowsinessStream`;
  incluí-las no treino exige um dataset de **vídeo** (UTA-RLDD), não de frames soltos.
- **Caminho B do TCC (YOLO) não está aqui.** O comparativo com deep learning sobre
  o frame bruto continua pendente.
- **Sem registro de modelo.** O artefato sobe versionado para o S3, mas não há
  SageMaker Model Registry nem promoção entre estágios.

---

## Repositórios relacionados

| Repositório | Conteúdo |
|---|---|
| [`Drive-Guard/Infra`](https://github.com/Drive-Guard/Infra) | Infraestrutura AWS em Terraform |
| [`Drive-Guard/Site`](https://github.com/Drive-Guard/Site) | Dashboard |
| [`Drive-Guard/ETL`](https://github.com/Drive-Guard/ETL) | Tratamento dos dados abertos da PRF |
