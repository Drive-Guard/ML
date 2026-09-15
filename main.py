"""
DriveGuard ML — ponto de entrada.

Modos:

    python main.py treinar    # treina o ensemble de votação e salva o modelo
    python main.py publicar   # envia o modelo ao S3 e as leituras à API
    python main.py monitorar  # roda a inferência ao vivo na câmera
    python main.py acidentes  # treina o modelo de acidentes (dados da PRF)

A configuração da nuvem vem de variáveis de ambiente — os valores saem do
`terraform output` do repositório Drive-Guard/Infra:

    DRIVEGUARD_API_URL           https://xxxx.execute-api.us-east-1.amazonaws.com/v1
    DRIVEGUARD_API_KEY           chave do API Gateway
    DRIVEGUARD_DEVICE_ID         identificador do dispositivo embarcado
    DRIVEGUARD_MOTORISTA_ID      CPF/matrícula (convertido em hash antes de sair)
    DRIVEGUARD_PLACA             placa do veículo (idem)
    DRIVEGUARD_ARTIFACTS_BUCKET  bucket de artefatos, para publicar o modelo
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("driveguard")


def treinar(args) -> int:
    from models.DrowsinessDetection import DrowsinessDetection

    modelo = DrowsinessDetection()
    metricas = modelo.train_model(validacao_cruzada=not args.sem_cv)
    modelo.salvar_modelo()

    print("\nGanho do ensemble sobre os estimadores isolados:")
    for nome, acuracia in metricas.get("acuracia_por_estimador", {}).items():
        print(f"  {nome:<8} {acuracia:.4f}")

    if not args.sem_grafico:
        modelo.test_model(plotar_matriz=True)
    return 0


def publicar(args) -> int:
    from models.DrowsinessDetection import DrowsinessDetection

    modelo = DrowsinessDetection()
    modelo.carregar_modelo()

    uri = modelo.publicar_modelo_no_s3()
    if uri:
        print(f"Modelo publicado em {uri}")

    enviadas = modelo.publicar_leituras_na_api(limite=args.limite)
    print(f"Leituras enviadas a API de ingestao: {enviadas}")
    return 0


def monitorar(args) -> int:
    from models.DrowsinessStream import DrowsinessStream

    fonte = int(args.fonte) if str(args.fonte).isdigit() else args.fonte
    stream = DrowsinessStream(fonte=fonte, exibir_video=not args.sem_video)
    resumo = stream.executar(duracao_maxima_segundos=args.duracao)

    print("\nResumo da sessao:")
    for chave, valor in resumo.items():
        print(f"  {chave:<22} {valor}")
    return 0


def acidentes(_args) -> int:
    from models.AccidentDetection import AccidentDetection

    AccidentDetection().train_model()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="driveguard-ml", description=__doc__)
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("treinar", help="treina o ensemble de votacao")
    p.add_argument("--sem-cv", action="store_true", help="pula a validacao cruzada")
    p.add_argument("--sem-grafico", action="store_true", help="nao abre a matriz de confusao")
    p.set_defaults(func=treinar)

    p = sub.add_parser("publicar", help="envia modelo ao S3 e leituras a API")
    p.add_argument("--limite", type=int, default=None, help="maximo de leituras a enviar")
    p.set_defaults(func=publicar)

    p = sub.add_parser("monitorar", help="inferencia ao vivo na camera")
    p.add_argument("--fonte", default="0", help="indice da webcam ou caminho do video")
    p.add_argument("--duracao", type=int, default=None, help="segundos ate encerrar")
    p.add_argument("--sem-video", action="store_true", help="nao abre a janela de video")
    p.set_defaults(func=monitorar)

    p = sub.add_parser("acidentes", help="treina o modelo de acidentes (PRF)")
    p.set_defaults(func=acidentes)

    args = parser.parse_args()
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuario.")
        return 130
    except Exception:
        logger.exception("Falha ao executar '%s'", args.comando)
        return 1


if __name__ == "__main__":
    sys.exit(main())
