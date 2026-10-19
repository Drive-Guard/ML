"""
Detecção de piscadas a partir da série de EAR (Eye Aspect Ratio).

Funciona quadro a quadro, com o instante de cada quadro (segundos), então vale
tanto para a webcam (relógio real) quanto para vídeo (quadro / fps).

Uma piscada é o olho fechado (EAR abaixo do limiar) por pelo menos
`MIN_PISCADA_MS` (abaixo disso é ruído do landmark) e menos que
`FECHAMENTO_LONGO_MS`. A partir daí não é piscada: é fechamento prolongado
(microssono), contado à parte.

Métricas na janela deslizante (`JANELA_S`, 60 s por padrão):

* `blink_rate`             piscadas por minuto;
* `duracao_media_ms`       duração média das piscadas;
* `piscadas_lentas`        piscadas com duração >= `PISCADA_LENTA_MS`;
* `fechamentos_longos`     fechamentos >= `FECHAMENTO_LONGO_MS` (microssono);
* `duracao_olhos_fechados_ms`  o fechamento em curso agora, ou o último.

Referências usadas para os limiares: piscada espontânea de 100 a 400 ms e
frequência de 15 a 20 por minuto em vigília; piscadas mais lentas e mais
frequentes com a sonolência, e fechamentos acima de 500 ms a 1 s associados a
microssono (Schleicher et al., 2008; Caffier et al., 2003).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class Piscada:
    inicio: float
    duracao_ms: float


class BlinkDetector:
    def __init__(
        self,
        limiar_ear: float = 0.20,
        janela_s: float = 60.0,
        min_piscada_ms: float = 50.0,
        piscada_lenta_ms: float = 300.0,
        fechamento_longo_ms: float = 1000.0,
    ) -> None:
        self.LIMIAR_EAR = limiar_ear
        self.JANELA_S = janela_s
        self.MIN_PISCADA_MS = min_piscada_ms
        self.PISCADA_LENTA_MS = piscada_lenta_ms
        self.FECHAMENTO_LONGO_MS = fechamento_longo_ms

        self._fechado_desde: float | None = None
        self._piscadas: deque[Piscada] = deque()
        self._fechamentos_longos: deque[Piscada] = deque()
        self._ultimo_fechamento_ms = 0.0
        self._inicio: float | None = None
        self._agora = 0.0
        self.novo_fechamento_longo = False

    def atualizar(self, ear: float, instante_s: float) -> None:
        """Registra o EAR de um quadro. `instante_s` precisa ser crescente."""
        if self._inicio is None:
            self._inicio = instante_s
        self._agora = instante_s
        self.novo_fechamento_longo = False

        fechado = ear < self.LIMIAR_EAR
        if fechado and self._fechado_desde is None:
            self._fechado_desde = instante_s
        elif not fechado and self._fechado_desde is not None:
            self._encerrar_fechamento(instante_s)

        # Um fechamento que passa do limite conta como longo assim que passa,
        # sem esperar o olho abrir: o alerta precisa sair durante o microssono.
        if self._fechado_desde is not None:
            duracao = (instante_s - self._fechado_desde) * 1000
            ja_contado = self._fechamentos_longos and self._fechamentos_longos[-1].inicio == self._fechado_desde
            if duracao >= self.FECHAMENTO_LONGO_MS and not ja_contado:
                self._fechamentos_longos.append(Piscada(self._fechado_desde, duracao))
                self.novo_fechamento_longo = True

        self._descartar_antigos()

    def _encerrar_fechamento(self, instante_s: float) -> None:
        duracao = (instante_s - self._fechado_desde) * 1000
        self._ultimo_fechamento_ms = duracao
        if self.MIN_PISCADA_MS <= duracao < self.FECHAMENTO_LONGO_MS:
            self._piscadas.append(Piscada(self._fechado_desde, duracao))
        elif self._fechamentos_longos and self._fechamentos_longos[-1].inicio == self._fechado_desde:
            self._fechamentos_longos[-1].duracao_ms = duracao
        self._fechado_desde = None

    def _descartar_antigos(self) -> None:
        limite = self._agora - self.JANELA_S
        while self._piscadas and self._piscadas[0].inicio < limite:
            self._piscadas.popleft()
        while self._fechamentos_longos and self._fechamentos_longos[0].inicio < limite:
            self._fechamentos_longos.popleft()

    # ------------------------------------------------------------- métricas

    @property
    def olho_fechado_ms(self) -> float:
        """Duração do fechamento em curso (0 com o olho aberto)."""
        return 0.0 if self._fechado_desde is None else (self._agora - self._fechado_desde) * 1000

    def metricas(self) -> dict:
        inicio = self._agora if self._inicio is None else self._inicio
        decorrido = max(1e-6, min(self.JANELA_S, self._agora - inicio))
        n = len(self._piscadas)
        duracoes = [p.duracao_ms for p in self._piscadas]
        return {
            # Antes de completar a janela, extrapola pelo tempo já observado.
            "blink_rate": round(n * 60.0 / decorrido, 2) if decorrido >= 10 else None,
            "duracao_media_ms": round(sum(duracoes) / n, 1) if n else None,
            "piscadas_lentas": sum(1 for d in duracoes if d >= self.PISCADA_LENTA_MS),
            "fechamentos_longos": len(self._fechamentos_longos),
            "duracao_olhos_fechados_ms": int(self.olho_fechado_ms or self._ultimo_fechamento_ms),
        }
