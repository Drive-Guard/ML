"""
Detecção de distração (olhar fora da via) e de rosto fora da câmera.

Distração é a cabeça virada (|yaw| acima do limiar) ou inclinada para baixo
sem o olho fechado (celular no colo), sustentada por `MIN_DISTRACAO_S`. Olhar
o retrovisor é rápido e não conta. Cabeça baixa COM olho fechado não é
distração: é sonolência, e fica com o modelo.

Rosto fora da câmera (`perda_de_face`) é a ausência de detecção por
`MIN_SEM_ROSTO_S`: câmera tampada, motorista fora do assento ou virado.

Os dois geram um evento por ocorrência (na transição), não um por quadro.
"""

from __future__ import annotations


class DistractionDetector:
    def __init__(
        self,
        limiar_yaw: float = 30.0,
        limiar_pitch_baixo: float = -20.0,
        min_distracao_s: float = 2.0,
        min_sem_rosto_s: float = 3.0,
    ) -> None:
        self.LIMIAR_YAW = limiar_yaw
        self.LIMIAR_PITCH_BAIXO = limiar_pitch_baixo
        self.MIN_DISTRACAO_S = min_distracao_s
        self.MIN_SEM_ROSTO_S = min_sem_rosto_s
        self._desviado_desde: float | None = None
        self._sem_rosto_desde: float | None = None
        self._distracao_ativa = False
        self._sem_rosto_ativo = False

    def atualizar(self, instante_s: float, rosto: bool, yaw: float = 0.0, pitch: float = 0.0, olho_fechado: bool = False) -> list[str]:
        """Processa um quadro e devolve os eventos novos: 'distracao', 'perda_de_face'."""
        eventos: list[str] = []

        if not rosto:
            self._desviado_desde = None
            self._distracao_ativa = False
            if self._sem_rosto_desde is None:
                self._sem_rosto_desde = instante_s
            if not self._sem_rosto_ativo and instante_s - self._sem_rosto_desde >= self.MIN_SEM_ROSTO_S:
                self._sem_rosto_ativo = True
                eventos.append("perda_de_face")
            return eventos

        self._sem_rosto_desde = None
        self._sem_rosto_ativo = False

        desviado = abs(yaw) >= self.LIMIAR_YAW or (pitch <= self.LIMIAR_PITCH_BAIXO and not olho_fechado)
        if not desviado:
            self._desviado_desde = None
            self._distracao_ativa = False
            return eventos

        if self._desviado_desde is None:
            self._desviado_desde = instante_s
        if not self._distracao_ativa and instante_s - self._desviado_desde >= self.MIN_DISTRACAO_S:
            self._distracao_ativa = True
            eventos.append("distracao")
        return eventos

    @property
    def distraido(self) -> bool:
        return self._distracao_ativa

    def segundos_desviado(self, instante_s: float) -> float:
        return 0.0 if self._desviado_desde is None else instante_s - self._desviado_desde
