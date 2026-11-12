"""
backend_bridge.py — Ponte entre o Front (CustomTkinter) e o Back (pipeline).

Este é o ÚNICO ponto de contato entre as duas pastas:

    Front/remover_frontend_v2.py  ->  Front/backend_bridge.py  ->  back (../*.py)

O back NÃO é modificado: a ponte só importa e chama funções públicas dele
(ai, pipeline, segmentation, image_processing, fft_mascara). Toda a tradução
de formatos acontece aqui:

    Front (entrada)                      Back
    ---------------                      ----
    caminho do arquivo (str)       ->    imagem BGR (numpy, via carregar_imagem)
    modo "fundo" | "objeto"        ->    máscara direta | máscara invertida
    qualidade (10-100)             ->    refinar (morfologia) liga/desliga
    sensibilidade (0.1-1.0)        ->    limiar da IA = 1 - sensibilidade
    objeto (texto) / método        ->    objeto / metodo de pipeline

    Back (saída)                         Front (saída)
    ------------                         -------------
    BGRA (numpy)                   ->    PIL.Image RGBA
    proporcao_objeto, alta_freq…   ->    ResultadoProcessamento.metricas
    SegmentacaoInvalida/ValueError ->    exceção com mensagem legível (messagebox)
"""

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# O back usa imports "planos" (import ai, import pipeline...). Basta colocar
# a pasta do back no sys.path; nenhum arquivo do back precisa mudar.
PASTA_BACK = Path(__file__).resolve().parent.parent
if str(PASTA_BACK) not in sys.path:
    sys.path.insert(0, str(PASTA_BACK))

import cv2
from PIL import Image

import ai
import fft_mascara
import pipeline
from image_processing import carregar_imagem, criar_imagem_transparente
from segmentation import refinar_mascara


# ─── Contrato de dados ───
METODOS = {
    "IA (segmentação)": "ia",
    "HSV": "hsv",
    "LAB": "lab",
    "RGB": "rgb",
}

# Qualidade >= este valor liga o refinamento morfológico do back.
QUALIDADE_MINIMA_REFINO = 50


@dataclass
class ResultadoProcessamento:
    imagem: Image.Image
    caminho_saida: str
    tempo_segundos: float
    mensagem: str = ""
    metricas: dict = field(default_factory=dict)


# ─── Cancelamento cooperativo ───
_cancel_event = threading.Event()


def cancelar_backend():
    """Pede para o processamento parar na próxima etapa."""
    _cancel_event.set()


def _checar_cancelamento():
    if _cancel_event.is_set():
        raise InterruptedError("Processamento cancelado")


# ─── Conversões ───
def _bgra_para_pil(bgra) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA), "RGBA")


def limiar_da_sensibilidade(sensibilidade: float) -> float:
    """Sensibilidade alta detecta mais => limiar de probabilidade menor."""
    return min(0.95, max(0.05, 1.0 - float(sensibilidade)))


# ─── Chamada principal (substitui o mock do front) ───
def backend_processar(
    caminho_entrada: str,
    modo: str,
    qualidade: int,
    sensibilidade: float,
    callback_progresso: Callable[[int, str], None],
    objeto: Optional[str] = None,
    metodo: str = "ia",
) -> ResultadoProcessamento:
    """
    modo:   "fundo"  -> remove o fundo, mantém o objeto;
            "objeto" -> remove o objeto, mantém o fundo.
    objeto: o que segmentar ("cachorro", "gato"...). Vazio/None = qualquer
            objeto de primeiro plano (só vale para a IA; cores ignoram).
    metodo: "ia", "hsv", "lab" ou "rgb".
    """
    _cancel_event.clear()
    inicio = time.perf_counter()

    if modo not in ("fundo", "objeto"):
        raise ValueError(f"Modo inválido: {modo}")
    if metodo not in pipeline.METODOS_COR and metodo != "ia":
        raise ValueError(f"Método inválido: {metodo}")

    objeto = (objeto or "").strip() or None
    if metodo == "ia":
        ai.classes_do_objeto(objeto)  # ValueError claro se não for suportado

    callback_progresso(10, "Carregando imagem...")
    imagem = carregar_imagem(caminho_entrada)
    _checar_cancelamento()

    callback_progresso(30, "Segmentando..." if metodo != "ia"
                       else "Segmentando com IA (pode demorar na 1ª vez)...")
    if metodo == "ia":
        mascara_bruta = ai.segmentar_ia(
            imagem, objeto, limiar=limiar_da_sensibilidade(sensibilidade)
        )
    else:
        mascara_bruta = pipeline.gerar_mascara(imagem, metodo)
    _checar_cancelamento()

    callback_progresso(55, "Verificando máscara...")
    ok, motivo = pipeline.verificar_mascara(mascara_bruta)
    if not ok:
        raise pipeline.SegmentacaoInvalida(
            f"Não consegui identificar o objeto com segurança: {motivo} "
            "Tente outro objeto, outro método ou ajuste a sensibilidade."
        )
    _checar_cancelamento()

    refinar = int(qualidade) >= QUALIDADE_MINIMA_REFINO
    callback_progresso(80, "Refinando bordas..." if refinar else "Sem refinamento...")
    mascara_final = refinar_mascara(mascara_bruta) if refinar else mascara_bruta

    ok, motivo = pipeline.verificar_mascara(mascara_final)
    if not ok:
        raise pipeline.SegmentacaoInvalida(f"Após o refinamento: {motivo}")
    _checar_cancelamento()

    callback_progresso(95, "Finalizando...")
    mascara_alpha = mascara_final if modo == "fundo" else cv2.bitwise_not(mascara_final)
    bgra = criar_imagem_transparente(imagem, mascara_alpha)

    proporcao = pipeline.proporcao_objeto(mascara_final)
    metricas = {
        "proporcao_objeto": proporcao,
        "alta_freq_bruta": fft_mascara.razao_alta_frequencia(mascara_bruta),
        "alta_freq_final": fft_mascara.razao_alta_frequencia(mascara_final),
        "metodo": metodo,
        "refinado": refinar,
    }
    if modo == "fundo":
        mensagem = (f"O objeto ocupa {proporcao * 100:.1f}% da imagem; "
                    "o fundo foi tornado transparente.")
    else:
        mensagem = (f"O objeto ocupava {proporcao * 100:.1f}% da imagem; "
                    "ele foi removido e ficou transparente.")

    callback_progresso(100, "Concluído!")
    return ResultadoProcessamento(
        imagem=_bgra_para_pil(bgra),
        caminho_saida="",
        tempo_segundos=time.perf_counter() - inicio,
        mensagem=mensagem,
        metricas=metricas,
    )
