"""
fft_mascara.py — FFT aplicada às MÁSCARAS do pipeline.

Ideia: uma máscara binária limpa é feita de regiões grandes e suaves
(baixas frequências). Ruído, pontinhos isolados e bordas serrilhadas são
variações rápidas de um pixel para o outro (altas frequências). Medir quanto
da energia do espectro está nas altas frequências indica o quanto a máscara
é "ruidosa", e permite ver o efeito da morfologia no domínio da frequência.

Este módulo só ANALISA. Não altera máscaras nem imagens.
"""

import numpy as np

# Frequência de corte em ciclos/pixel (o máximo possível é 0.5).
# Acima dela, a energia é considerada "alta frequência".
CORTE_PADRAO = 0.1


def _preparar(matriz):
    """Converte para float 2D; máscaras viram 0/1."""
    a = np.asarray(matriz)
    if a.ndim != 2 or a.size == 0:
        raise ValueError("A entrada deve ser uma matriz 2D não vazia.")
    return (a > 0).astype(np.float64) if a.dtype == np.uint8 else a.astype(np.float64)


def _raio_frequencia(forma):
    """Distância de cada frequência à origem, em ciclos/pixel."""
    fy = np.fft.fftfreq(forma[0])[:, None]
    fx = np.fft.fftfreq(forma[1])[None, :]
    return np.hypot(fx, fy)


def razao_alta_frequencia(matriz, corte=CORTE_PADRAO):
    """
    Fração (0 a 1) da energia espectral acima da frequência de corte.

    O nível médio (componente DC) é removido antes da FFT; senão ele
    dominaria o espectro e esconderia o que interessa (as bordas e o ruído).
    Como o corte é dado em ciclos/pixel, imagens de tamanhos diferentes
    podem ser comparadas.
    """
    a = _preparar(matriz)
    a = a - a.mean()

    energia = np.abs(np.fft.fft2(a)) ** 2
    total = energia.sum()

    if total == 0:  # matriz constante: não há variação alguma
        return 0.0

    return float(energia[_raio_frequencia(a.shape) > corte].sum() / total)


def espectro_log(matriz):
    """Magnitude do espectro, centralizada e em escala log (para exibir)."""
    a = _preparar(matriz)
    a = a - a.mean()
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(a))))


def perfil_radial(matriz, bins=48):
    """
    Energia média por faixa de frequência (do centro para fora).
    Retorna (frequências em ciclos/pixel, energia média por faixa).
    Mostra de forma direta onde está a energia: baixas x altas frequências.
    """
    a = _preparar(matriz)
    a = a - a.mean()

    energia = np.abs(np.fft.fft2(a)) ** 2
    raio = _raio_frequencia(a.shape)

    limites = np.linspace(0, 0.5, bins + 1)
    faixa = np.clip(np.digitize(raio.ravel(), limites) - 1, 0, bins - 1)

    soma = np.bincount(faixa, weights=energia.ravel(), minlength=bins)
    contagem = np.bincount(faixa, minlength=bins)
    media = np.divide(soma, contagem, out=np.zeros(bins), where=contagem > 0)

    centros = (limites[:-1] + limites[1:]) / 2
    return centros, media


def salvar_comparacao_espectros(mascara_antes, mascara_depois, caminho,
                                nome="Método", corte=CORTE_PADRAO):
    """
    Figura com: máscaras, espectros (log) e perfil radial de energia,
    antes e depois do refinamento.
    """
    from matplotlib.figure import Figure  # sem depender de janela/backend

    itens = [("Bruta", mascara_antes), ("Refinada", mascara_depois)]

    fig = Figure(figsize=(15, 8))
    eixos = fig.subplots(2, 3)

    for linha, (titulo, mascara) in enumerate(itens):
        razao = razao_alta_frequencia(mascara, corte)

        eixos[linha, 0].imshow(mascara, cmap="gray", vmin=0, vmax=255)
        eixos[linha, 0].set_title(f"{nome} — máscara {titulo.lower()}")
        eixos[linha, 0].axis("off")

        eixos[linha, 1].imshow(espectro_log(mascara), cmap="gray")
        eixos[linha, 1].set_title(
            f"Espectro de magnitude ({razao * 100:.2f}% acima de {corte} c/px)"
        )
        eixos[linha, 1].axis("off")

    # Perfil radial das duas versões no mesmo gráfico.
    ax = eixos[0, 2]
    for titulo, mascara in itens:
        freq, energia = perfil_radial(mascara)
        ax.semilogy(freq, energia + 1e-12, label=titulo)
    ax.axvline(corte, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Frequência (ciclos/pixel)")
    ax.set_ylabel("Energia média")
    ax.set_title("Perfil radial do espectro")
    ax.legend()
    ax.grid(alpha=0.3)

    eixos[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig(str(caminho), dpi=120)