
import cv2
import numpy as np
import matplotlib.pyplot as plt


def analisar_fft(imagem, mostrar=True):
    """
    Analisa uma imagem usando a Transformada Rápida de Fourier (FFT).

    Parâmetros:
        imagem: imagem colorida em BGR ou imagem em escala de cinza.
        mostrar: se True, exibe a imagem original e o espectro.

    Retorna:
        espectro: resultado complexo da FFT centralizada.
        magnitude: espectro de magnitude em escala logarítmica.
        imagem_reconstruida: imagem reconstruída a partir da FFT.
    """

    if imagem is None:
        raise ValueError("A imagem não foi carregada corretamente.")

    # 1. Converter a imagem para escala de cinza, se necessário
    if imagem.ndim == 3:
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    elif imagem.ndim == 2:
        imagem_cinza = imagem.copy()
    else:
        raise ValueError("A imagem deve ter 2 ou 3 dimensões.")

    # A FFT trabalha melhor com valores numéricos em ponto flutuante
    imagem_float = imagem_cinza.astype(np.float32)

    # 2. Calcular a Transformada Rápida de Fourier 2D
    espectro = np.fft.fft2(imagem_float)

    # 3. Centralizar as frequências baixas no centro do espectro
    espectro_centralizado = np.fft.fftshift(espectro)

    # 4. Calcular o espectro de magnitude
    # log1p facilita a visualização de valores muito diferentes
    magnitude = np.log1p(np.abs(espectro_centralizado))

    # 5. Reconstruir a imagem a partir do espectro original
    imagem_reconstruida = np.fft.ifft2(espectro).real

    # 6. Exibir os resultados
    if mostrar:
        fig, eixos = plt.subplots(1, 3, figsize=(16, 5))

        eixos[0].imshow(imagem_cinza, cmap="gray")
        eixos[0].set_title("Imagem original em escala de cinza")
        eixos[0].axis("off")

        eixos[1].imshow(magnitude, cmap="gray")
        eixos[1].set_title("Espectro de magnitude da FFT")
        eixos[1].axis("off")

        eixos[2].imshow(imagem_reconstruida, cmap="gray")
        eixos[2].set_title("Imagem reconstruída pela FFT")
        eixos[2].axis("off")

        plt.tight_layout()
        plt.show()

    return espectro_centralizado, magnitude, imagem_reconstruida


import cv2
import numpy as np
import matplotlib.pyplot as plt


def comparar_visualizacoes_fft(imagem):
    if imagem is None:
        raise ValueError("A imagem não foi carregada.")

    if imagem.ndim == 3:
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    else:
        cinza = imagem.copy()

    # Calcula a FFT bidimensional
    fft = np.fft.fft2(cinza)

    # Visualização 1: espectro sem shift
    magnitude_original = np.log1p(np.abs(fft))

    # Centraliza o espectro para facilitar a leitura
    fft_centralizada = np.fft.fftshift(fft)

    # Visualização 2: magnitude logarítmica
    magnitude_centralizada = np.log1p(
        np.abs(fft_centralizada)
    )

    # Visualização 3: fase
    fase = np.angle(fft_centralizada)

    fig, eixos = plt.subplots(1, 3, figsize=(16, 5))

    eixos[0].imshow(magnitude_original, cmap="gray")
    eixos[0].set_title("FFT sem shift")
    eixos[0].axis("off")

    eixos[1].imshow(magnitude_centralizada, cmap="gray")
    eixos[1].set_title("Magnitude logarítmica com shift")
    eixos[1].axis("off")

    eixos[2].imshow(fase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    eixos[2].set_title("Espectro de fase")
    eixos[2].axis("off")

    plt.tight_layout()
    plt.show()



import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualizar_fase_fft(imagem):
    if imagem is None:
        raise ValueError("A imagem não foi carregada.")

    # Converte BGR para escala de cinza
    if imagem.ndim == 3:
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    else:
        cinza = imagem.copy()

    # Converte para ponto flutuante
    cinza = cinza.astype(np.float64)

    # Calcula a FFT e centraliza as frequências
    fft = np.fft.fft2(cinza)
    fft_centralizada = np.fft.fftshift(fft)

    # Calcula fase e magnitude
    fase = np.angle(fft_centralizada)
    magnitude = np.abs(fft_centralizada)

    # Converte fase para graus apenas para exibição
    fase_graus = np.degrees(fase)

    # Valores calculados
    altura, largura = cinza.shape
    total_pixels = cinza.size
    fase_min = fase.min()
    fase_max = fase.max()
    magnitude_max = magnitude.max()

    # Frequência de maior magnitude, excluindo o componente DC
    magnitude_sem_dc = magnitude.copy()
    centro_y, centro_x = altura // 2, largura // 2
    magnitude_sem_dc[centro_y, centro_x] = 0

    indice_y, indice_x = np.unravel_index(
        np.argmax(magnitude_sem_dc),
        magnitude_sem_dc.shape
    )

    # Converte índices para frequências em ciclos por pixel
    frequencias_y = np.fft.fftshift(np.fft.fftfreq(altura))
    frequencias_x = np.fft.fftshift(np.fft.fftfreq(largura))

    freq_y = frequencias_y[indice_y]
    freq_x = frequencias_x[indice_x]

    # Gráfico e painel informativo
    fig, (ax, painel) = plt.subplots(
        1, 2, figsize=(12, 6),
        gridspec_kw={"width_ratios": [3, 1.5]}
    )

    grafico = ax.imshow(
        fase_graus,
        cmap="twilight",
        vmin=-180,
        vmax=180
    )
    ax.set_title("Espectro de fase da FFT")
    ax.axis("off")
    fig.colorbar(grafico, ax=ax, label="Fase (graus)")

    painel.axis("off")
    informacoes = (
        "VALORES CALCULADOS\n\n"
        f"Altura: {altura} pixels\n"
        f"Largura: {largura} pixels\n"
        f"Total de pixels: {total_pixels:,}\n\n"
        f"Fase mínima: {fase_min:.4f} rad\n"
        f"Fase máxima: {fase_max:.4f} rad\n\n"
        f"Maior magnitude: {magnitude_max:.4f}\n\n"
        "Frequência de maior magnitude\n"
        "(excluindo DC):\n"
        f"Horizontal: {freq_x:.4f} ciclos/pixel\n"
        f"Vertical: {freq_y:.4f} ciclos/pixel"
    )

    painel.text(
        0.02, 0.95, informacoes,
        va="top", ha="left",
        fontsize=10,
        family="monospace",
        transform=painel.transAxes
    )

    fig.tight_layout()
    plt.show()

    return {
        "dimensoes": (altura, largura),
        "total_pixels": total_pixels,
        "fase_min_rad": float(fase_min),
        "fase_max_rad": float(fase_max),
        "magnitude_max": float(magnitude_max),
        "frequencia_dominante": {
            "horizontal_ciclos_por_pixel": float(freq_x),
            "vertical_ciclos_por_pixel": float(freq_y)
        }
    }
