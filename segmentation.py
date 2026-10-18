
import cv2
import numpy as np

from image_processing import (
    converter_rgb,
    converter_hsv,
    converter_lab,
    converter_grayscale,
    _normalizar_mascara,
)


def criar_mascara_rgb(imagem_bgr, limite_inferior, limite_superior):
    """Cria uma máscara selecionando uma faixa de cores RGB."""
    imagem_rgb = converter_rgb(imagem_bgr)

    return cv2.inRange(
        imagem_rgb,
        np.array(limite_inferior, dtype=np.uint8),
        np.array(limite_superior, dtype=np.uint8),
    )


def criar_mascara_hsv(imagem_bgr, limite_inferior, limite_superior):
    """Cria uma máscara selecionando uma faixa no espaço HSV."""
    imagem_hsv = converter_hsv(imagem_bgr)

    return cv2.inRange(
        imagem_hsv,
        np.array(limite_inferior, dtype=np.uint8),
        np.array(limite_superior, dtype=np.uint8),
    )


def criar_mascara_lab(imagem_bgr, limite_inferior, limite_superior):
    """Cria uma máscara selecionando uma faixa no espaço LAB."""
    imagem_lab = converter_lab(imagem_bgr)

    return cv2.inRange(
        imagem_lab,
        np.array(limite_inferior, dtype=np.uint8),
        np.array(limite_superior, dtype=np.uint8),
    )


def criar_mascara_grayscale(imagem_bgr):
    """Cria uma máscara usando o limiar automático de Otsu."""
    imagem_gray = converter_grayscale(imagem_bgr)

    _, mascara = cv2.threshold(
        imagem_gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return mascara

def refinar_mascara(mascara):
	# Primeiro padroniza a máscara para uma imagem binária.
	mascara = _normalizar_mascara(mascara)
	# Elemento elíptico pequeno para suavizar ruídos preservando formas arredondadas.
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	# A abertura remove pequenos pontos brancos isolados.
	mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
	# O fechamento preenche pequenos buracos e une falhas curtas na região branca.
	return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

def abertura(mascara, tamanho_kernel = 3):
    # Cria um kernel elíptico para a operação de abertura.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamanho_kernel, tamanho_kernel))
    # Abertura: erosão seguida de dilatação, útil para remover ruídos.     
    # Aplica a abertura na máscara binária.
    mascara_aberta = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    return mascara_aberta 

def fechamento(mascara, tamanho_kernel = 3):
    # Cria um kernel elíptico para a operação de fechamento.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamanho_kernel, tamanho_kernel))
    # Fechamento: dilatação seguida de erosão, útil para preencher buracos.     
    # Aplica o fechamento na máscara binária.
    mascara_fechada = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    return mascara_fechada

def erosao(mascara, tamanho_kernel = 3):
    # Cria um kernel elíptico para a operação de erosão.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamanho_kernel, tamanho_kernel))
    # Aplica a erosão na máscara binária.
    mascara_erodida = cv2.erode(mascara, kernel, iterations=1)
    return mascara_erodida

def dilatacao(mascara, tamanho_kernel = 3):
    # Cria um kernel elíptico para a operação de dilatação.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamanho_kernel, tamanho_kernel))
    # Aplica a dilatação na máscara binária.
    mascara_dilatada = cv2.dilate(mascara, kernel, iterations=1)
    return mascara_dilatada

def gradiente(mascara):
    # Calcula o gradiente morfológico, que destaca as bordas da máscara.
    # Cria um kernel elíptico para a operação de gradiente.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Aplica o gradiente na máscara binária.
    mascara_gradiente = cv2.morphologyEx(mascara, cv2.MORPH_GRADIENT, kernel)
    return mascara_gradiente