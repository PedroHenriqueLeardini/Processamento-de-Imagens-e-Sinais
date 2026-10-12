import cv2
import numpy as np


def _validar_imagem(imagem):
	if not isinstance(imagem, np.ndarray) or imagem.size == 0:
		raise ValueError("A imagem deve ser um array NumPy não vazio.")
	if imagem.ndim != 3 or imagem.shape[2] != 3:
		raise ValueError("A imagem deve estar no formato BGR com três canais.")


def _normalizar_mascara(mascara):
	if not isinstance(mascara, np.ndarray) or mascara.size == 0:
		raise ValueError("A máscara deve ser um array NumPy não vazio.")

	if mascara.ndim == 3:
		if mascara.shape[2] == 3:
			mascara = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY)
		elif mascara.shape[2] == 4:
			mascara = cv2.cvtColor(mascara, cv2.COLOR_BGRA2GRAY)
		elif mascara.shape[2] == 1:
			mascara = mascara[:, :, 0]
		else:
			raise ValueError("A máscara deve ter um, três ou quatro canais.")
	elif mascara.ndim != 2:
		raise ValueError("A máscara deve ser bidimensional ou ter canais de cor.")

	return np.where(mascara > 0, 255, 0).astype(np.uint8)


def carregar_imagem(caminho):
	imagem = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
	if imagem is None:
		raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho}")
	return imagem


def converter_rgb(imagem):
	_validar_imagem(imagem)
	return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)


def converter_hsv(imagem):
	_validar_imagem(imagem)
	return cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)


def converter_lab(imagem):
	_validar_imagem(imagem)
	return cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)


def converter_grayscale(imagem):
	_validar_imagem(imagem)
	return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


def aplicar_mascara(imagem, mascara):
	_validar_imagem(imagem)
	mascara = _normalizar_mascara(mascara)
	if mascara.shape != imagem.shape[:2]:
		raise ValueError("A máscara deve ter as mesmas dimensões da imagem.")
	return cv2.bitwise_and(imagem, imagem, mask=mascara)


def refinar_mascara(mascara):
	mascara = _normalizar_mascara(mascara)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
	return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)


def criar_alpha(mascara):
	return _normalizar_mascara(mascara)


def criar_imagem_transparente(imagem, mascara):
	_validar_imagem(imagem)
	alpha = criar_alpha(mascara)
	if alpha.shape != imagem.shape[:2]:
		raise ValueError("A máscara deve ter as mesmas dimensões da imagem.")

	imagem_bgra = cv2.cvtColor(imagem, cv2.COLOR_BGR2BGRA)
	imagem_bgra[:, :, 3] = alpha
	return imagem_bgra


def salvar_imagem(caminho, imagem):
	if not isinstance(imagem, np.ndarray) or imagem.size == 0:
		raise ValueError("A imagem a salvar deve ser um array NumPy não vazio.")
	if not cv2.imwrite(str(caminho), imagem):
		raise OSError(f"Não foi possível salvar a imagem: {caminho}")