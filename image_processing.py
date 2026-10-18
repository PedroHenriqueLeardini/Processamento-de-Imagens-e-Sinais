import cv2
import numpy as np


def _validar_imagem(imagem):
	# Garante que as funções de conversão recebam dados de imagem utilizáveis.
	if not isinstance(imagem, np.ndarray) or imagem.size == 0:
		raise ValueError("A imagem deve ser um array NumPy não vazio.")
	# As funções deste módulo esperam uma imagem colorida BGR com 3 canais.
	if imagem.ndim != 3 or imagem.shape[2] != 3:
		raise ValueError("A imagem deve estar no formato BGR com três canais.")


def _normalizar_mascara(mascara):
	# Confirma que existe uma máscara válida antes de processá-la.
	if not isinstance(mascara, np.ndarray) or mascara.size == 0:
		raise ValueError("A máscara deve ser um array NumPy não vazio.")

	# Converte máscaras coloridas ou com canal único para uma matriz 2D.
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

	# Transforma qualquer valor positivo em branco (selecionado) e zero
	# nos demais pixels (não selecionados); o tipo uint8 é exigido pelo OpenCV.
	return np.where(mascara > 0, 255, 0).astype(np.uint8)


def carregar_imagem(caminho):
	# Abre o arquivo como imagem colorida no formato BGR padrão do OpenCV.
	imagem = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
	if imagem is None:
		raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho}")
	return imagem


def converter_rgb(imagem):
	_validar_imagem(imagem)
	# Troca a ordem dos canais BGR para RGB, usada por várias bibliotecas de visualização.
	return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)


def converter_hsv(imagem):
	_validar_imagem(imagem)
	# Converte para matiz, saturação e brilho, componentes úteis para analisar cores.
	return cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)


def converter_lab(imagem):
	_validar_imagem(imagem)
	# Converte para o espaço perceptual Lab, que separa luminosidade de informação de cor.
	return cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)


def converter_grayscale(imagem):
	_validar_imagem(imagem)
	# Reduz a imagem colorida a um canal de intensidade (tons de cinza).
	return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


def aplicar_mascara(imagem, mascara):
	_validar_imagem(imagem)
	# Padroniza a máscara para valores binários aceitos pelo OpenCV.
	mascara = _normalizar_mascara(mascara)
	# Exige alinhamento pixel a pixel para não aplicar a máscara incorretamente.
	if mascara.shape != imagem.shape[:2]:
		raise ValueError("A máscara deve ter as mesmas dimensões da imagem.")
	# Mantém os pixels brancos na máscara e zera os pixels fora da região selecionada.
	return cv2.bitwise_and(imagem, imagem, mask=mascara)


def criar_alpha(mascara):
	# Usa a máscara binária como canal alfa: 255 opaco e 0 transparente.
	return _normalizar_mascara(mascara)


def criar_imagem_transparente(imagem, mascara):
	_validar_imagem(imagem)
	# Normaliza a máscara e verifica que cada pixel corresponde à imagem.
	alpha = criar_alpha(mascara)
	if alpha.shape != imagem.shape[:2]:
		raise ValueError("A máscara deve ter as mesmas dimensões da imagem.")

	# Adiciona o quarto canal e o preenche com os níveis de transparência da máscara.
	imagem_bgra = cv2.cvtColor(imagem, cv2.COLOR_BGR2BGRA)
	imagem_bgra[:, :, 3] = alpha
	return imagem_bgra


def salvar_imagem(caminho, imagem):
	# Evita tentar gravar dados ausentes ou que não estejam em formato de array.
	if not isinstance(imagem, np.ndarray) or imagem.size == 0:
		raise ValueError("A imagem a salvar deve ser um array NumPy não vazio.")
	# cv2.imwrite retorna False quando não consegue gravar no caminho informado.
	if not cv2.imwrite(str(caminho), imagem):
		raise OSError(f"Não foi possível salvar a imagem: {caminho}")