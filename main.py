
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from image_processing import (
    carregar_imagem,
    aplicar_mascara,
    criar_imagem_transparente,
)

nome = "Abyssinian_1"

caminho_imagem = (
    Path("datasets/oxford_pets/images/images")
    / f"{nome}.jpg"
)

caminho_mascara = (
    Path("datasets/oxford_pets/ground_truth/mascaras_binarias")
    / f"{nome}.png"
)

imagem = carregar_imagem(caminho_imagem)

mascara = cv2.imread(
    str(caminho_mascara),
    cv2.IMREAD_GRAYSCALE
)

if mascara is None:
    raise FileNotFoundError(caminho_mascara)

resultado = aplicar_mascara(imagem, mascara)

transparente = criar_imagem_transparente(
    imagem,
    mascara
)

Path("output").mkdir(
    parents=True,
    exist_ok=True
)

cv2.imwrite(
    "output/animal_recortado.png",
    resultado
)

cv2.imwrite(
    "output/animal_transparente.png",
    transparente
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mascara, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(transparente, cv2.COLOR_BGRA2RGBA))
plt.title("Fundo transparente")
plt.axis("off")

plt.tight_layout()
plt.show()
