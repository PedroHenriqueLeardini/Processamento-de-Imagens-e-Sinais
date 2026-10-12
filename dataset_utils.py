
from pathlib import Path

import cv2
import numpy as np


from pathlib import Path

pasta_projeto = Path.cwd()

pasta_imagens = (
    pasta_projeto
    / "datasets/oxford_pets/images/images"
)

pasta_trimap = (
    pasta_projeto
    / "datasets/oxford_pets/annotations/annotations/trimaps"
)

print("Diretório atual:", pasta_projeto)
print("Pasta de imagens:", pasta_imagens)
print("Pasta de imagens existe:", pasta_imagens.exists())
print("Pasta de trimaps:", pasta_trimap)
print("Pasta de trimaps existe:", pasta_trimap.exists())

if pasta_imagens.exists():
    imagens = list(pasta_imagens.glob("*"))
    print("Arquivos encontrados nas imagens:", len(imagens))
    print("Exemplos:", [p.name for p in imagens[:5]])

if pasta_trimap.exists():
    mascaras = list(pasta_trimap.glob("*"))
    print("Arquivos encontrados nas trimaps:", len(mascaras))
    print("Exemplos:", [p.name for p in mascaras[:5]])


def converter_trimap_para_binaria(caminho_trimap):
    trimap = cv2.imread(
        str(caminho_trimap),
        cv2.IMREAD_GRAYSCALE
    )

    if trimap is None:
        raise FileNotFoundError(
            f"Não foi possível carregar: {caminho_trimap}"
        )

    # Inicialmente, tudo será considerado fundo.
    mascara = np.zeros(trimap.shape, dtype=np.uint8)

    # Valor 1: animal.
    mascara[trimap == 1] = 255

    # Valor 3: contorno incerto.
    # Nesta primeira versão, vamos preservá-lo
    # junto com o animal.
    mascara[trimap == 3] = 255

    return mascara


def gerar_ground_truth():
    pasta_imagens = Path(
        "datasets/oxford_pets/images/images"
    )

    pasta_trimap = Path(
        "datasets/oxford_pets/annotations/annotations/trimaps"
    )

    pasta_saida = Path(
        "datasets/oxford_pets/ground_truth/mascaras_binarias"
    )

    pasta_saida.mkdir(
        parents=True,
        exist_ok=True
    )

    extensoes = {".jpg", ".jpeg"}

    imagens = [
        caminho
        for caminho in pasta_imagens.iterdir()
        if caminho.suffix.lower() in extensoes
    ]

    total = 0

    for caminho_imagem in sorted(imagens):
        caminho_mascara = (
            pasta_trimap / f"{caminho_imagem.stem}.png"
        )

        if not caminho_mascara.exists():
            print(
                f"Anotação não encontrada: "
                f"{caminho_imagem.name}"
            )
            continue

        mascara = converter_trimap_para_binaria(
            caminho_mascara
        )

        caminho_saida = (
            pasta_saida / f"{caminho_imagem.stem}.png"
        )

        if not cv2.imwrite(str(caminho_saida), mascara):
            raise OSError(
                f"Não foi possível salvar: {caminho_saida}"
            )

        total += 1

    print(f"Máscaras geradas: {total}")


if __name__ == "__main__":
    gerar_ground_truth()
