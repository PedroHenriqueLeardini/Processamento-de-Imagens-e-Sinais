from pathlib import Path
import cv2
import numpy as np

# Diretório principal do dataset; os caminhos abaixo partem desta pasta.
RAIZ_DATASET = Path("datasets\\oxford_pets")

# Imagens originais e trimaps (máscaras com rótulos) do dataset.
PASTA_IMAGENS = RAIZ_DATASET / "images" / "images"
PASTA_TRIMAPS = RAIZ_DATASET / "annotations" / "annotations" / "trimaps"


def listar_imagens():
    """Lista imagens JPG/JPEG válidas, em ordem alfabética."""
    # Falha explicitamente se o diretório configurado não existir.
    if not PASTA_IMAGENS.exists():
        raise FileNotFoundError(
            f"Pasta de imagens não encontrada: {PASTA_IMAGENS}"
        )

    # Ignora arquivos que não sejam imagens e arquivos auxiliares do macOS
    # (com prefixo "._"), e ordena o resultado para torná-lo previsível.
    imagens = sorted(
        caminho
        for caminho in PASTA_IMAGENS.iterdir()
        if caminho.is_file()
        and not caminho.name.startswith("._")
        and caminho.suffix.lower() in {".jpg", ".jpeg"}
    )

    return imagens


def obter_trimap(caminho_imagem):
    """Encontra a anotação original correspondente à imagem."""
    # Cada trimap tem o mesmo nome-base da imagem, mas usa a extensão PNG.
    caminho_trimap = PASTA_TRIMAPS / (
        f"{caminho_imagem.stem}.png"
    )

    # Não prossegue com um par incompleto: cada imagem precisa de anotação.
    if not caminho_trimap.is_file():
        raise FileNotFoundError(
            f"Trimap não encontrada: {caminho_trimap}"
        )

    return caminho_trimap


def carregar_par(caminho_imagem):
    """Carrega e valida uma imagem junto com sua trimap correspondente."""
    # Lê a imagem colorida no formato BGR usado pelo OpenCV.
    imagem = cv2.imread(
        str(caminho_imagem),
        cv2.IMREAD_COLOR
    )

    # O OpenCV retorna None quando não consegue decodificar o arquivo.
    if imagem is None:
        raise ValueError(
            f"Não foi possível carregar a imagem: {caminho_imagem}"
        )

    # Localiza e lê a trimap sem alterar seus valores de rótulo.
    caminho_trimap = obter_trimap(caminho_imagem)

    trimap = cv2.imread(
        str(caminho_trimap),
        cv2.IMREAD_UNCHANGED
    )

    # Confirma que a anotação foi efetivamente carregada.
    if trimap is None:
        raise ValueError(
            f"Não foi possível carregar a trimap: {caminho_trimap}"
        )

    # Uma trimap deve conter um rótulo por pixel, portanto ter apenas 2 dimensões.
    if trimap.ndim != 2:
        raise ValueError(
            "A trimap deve ser uma imagem de um canal."
        )

    # Cada pixel da imagem precisa ter um pixel correspondente na anotação.
    if imagem.shape[:2] != trimap.shape:
        raise ValueError(
            "A imagem e a trimap possuem dimensões diferentes."
        )

    # Verifica que os pixels usam somente os três rótulos previstos pelo dataset.
    valores = set(np.unique(trimap).tolist())

    if not valores.issubset({1, 2, 3}):
        raise ValueError(
            f"Valores inesperados na trimap: {valores}"
        )

    # Devolve o par já conferido para uso nas etapas seguintes do processamento.
    return imagem, trimap
