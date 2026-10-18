
import numpy as np


def avaliar_mascara(mascara, trimap):
    if not isinstance(mascara, np.ndarray) or mascara.size == 0:
        raise ValueError("A máscara deve ser um array NumPy não vazio.")

    if not isinstance(trimap, np.ndarray) or trimap.size == 0:
        raise ValueError("O trimap deve ser um array NumPy não vazio.")

    if mascara.ndim != 2 or trimap.ndim != 2:
        raise ValueError("Máscara e trimap devem ser bidimensionais.")

    if mascara.shape != trimap.shape:
        raise ValueError(
            "Máscara e trimap devem ter as mesmas dimensões."
        )

    valores = set(np.unique(trimap).tolist())

    if not valores.issubset({1, 2, 3}):
        raise ValueError(
            "O trimap deve conter somente os valores 1, 2 e 3."
        )

    # Máscara gerada: pixels brancos representam o animal.
    previsao = mascara > 0

    # Trimap: 1 = animal, 2 = fundo, 3 = região incerta.
    referencia = trimap == 1
    regiao_incerta = trimap == 3

    # --------------------------------------------------
    # 1. MÉTRICAS TRADICIONAIS
    # Considera apenas as classes 1 e 2.
    # --------------------------------------------------
    pixels_validos = trimap != 3

    previsao_valida = previsao[pixels_validos]
    referencia_valida = referencia[pixels_validos]

    tp = int(np.sum(previsao_valida & referencia_valida))
    fp = int(np.sum(previsao_valida & ~referencia_valida))
    fn = int(np.sum(~previsao_valida & referencia_valida))
    tn = int(np.sum(~previsao_valida & ~referencia_valida))

    def dividir_seguro(numerador, denominador):
        if denominador == 0:
            return 0.0
        return numerador / denominador

    iou = dividir_seguro(tp, tp + fp + fn)
    precisao = dividir_seguro(tp, tp + fp)
    recall = dividir_seguro(tp, tp + fn)
    f1 = dividir_seguro(2 * tp, 2 * tp + fp + fn)
    acuracia = dividir_seguro(
        tp + tn, tp + tn + fp + fn
    )

    # --------------------------------------------------
    # 2. ANÁLISE DA REGIÃO DE INCERTEZA
    # --------------------------------------------------
    total_incerto = int(np.sum(regiao_incerta))
    total_previsto_animal = int(np.sum(previsao))

    # Quantos pixels previstos como animal estão na
    # região incerta, em relação a todos os pixels
    # previstos como animal.
    pixels_previstos_na_incerteza = int(
        np.sum(previsao & regiao_incerta)
    )

    proporcao_mascara_na_incerteza = dividir_seguro(
        pixels_previstos_na_incerteza,
        total_previsto_animal
    )

    # Qual porcentagem da região incerta foi marcada
    # como animal pela máscara.
    proporcao_incerteza_como_animal = dividir_seguro(
        pixels_previstos_na_incerteza,
        total_incerto
    )

    # Qual porcentagem da região incerta foi marcada
    # como fundo pela máscara.
    proporcao_incerteza_como_fundo = dividir_seguro(
        total_incerto - pixels_previstos_na_incerteza,
        total_incerto
    )

    return {
        # Métricas tradicionais
        "iou": iou,
        "precisao": precisao,
        "recall": recall,
        "f1_score": f1,
        "acuracia": acuracia,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,

        # Diagnóstico da região incerta
        "pixels_regiao_incerta": total_incerto,
        "pixels_previstos_na_incerteza":
            pixels_previstos_na_incerteza,
        "proporcao_mascara_na_incerteza":
            proporcao_mascara_na_incerteza,
        "proporcao_incerteza_como_animal":
            proporcao_incerteza_como_animal,
        "proporcao_incerteza_como_fundo":
            proporcao_incerteza_como_fundo,
    }

