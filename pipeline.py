"""
pipeline.py — Orquestração do projeto.
 
Este módulo NÃO implementa algoritmos de imagem: ele só chama, na ordem
certa, as funções de image_processing.py, segmentation.py, evaluation.py
e ai.py.
 
Dois fluxos:
  A) Aplicação:    texto do usuário -> IA generativa (escolhe a tool) ->
                   validação -> tools.py -> segmentação -> checagem ->
                   refinamento -> alpha -> PNG
  B) Experimento:  mesma imagem em RGB/HSV/LAB/IA, com e sem refinamento,
                   avaliados contra o trimap -> tabela e ganho do refinamento
"""
 
from pathlib import Path
 
import numpy as np
 
import ai
import fft_mascara
from evaluation import avaliar_mascara
from image_processing import criar_imagem_transparente, salvar_imagem
from segmentation import (
    criar_mascara_hsv,
    criar_mascara_lab,
    criar_mascara_rgb,
    refinar_mascara,
)
 
# ---------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------
 
# Limites de exemplo (os mesmos do main.py). Segmentação por cor depende
# da imagem: ajuste-os para cada caso.
PARAMETROS_PADRAO = {
    "rgb": {"inferior": (80, 30, 20),  "superior": (255, 180, 160)},
    "hsv": {"inferior": (0, 30, 30),   "superior": (25, 255, 255)},
    "lab": {"inferior": (0, 120, 120), "superior": (255, 180, 180)},
}
 
METODOS_COR = {
    "rgb": criar_mascara_rgb,
    "hsv": criar_mascara_hsv,
    "lab": criar_mascara_lab,
}
 
# Faixa de área aceitável para o objeto (README, seção 4.1).
AREA_MINIMA = 0.01   # menos de 1% da imagem: provavelmente não achou nada
AREA_MAXIMA = 0.95   # mais de 95%: provavelmente pegou a imagem inteira
 
# Ordem fixa dos 8 experimentos do ROADMAP (seção 11).
EXPERIMENTOS = [
    ("RGB", "rgb"), ("HSV", "hsv"), ("LAB", "lab"), ("IA", "ia"),
]
 
 
class SegmentacaoInvalida(Exception):
    """A máscara gerada não é confiável; o processamento é interrompido."""
 
 
# ---------------------------------------------------------------------
# Peças básicas
# ---------------------------------------------------------------------
 
def gerar_mascara(imagem, metodo, objeto=None, parametros=None):
    """Máscara BRUTA (sem refinamento) pelo método pedido."""
    if metodo in METODOS_COR:
        limites = (parametros or PARAMETROS_PADRAO)[metodo]
        return METODOS_COR[metodo](
            imagem, limites["inferior"], limites["superior"]
        )
 
    if metodo == "ia":
        return ai.segmentar_ia(imagem, objeto)
 
    raise ValueError(f"Método desconhecido: {metodo}")
 
 
def proporcao_objeto(mascara):
    """Fração da imagem marcada como objeto (0 a 1)."""
    return float(np.count_nonzero(mascara)) / mascara.size
 
 
def verificar_mascara(mascara):
    """
    Decide se a máscara merece seguir no pipeline.
    A IA pode errar; esta etapa separa 'a IA previu' de 'o sistema aceita'.
    Retorna (ok, motivo).
    """
    area = proporcao_objeto(mascara)
 
    if area == 0:
        return False, "Nenhum objeto foi identificado."
    if area < AREA_MINIMA:
        return False, (
            f"O objeto encontrado é pequeno demais ({area * 100:.2f}% da imagem)."
        )
    if area > AREA_MAXIMA:
        return False, (
            f"A região ocupa quase toda a imagem ({area * 100:.1f}%)."
        )
    return True, ""
 
 
# ---------------------------------------------------------------------
# Fluxo A — remoção de fundo (usado pela aplicação)
# ---------------------------------------------------------------------
 
def remover_fundo(imagem, metodo="ia", objeto=None, refinar=True,
                  parametros=None):
    """
    imagem -> máscara -> checagem -> (refinamento) -> alpha -> BGRA.
 
    Levanta SegmentacaoInvalida se a máscara não for confiável.
    """
    mascara_bruta = gerar_mascara(imagem, metodo, objeto, parametros)
 
    ok, motivo = verificar_mascara(mascara_bruta)
    if not ok:
        raise SegmentacaoInvalida(motivo)
 
    mascara_final = refinar_mascara(mascara_bruta) if refinar else mascara_bruta
 
    # O refinamento (abertura) pode apagar objetos muito finos: confere de novo.
    ok, motivo = verificar_mascara(mascara_final)
    if not ok:
        raise SegmentacaoInvalida(f"Após o refinamento: {motivo}")
 
    return {
        "mascara_bruta": mascara_bruta,
        "mascara_final": mascara_final,
        "imagem_transparente": criar_imagem_transparente(imagem, mascara_final),
        "proporcao_objeto": proporcao_objeto(mascara_final),
        # FFT: quanto da energia da máscara está em alta frequência (ruído/bordas)
        "alta_freq_bruta": fft_mascara.razao_alta_frequencia(mascara_bruta),
        "alta_freq_final": fft_mascara.razao_alta_frequencia(mascara_final),
    }
 
 
# ---------------------------------------------------------------------
# Fluxo B — experimento comparativo
# ---------------------------------------------------------------------
 
def _ganho(antes, depois):
    """Ganho absoluto e percentual do refinamento sobre uma métrica."""
    absoluto = depois - antes
    percentual = (absoluto / antes * 100) if antes > 0 else None
    return {"absoluto": absoluto, "percentual": percentual}
 
 
def executar_experimento(imagem, trimap=None, objeto="animal",
                         parametros=None, metodos=None):
    """
    Gera as máscaras do ROADMAP (cada método com e sem refinamento) para a
    MESMA imagem e, se houver trimap, avalia todas com o mesmo protocolo.
 
    metodos: lista como ["rgb", "hsv", "ia"]; None = todos (RGB, HSV, LAB, IA).
 
    Retorna  { rótulo: {"mascara", "metricas", "alta_freq"} }
    e, em "ganho_refinamento", quanto o refinamento mudou cada método.
    """
    escolhidos = [
        (nome, metodo) for nome, metodo in EXPERIMENTOS
        if metodos is None or metodo in metodos
    ]
    resultados = {}
 
    for nome, metodo in escolhidos:
        bruta = gerar_mascara(imagem, metodo, objeto, parametros)
        variantes = {
            nome: bruta,
            f"{nome} + refinamento": refinar_mascara(bruta),
        }
 
        for rotulo, mascara in variantes.items():
            resultados[rotulo] = {
                "mascara": mascara,
                "metricas": (
                    avaliar_mascara(mascara, trimap) if trimap is not None else None
                ),
                # Energia da máscara em alta frequência (ver fft_mascara.py)
                "alta_freq": fft_mascara.razao_alta_frequencia(mascara),
            }
 
    if trimap is not None:
        ganho = {}
        for nome, _ in escolhidos:
            antes = resultados[nome]["metricas"]
            depois = resultados[f"{nome} + refinamento"]["metricas"]
            ganho[nome] = {
                chave: _ganho(antes[chave], depois[chave])
                for chave in ("iou", "precisao", "recall", "f1_score")
            }
        resultados["ganho_refinamento"] = ganho
 
    return resultados
 
 
def tabela_resultados(resultados):
    """Tabela em Markdown no formato do ROADMAP (seção 12), com a coluna de FFT."""
    linhas = [
        "| Método | Refinamento | IoU | Precision | Recall | F1 | Alta freq. (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
 
    for rotulo, dados in resultados.items():
        if rotulo == "ganho_refinamento" or dados["metricas"] is None:
            continue
 
        m = dados["metricas"]
        metodo = rotulo.replace(" + refinamento", "")
        refinado = "Sim" if "refinamento" in rotulo else "Não"
        linhas.append(
            f"| {metodo} | {refinado} | {m['iou']:.4f} | {m['precisao']:.4f} "
            f"| {m['recall']:.4f} | {m['f1_score']:.4f} "
            f"| {dados['alta_freq'] * 100:.2f} |"
        )
 
    return "\n".join(linhas)
 
 
def salvar_experimento(imagem, resultados, pasta="output/comparisons"):
    """Salva máscara, PNG transparente, espectros (FFT) e tabela."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
 
    for rotulo, dados in resultados.items():
        if rotulo == "ganho_refinamento":
            continue
 
        nome = rotulo.lower().replace(" + ", "_").replace(" ", "_")
        mascara = dados["mascara"]
 
        salvar_imagem(pasta / f"mascara_{nome}.png", mascara)
        salvar_imagem(
            pasta / f"resultado_{nome}.png",
            criar_imagem_transparente(imagem, mascara),
        )
 
        # Para cada método, compara o espectro antes e depois do refinamento.
        if "refinamento" not in rotulo:
            refinada = resultados[f"{rotulo} + refinamento"]["mascara"]
            fft_mascara.salvar_comparacao_espectros(
                mascara, refinada, pasta / f"fft_{nome}.png", nome=rotulo
            )
 
    if any(d["metricas"] is not None
           for k, d in resultados.items() if k != "ganho_refinamento"):
        (pasta / "tabela.md").write_text(
            tabela_resultados(resultados), encoding="utf-8"
        )
 
 
# ---------------------------------------------------------------------
# Ponto de entrada da aplicação: texto livre -> resultado
# ---------------------------------------------------------------------
 
def executar_pedido(imagem, texto, pasta_saida="output/app",
                    trimap=None, usar_llm=True):
    """
    Fluxo completo do usuário.
 
    Retorna sempre um dicionário com "status":
      "ok"                      -> processado (há "mensagem" e os dados);
      "precisa_esclarecimento"  -> faltou/errou algo no pedido;
      "falha_segmentacao"       -> a máscara foi rejeitada.
    """
    # 1. IA generativa: texto -> JSON
    pedido = ai.interpretar_comando(texto, usar_llm=usar_llm)
 
    # 2. Guardrail: só executa ações conhecidas e completas
    valido, acao, mensagem = ai.validar_acao(pedido)
    if not valido:
        return {"status": "precisa_esclarecimento", "mensagem": mensagem}
 
    # 3. Execução da tool escolhida pelo modelo (funções em tools.py).
    # Import tardio: tools.py importa este módulo.
    import tools
 
    resultado = tools.executar_tool(
        acao["acao"],
        imagem=imagem,
        objeto=acao["objeto"],
        metodo=acao["metodo"],
        refinar=acao["refinar"],
        trimap=trimap,
        pasta_saida=pasta_saida,
        usar_llm=usar_llm,
    )
 
    return {"acao": acao, **resultado}
 
 
# ---------------------------------------------------------------------
# Uso pela linha de comando:
#   python pipeline.py foto.jpg "remova o fundo e mantenha o cachorro"
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys
 
    from image_processing import carregar_imagem
 
    if len(sys.argv) < 3:
        sys.exit('Uso: python pipeline.py <imagem> "<pedido>"')
 
    resultado = executar_pedido(carregar_imagem(sys.argv[1]), sys.argv[2])
    print(resultado["status"], "-", resultado["mensagem"])
    if resultado["status"] == "ok" and "caminho_saida" in resultado:
        print("Salvo em:", resultado["caminho_saida"])
 