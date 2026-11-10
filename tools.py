"""
tools.py — Funções que a IA generativa pode chamar (tool calling).
 
Mesmo padrão do exemplo da disciplina: o modelo de linguagem lê o pedido em
linguagem natural e ESCOLHE uma função daqui; o código Python é quem executa.
 
  ai.py        declara as tools (nome + argumentos) que o modelo enxerga
  tools.py     contém as funções reais que rodam   <-- este arquivo
  pipeline.py  decide a ordem: interpreta -> valida -> executar_tool()
 
Escopo limitado (guardrail): o modelo só consegue acionar o que está em
REGISTRO, e só com os argumentos que cada função declara. Qualquer outro
nome ou argumento inventado pelo modelo é recusado/ignorado.
 
Toda tool devolve um dicionário com "status":
  "ok"                -> executou (há "mensagem" e os dados);
  "falha_segmentacao" -> a máscara foi rejeitada pela checagem.
"""
 
import inspect
from pathlib import Path
 
import ai
import pipeline
from image_processing import salvar_imagem
 
 
# ---------------------------------------------------------------------
# Tool 1 — remover_fundo
# ---------------------------------------------------------------------
 
def remover_fundo(imagem, objeto=None, metodo="ia", refinar=True,
                  pasta_saida="output/app", usar_llm=True):
    """
    Remove o fundo mantendo o objeto e salva um PNG com transparência.
 
    imagem:  imagem BGR do OpenCV (entregue pelo sistema, não pelo modelo).
    objeto:  o que manter ("cachorro", "gato"...); obrigatório para metodo="ia".
    metodo:  "ia", "rgb", "hsv" ou "lab".
    refinar: aplica o refinamento morfológico à máscara.
    """
    try:
        saida = pipeline.remover_fundo(
            imagem, metodo=metodo, objeto=objeto, refinar=refinar
        )
    except pipeline.SegmentacaoInvalida as erro:
        return {
            "status": "falha_segmentacao",
            "mensagem": (
                f"Não consegui identificar o objeto com segurança: {erro} "
                "Tente outra imagem ou um pedido mais específico."
            ),
        }
 
    # Saída: PNG com canal alfa
    caminho = Path(pasta_saida) / "resultado.png"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    salvar_imagem(caminho, saida["imagem_transparente"])
 
    return {
        "status": "ok",
        "caminho_saida": str(caminho),
        "mensagem": ai.explicar_resultado(
            "remover_fundo", saida, usar_llm=usar_llm
        ),
        **saida,
    }
 
 
# ---------------------------------------------------------------------
# Tool 2 — comparar_metodos
# ---------------------------------------------------------------------
 
def comparar_metodos(imagem, objeto=None, trimap=None,
                     pasta_saida="output/app"):
    """
    Gera as máscaras de RGB, HSV, LAB e IA, com e sem refinamento.
 
    Com trimap, também calcula IoU, Precision, Recall e F1 de cada uma
    e a tabela comparativa. Salva máscaras, PNGs e espectros (FFT).
    """
    resultados = pipeline.executar_experimento(
        imagem, trimap, objeto=objeto or "animal"
    )
    pipeline.salvar_experimento(
        imagem, resultados, Path(pasta_saida) / "comparacao"
    )
 
    return {
        "status": "ok",
        "resultados": resultados,
        "tabela": (
            pipeline.tabela_resultados(resultados) if trimap is not None else None
        ),
        "mensagem": (
            "Comparação gerada para RGB, HSV, LAB e IA."
            if trimap is not None
            else "Comparação gerada (sem métricas, pois não há trimap)."
        ),
    }
 
 
# ---------------------------------------------------------------------
# Registro e despacho
# ---------------------------------------------------------------------
 
# Nomes aqui devem ser os mesmos declarados em ai.TOOLS.
REGISTRO = {
    "remover_fundo": remover_fundo,
    "comparar_metodos": comparar_metodos,
}
 
 
def executar_tool(nome, **argumentos):
    """
    Executa a tool escolhida pelo modelo.
 
    - Nome fora do REGISTRO: ValueError (o modelo não ganha novos poderes).
    - Argumentos que a função não declara são descartados.
    """
    if nome not in REGISTRO:
        raise ValueError(f"Tool não permitida: {nome}")
 
    funcao = REGISTRO[nome]
    aceitos = inspect.signature(funcao).parameters
    filtrados = {k: v for k, v in argumentos.items() if k in aceitos}
 
    return funcao(**filtrados)
 