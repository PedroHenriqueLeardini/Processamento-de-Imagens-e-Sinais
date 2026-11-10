"""
experimento_lote.py — Roda os experimentos do ROADMAP em várias imagens.
 
Para cada imagem sorteada do Oxford-IIIT Pet:
  RGB, HSV, LAB e IA, cada um com e sem refinamento, avaliados contra o trimap
  e medidos na FFT (energia em alta frequência da máscara).
 
Saídas em --saida (padrão: output/lote):
  resultados_por_imagem.csv   todas as métricas, uma linha por imagem/variante
  tabela_media.md             média ± desvio-padrão por variante
  ganho_refinamento.md        o refinamento melhora a máscara? (por método)
  grafico_metricas.png        IoU, Precision, Recall e F1 por variante
  grafico_ganho.png           ganho médio de IoU do refinamento
  grafico_fft.png             alta frequência antes x depois do refinamento
  exemplo/                    máscaras, PNGs e espectros de uma imagem
 
Uso:
  python experimento_lote.py --n 20 --semente 42
  python experimento_lote.py --n 20 --sem-ia        (sem PyTorch)
"""
 
import argparse
import csv
import random
import time
from pathlib import Path
 
import numpy as np
 
import pipeline
from dataset_utils import carregar_par, listar_imagens
 
METRICAS = ("iou", "precisao", "recall", "f1_score")
NOMES_METRICAS = {
    "iou": "IoU", "precisao": "Precision", "recall": "Recall", "f1_score": "F1",
}
 
 
def _media_desvio(valores):
    return float(np.mean(valores)), float(np.std(valores))
 
 
def executar_lote(n=20, semente=42, pasta_saida="output/lote",
                  metodos=None, objeto="animal"):
    """Roda o experimento em n imagens e salva tabelas e gráficos."""
    pasta = Path(pasta_saida)
    pasta.mkdir(parents=True, exist_ok=True)
 
    todas = listar_imagens()
    if not todas:
        raise FileNotFoundError("Nenhuma imagem encontrada no dataset.")
 
    # Sorteio com semente fixa: o mesmo grupo de imagens pode ser repetido.
    amostra = random.Random(semente).sample(todas, min(n, len(todas)))
 
    linhas = []          # uma linha por imagem x variante
    ganhos = {}          # método -> lista de ganhos de IoU por imagem
    erros = []
    exemplo_salvo = False
    inicio = time.time()
 
    for indice, caminho in enumerate(amostra, start=1):
        print(f"[{indice}/{len(amostra)}] {caminho.name}")
 
        try:
            imagem, trimap = carregar_par(caminho)
            resultados = pipeline.executar_experimento(
                imagem, trimap, objeto=objeto, metodos=metodos
            )
        except Exception as erro:  # uma imagem ruim não derruba o lote
            erros.append((caminho.name, str(erro)))
            print(f"   ignorada: {erro}")
            continue
 
        for rotulo, dados in resultados.items():
            if rotulo == "ganho_refinamento":
                continue
            linha = {"imagem": caminho.name, "variante": rotulo,
                     "alta_freq": dados["alta_freq"]}
            linha.update({k: dados["metricas"][k] for k in METRICAS})
            linhas.append(linha)
 
        for metodo, g in resultados["ganho_refinamento"].items():
            ganhos.setdefault(metodo, []).append(g["iou"]["absoluto"])
 
        # Guarda os arquivos visuais da primeira imagem para os slides.
        if not exemplo_salvo:
            pipeline.salvar_experimento(imagem, resultados, pasta / "exemplo")
            exemplo_salvo = True
 
    if not linhas:
        raise RuntimeError("Nenhuma imagem foi processada com sucesso.")
 
    # ----- CSV completo --------------------------------------------------
    campos = ["imagem", "variante", *METRICAS, "alta_freq"]
    with open(pasta / "resultados_por_imagem.csv", "w",
              newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)
 
    # ----- Agregação por variante ---------------------------------------
    variantes = list(dict.fromkeys(l["variante"] for l in linhas))
    resumo = {}
    for v in variantes:
        grupo = [l for l in linhas if l["variante"] == v]
        resumo[v] = {k: _media_desvio([l[k] for l in grupo])
                     for k in (*METRICAS, "alta_freq")}
        resumo[v]["n"] = len(grupo)
 
    _salvar_tabela_media(resumo, pasta / "tabela_media.md")
    resumo_ganho = _salvar_ganho(ganhos, linhas, pasta / "ganho_refinamento.md")
    _salvar_graficos(resumo, resumo_ganho, pasta)
 
    print(f"\nConcluído em {time.time() - inicio:.1f}s — "
          f"{len(amostra) - len(erros)} imagens, {len(erros)} ignoradas.")
    print(f"Resultados em: {pasta.resolve()}")
    return {"resumo": resumo, "ganho": resumo_ganho, "erros": erros}
 
 
# ---------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------
 
def _salvar_tabela_media(resumo, caminho):
    linhas = [
        "| Método | Refinamento | N | IoU | Precision | Recall | F1 | Alta freq. (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rotulo, r in resumo.items():
        metodo = rotulo.replace(" + refinamento", "")
        refinado = "Sim" if "refinamento" in rotulo else "Não"
        celulas = [f"{r[k][0]:.3f} ± {r[k][1]:.3f}" for k in METRICAS]
        linhas.append(
            f"| {metodo} | {refinado} | {r['n']} | " + " | ".join(celulas)
            + f" | {r['alta_freq'][0] * 100:.2f} |"
        )
    caminho.write_text("\n".join(linhas), encoding="utf-8")
    print("\n" + "\n".join(linhas))
 
 
def _salvar_ganho(ganhos, linhas, caminho):
    """
    Responde a hipótese central: o refinamento melhora a máscara?
    Compara, por imagem, o IoU refinado contra o bruto.
    """
    saida = {}
    texto = [
        "| Método | Ganho médio de IoU | Melhorou | Piorou | Igual |",
        "|---|---:|---:|---:|---:|",
    ]
    for metodo, valores in ganhos.items():
        v = np.array(valores)
        melhorou = int(np.sum(v > 1e-9))
        piorou = int(np.sum(v < -1e-9))
        igual = len(v) - melhorou - piorou
        saida[metodo] = {"medio": float(v.mean()), "melhorou": melhorou,
                         "piorou": piorou, "igual": igual, "n": len(v)}
        texto.append(
            f"| {metodo} | {v.mean():+.4f} | {melhorou}/{len(v)} "
            f"| {piorou}/{len(v)} | {igual}/{len(v)} |"
        )
    caminho.write_text("\n".join(texto), encoding="utf-8")
    print("\n" + "\n".join(texto))
    return saida
 
 
# ---------------------------------------------------------------------
# Gráficos (Figure direta: salva em arquivo sem abrir janela)
# ---------------------------------------------------------------------
 
def _salvar_graficos(resumo, resumo_ganho, pasta):
    from matplotlib.figure import Figure
 
    rotulos = list(resumo.keys())
    posicoes = np.arange(len(rotulos))
 
    # 1) Métricas por variante, com barra de erro (desvio-padrão)
    fig = Figure(figsize=(16, 9))
    eixos = fig.subplots(2, 2).flatten()
    for eixo, chave in zip(eixos, METRICAS):
        medias = [resumo[r][chave][0] for r in rotulos]
        desvios = [resumo[r][chave][1] for r in rotulos]
        eixo.bar(posicoes, medias, yerr=desvios, capsize=3)
        eixo.set_title(NOMES_METRICAS[chave])
        eixo.set_ylim(0, 1.05)
        eixo.set_xticks(posicoes)
        eixo.set_xticklabels(rotulos, rotation=45, ha="right", fontsize=8)
        eixo.grid(axis="y", linestyle="--", alpha=0.3)
    fig.suptitle("Métricas médias por variante (barra = desvio-padrão)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(pasta / "grafico_metricas.png"), dpi=120)
 
    # 2) Ganho médio de IoU do refinamento, por método
    fig = Figure(figsize=(8, 5))
    eixo = fig.subplots()
    nomes = list(resumo_ganho.keys())
    valores = [resumo_ganho[m]["medio"] for m in nomes]
    barras = eixo.bar(nomes, valores)
    eixo.axhline(0, color="black", linewidth=0.8)
    eixo.bar_label(barras, fmt="%+.4f", padding=2)
    eixo.set_ylabel("Δ IoU (refinada − bruta)")
    eixo.set_title("O refinamento melhora a máscara?")
    eixo.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(pasta / "grafico_ganho.png"), dpi=120)
 
    # 3) FFT: energia em alta frequência antes x depois do refinamento
    fig = Figure(figsize=(8, 5))
    eixo = fig.subplots()
    metodos = [r for r in rotulos if "refinamento" not in r]
    x = np.arange(len(metodos))
    antes = [resumo[m]["alta_freq"][0] * 100 for m in metodos]
    depois = [resumo[f"{m} + refinamento"]["alta_freq"][0] * 100 for m in metodos]
    eixo.bar(x - 0.2, antes, 0.4, label="Bruta")
    eixo.bar(x + 0.2, depois, 0.4, label="Refinada")
    eixo.set_xticks(x)
    eixo.set_xticklabels(metodos)
    eixo.set_ylabel("Energia acima de 0,1 ciclos/pixel (%)")
    eixo.set_title("Alta frequência na máscara (FFT)")
    eixo.legend()
    eixo.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(pasta / "grafico_fft.png"), dpi=120)
 
 
# ---------------------------------------------------------------------
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--n", type=int, default=20, help="número de imagens")
    parser.add_argument("--semente", type=int, default=42, help="semente do sorteio")
    parser.add_argument("--saida", default="output/lote", help="pasta de saída")
    parser.add_argument("--sem-ia", action="store_true",
                        help="não roda o modelo de IA (não precisa de PyTorch)")
    args = parser.parse_args()
 
    executar_lote(
        n=args.n,
        semente=args.semente,
        pasta_saida=args.saida,
        metodos=["rgb", "hsv", "lab"] if args.sem_ia else None,
    )
 