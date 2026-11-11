"""
ai.py — As duas camadas de IA do projeto.
 
1) IA PERCEPTIVA  -> segmentar_ia(): modelo pré-treinado que responde
                     "onde está o objeto?" e devolve uma máscara.
2) IA GENERATIVA  -> interpretar_comando(): modelo de linguagem que responde
                     "o que o usuário quer?" e devolve um JSON.
 
Regra do projeto: nenhuma das duas IAs altera pixels da imagem.
A IA perceptiva só produz a máscara; a generativa só produz uma decisão.
"""
 
import json
import os
import re
import unicodedata
 
import cv2
import numpy as np
 
# Carrega GROQ_API_KEY de um arquivo .env, se existir (pip install python-dotenv).
# Sem a biblioteca, a chave continua podendo vir das variáveis de ambiente.
try:
    from dotenv import load_dotenv
 
    load_dotenv()
except ImportError:
    pass
 
 
# =====================================================================
# 1. IA PERCEPTIVA — segmentação semântica pré-treinada (somente inferência)
# =====================================================================
 
# O DeepLabV3 do torchvision foi treinado com as 21 classes do Pascal VOC.
# O índice 0 é o fundo; "cat" e "dog" existem, o que combina com o
# dataset Oxford-IIIT Pet.
CATEGORIAS_VOC = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
)
 
# Tradução dos nomes que o usuário digita para as classes do modelo.
# Uma palavra pode corresponder a mais de uma classe (ex.: "animal").
OBJETOS_PT = {
    "cachorro": ["dog"], "cao": ["dog"], "cachorra": ["dog"],
    "gato": ["cat"], "gata": ["cat"],
    "animal": ["cat", "dog"], "pet": ["cat", "dog"],
    "pessoa": ["person"], "gente": ["person"], "homem": ["person"],
    "mulher": ["person"], "crianca": ["person"],
    "carro": ["car"], "onibus": ["bus"], "trem": ["train"],
    "bicicleta": ["bicycle"], "moto": ["motorbike"],
    "cavalo": ["horse"], "vaca": ["cow"], "ovelha": ["sheep"],
    "passaro": ["bird"], "ave": ["bird"], "barco": ["boat"],
    "aviao": ["aeroplane"], "garrafa": ["bottle"],
    "cadeira": ["chair"], "sofa": ["sofa"], "planta": ["pottedplant"],
    "mesa": ["diningtable"], "tv": ["tvmonitor"],
}
 
# O modelo é pesado: é carregado uma única vez e reaproveitado.
_MODELO = None
 
 
def _normalizar_texto(texto):
    """Minúsculas e sem acentos, para comparar palavras de forma robusta."""
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(c for c in texto if not unicodedata.combining(c)).strip()
 
 
def classes_do_objeto(objeto):
    """
    Converte o nome do objeto em índices de classe do modelo.
    objeto=None significa 'qualquer objeto de primeiro plano'.
    Levanta ValueError se o objeto não for reconhecido.
    """
    if objeto is None:
        return list(range(1, len(CATEGORIAS_VOC)))
 
    nome = _normalizar_texto(objeto)
    nomes_voc = OBJETOS_PT.get(nome)
 
    if nomes_voc is None and nome in CATEGORIAS_VOC and nome != "background":
        nomes_voc = [nome]  # o usuário já escreveu o nome em inglês
 
    if nomes_voc is None:
        raise ValueError(f"Objeto não suportado pelo modelo: {objeto}")
 
    return [CATEGORIAS_VOC.index(n) for n in nomes_voc]
 
 
def carregar_modelo():
    """Carrega (uma vez) o modelo pré-treinado e seu pré-processamento."""
    global _MODELO
 
    if _MODELO is None:
        # Import tardio: o resto do projeto funciona mesmo sem o PyTorch.
        from torchvision.models.segmentation import (
            DeepLabV3_ResNet50_Weights,
            deeplabv3_resnet50,
        )
 
        pesos = DeepLabV3_ResNet50_Weights.DEFAULT
        modelo = deeplabv3_resnet50(weights=pesos)
        modelo.eval()  # modo de inferência: sem treino, sem dropout
 
        # transforms() redimensiona e normaliza como no treinamento.
        _MODELO = (modelo, pesos.transforms())
 
    return _MODELO
 
 
def segmentar_ia(imagem_bgr, objeto=None, limiar=0.5):
    """
    Gera a máscara bruta do objeto usando o modelo pré-treinado.
 
    imagem_bgr: imagem OpenCV (altura x largura x 3, uint8).
    objeto:     nome do objeto ("cachorro", "gato"...) ou None.
    limiar:     probabilidade mínima para o pixel contar como objeto.
 
    Retorna uma máscara uint8 do mesmo tamanho da imagem
    (255 = objeto, 0 = fundo), SEM nenhum refinamento.
    """
    if (
        not isinstance(imagem_bgr, np.ndarray)
        or imagem_bgr.ndim != 3
        or imagem_bgr.shape[2] != 3
    ):
        raise ValueError("A imagem deve ser um array BGR com três canais.")
 
    import torch
    import torch.nn.functional as F
 
    indices = classes_do_objeto(objeto)
    modelo, transformacao = carregar_modelo()
 
    # O modelo foi treinado em RGB; o OpenCV entrega BGR.
    rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # HxWx3 -> 3xHxW
 
    with torch.no_grad():
        entrada = transformacao(tensor).unsqueeze(0)
        logits = modelo(entrada)["out"]  # 1 x 21 x h x w (tamanho reduzido)
 
        # Devolve os logits ao tamanho original para que a máscara
        # tenha as mesmas dimensões da imagem (e do trimap).
        logits = F.interpolate(
            logits,
            size=imagem_bgr.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
 
        # Probabilidade de cada classe por pixel; somamos as classes pedidas
        # (ex.: "animal" = gato + cachorro) e binarizamos pelo limiar.
        probabilidades = logits.softmax(dim=1)[0]
        prob_objeto = probabilidades[indices].sum(dim=0)
        mascara = (prob_objeto > limiar).cpu().numpy()
 
    return mascara.astype(np.uint8) * 255
 
 
# =====================================================================
# 2. IA GENERATIVA — linguagem natural -> ação estruturada (JSON)
# =====================================================================
 
# Modelos da Groq a tentar, em ordem. O primeiro que existir na SUA conta é
# usado. Para forçar um específico: variável de ambiente MODELO_LINGUAGEM.
# A chave é lida de GROQ_API_KEY.
MODELOS_CANDIDATOS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]
if os.environ.get("MODELO_LINGUAGEM"):
    MODELOS_CANDIDATOS.insert(0, os.environ["MODELO_LINGUAGEM"])
 
_MODELO_FUNCIONAL = None  # lembra qual modelo respondeu, para não retestar
 
 
def _modelo_indisponivel(erro):
    """True se o erro for 'esse modelo não existe/não está liberado'."""
    texto = str(erro).lower()
    return any(
        t in texto
        for t in ("model_not_found", "does not exist", "decommissioned",
                  "no access", "not supported")
    )
 
 
def _chamar_groq(**argumentos):
    """
    chat.completions.create com troca automática de modelo: se um nome
    não existir na conta, tenta o próximo da lista.
    """
    global _MODELO_FUNCIONAL
    from groq import Groq  # lê a chave em GROQ_API_KEY
 
    cliente = Groq()
    candidatos = [_MODELO_FUNCIONAL] if _MODELO_FUNCIONAL else MODELOS_CANDIDATOS
    ultimo_erro = None
 
    for modelo in candidatos:
        extras = dict(argumentos)
        if modelo.startswith("openai/gpt-oss"):
            # extra_body funciona em qualquer versão do SDK da Groq
            extras["extra_body"] = {"reasoning_effort": "low"}
 
        try:
            resposta = cliente.chat.completions.create(model=modelo, **extras)
            _MODELO_FUNCIONAL = modelo
            return resposta
        except Exception as erro:
            if not _modelo_indisponivel(erro):
                raise  # chave inválida, limite de cota, rede...: não adianta trocar
            ultimo_erro = erro
 
    raise RuntimeError(f"Nenhum modelo da lista está disponível: {ultimo_erro}")
 
 
ACOES_VALIDAS = {"remover_fundo", "comparar_metodos"}
METODOS_VALIDOS = {"rgb", "hsv", "lab", "ia"}
 
PROMPT_SISTEMA = (
    "Você é um assistente que decide qual função usar em um removedor de "
    "fundo de imagens. Você NUNCA descreve nem altera pixels: apenas escolhe "
    "a função e preenche os argumentos. Use remover_fundo para "
    "remover/recortar/isolar um objeto; use comparar_metodos quando o "
    "usuário pedir para comparar técnicas. O argumento objeto é o que deve "
    "ser mantido (ex.: cachorro, gato, pessoa); use string vazia se o "
    "usuário não disse. O método padrão é 'ia'; use 'rgb', 'hsv' ou 'lab' "
    "somente se o usuário pedir esse espaço de cor. refinar é true, a menos "
    "que o usuário peça sem refinamento. Se o pedido não for sobre processar "
    "imagem, não chame nenhuma função."
)
 
# Tools no formato de function calling (compatível com a Groq), como no
# exemplo da disciplina. O modelo escolhe qual delas chamar (tool_choice=auto).
# As funções que realmente executam estão em tools.py (mesmos nomes).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "remover_fundo",
            "description": "Remove o fundo da imagem mantendo um objeto.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objeto": {
                        "type": "string",
                        "description": "Objeto a manter (cachorro, gato, pessoa...); vazio se não foi dito.",
                    },
                    "metodo": {
                        "type": "string",
                        "enum": ["ia", "rgb", "hsv", "lab"],
                        "description": "Técnica de segmentação; 'ia' por padrão.",
                    },
                    "refinar": {
                        "type": "boolean",
                        "description": "Aplicar refinamento morfológico; true por padrão.",
                    },
                },
                "required": ["objeto", "metodo", "refinar"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "comparar_metodos",
            "description": "Compara RGB, HSV, LAB e IA, com e sem refinamento.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objeto": {
                        "type": "string",
                        "description": "Objeto de interesse; vazio se não foi dito.",
                    },
                },
                "required": ["objeto"],
            },
        },
    },
]
 
 
def _interpretar_com_llm(texto):
    """O modelo (Groq) escolhe a função e os argumentos; nós só lemos."""
    resposta = _chamar_groq(
        temperature=0,  # interpretação deve ser determinística
        max_tokens=1024,  # folga: modelos de raciocínio gastam tokens pensando
        messages=[
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": texto},
        ],
        tools=TOOLS,
        tool_choice="auto",
    )
 
    mensagem = resposta.choices[0].message
 
    # Sem tool call: o modelo entendeu que não é um pedido de imagem.
    if not mensagem.tool_calls:
        return {"acao": "desconhecida", "objeto": None,
                "metodo": "ia", "refinar": True}
 
    chamada = mensagem.tool_calls[0]
    # Os argumentos chegam como uma string JSON.
    argumentos = json.loads(chamada.function.arguments or "{}")
 
    # Normaliza "sem objeto": string vazia ou "null" viram None.
    objeto = str(argumentos.get("objeto") or "").strip()
    objeto = None if objeto.lower() in ("", "null", "none") else objeto
 
    return {
        "acao": chamada.function.name,  # nome da tool escolhida pelo modelo
        "objeto": objeto,
        "metodo": argumentos.get("metodo", "ia"),
        "refinar": bool(argumentos.get("refinar", True)),
    }
 
 
def _interpretar_por_regras(texto):
    """
    Alternativa sem internet/chave de API, baseada em palavras-chave.
    Útil para testes e como plano B na apresentação.
    """
    t = _normalizar_texto(texto)
    palavras = set(re.findall(r"[a-z]+", t))
 
    if "compar" in t:
        acao = "comparar_metodos"
    elif any(p in t for p in ("remov", "tire", "tirar", "recort",
                              "apag", "isol", "fundo", "transparent")):
        acao = "remover_fundo"
    else:
        acao = "desconhecida"
 
    objeto = next((p for p in OBJETOS_PT if p in palavras), None)
 
    metodo = "ia"
    for nome in ("hsv", "lab", "rgb"):
        if nome in palavras:
            metodo = nome
 
    refinar = not ("sem refin" in t or "sem morfolog" in t)
 
    return {"acao": acao, "objeto": objeto, "metodo": metodo, "refinar": refinar}
 
 
def interpretar_comando(texto, usar_llm=True):
    """
    Traduz uma frase do usuário em um dicionário de ação.
 
    Exemplo:
        "Remova o fundo e mantenha só o cachorro" ->
        {"acao": "remover_fundo", "objeto": "cachorro",
         "metodo": "ia", "refinar": True}
 
    Tenta o modelo de linguagem; se não houver chave/internet,
    cai para o interpretador por regras.
    """
    if not isinstance(texto, str) or not texto.strip():
        raise ValueError("O comando deve ser um texto não vazio.")
 
    if usar_llm and os.environ.get("GROQ_API_KEY"):
        try:
            return _interpretar_com_llm(texto)
        except Exception as erro:  # rede, cota, SDK ausente...
            print(f"[ai] LLM indisponível ({erro}); usando regras.")
 
    return _interpretar_por_regras(texto)
 
 
def validar_acao(dados):
    """
    Guardrail: confere a saída da IA generativa ANTES de executar algo.
 
    Retorna (ok, acao_limpa, mensagem):
      ok=True  -> acao_limpa pode ser executada;
      ok=False -> mensagem é uma pergunta/erro para mostrar ao usuário
                  (o sistema pergunta em vez de adivinhar).
    """
    if not isinstance(dados, dict):
        return False, None, "Não consegui interpretar o pedido."
 
    acao = dados.get("acao")
    if acao not in ACOES_VALIDAS:
        return False, None, (
            "Não entendi a ação. Peça, por exemplo: "
            "'remova o fundo e mantenha o cachorro'."
        )
 
    metodo = dados.get("metodo", "ia")
    if metodo not in METODOS_VALIDOS:
        return False, None, f"Método desconhecido: {metodo}."
 
    objeto = dados.get("objeto")
 
    # Métodos de cor não precisam do objeto; a IA de segmentação precisa.
    if acao == "remover_fundo" and metodo == "ia":
        if not objeto:
            return False, None, "Qual objeto devo manter na imagem?"
        try:
            classes_do_objeto(objeto)
        except ValueError:
            return False, None, (
                f"Ainda não sei reconhecer '{objeto}'. "
                "Tente: cachorro, gato, pessoa, carro..."
            )
 
    limpa = {
        "acao": acao,
        "objeto": objeto,
        "metodo": metodo,
        "refinar": bool(dados.get("refinar", True)),
    }
    return True, limpa, ""
 
 
def explicar_resultado(acao, resultado, usar_llm=True):
    """
    Explica o resultado em linguagem natural. Recebe só números e textos
    (nunca a imagem), então a IA generativa continua sem tocar em pixels.
    """
    resumo = json.dumps(resultado, ensure_ascii=False, default=str)
 
    if usar_llm and os.environ.get("GROQ_API_KEY"):
        try:
            resposta = _chamar_groq(
                temperature=0.2,
                max_tokens=200,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Explique em português, em até 3 frases curtas, "
                            "o resultado de um processamento de imagem. Use "
                            "somente os dados recebidos; não invente números."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Ação: {acao}\nResultado: {resumo}",
                    },
                ],
            )
            return resposta.choices[0].message.content
        except Exception:
            pass  # cai para a explicação padrão abaixo
 
    area = resultado.get("proporcao_objeto")
    if area is not None:
        return (
            f"Ação '{acao}' concluída. O objeto ocupa "
            f"{area * 100:.1f}% da imagem e o fundo foi tornado transparente."
        )
    return f"Ação '{acao}' concluída."
 