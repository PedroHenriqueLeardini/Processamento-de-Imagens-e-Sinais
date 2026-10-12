# Removedor de Fundo Inteligente
## Roadmap de Desenvolvimento e Avaliação

### 1. Objetivo do projeto

Desenvolver uma aplicação capaz de remover automaticamente o fundo de imagens coloridas, comparando diferentes técnicas de segmentação baseadas em processamento digital de imagens e inteligência artificial.

O projeto também deverá investigar se a combinação entre uma máscara produzida por IA e técnicas tradicionais de processamento de imagens consegue produzir um resultado superior ao uso isolado da IA.

O sistema será dividido em quatro grandes áreas:

1. **Processamento tradicional de imagens**
2. **Segmentação baseada em IA**
3. **Avaliação e comparação dos resultados**
4. **IA generativa para interação e orquestração**

---

# 2. Arquitetura geral

O projeto deverá possuir dois fluxos principais.

## 2.1 Fluxo experimental

Utilizado para comparar as diferentes técnicas:

```text
                         IMAGEM ORIGINAL
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
             ▼                  ▼                  ▼
            RGB                HSV                LAB
             │                  │                  │
             ▼                  ▼                  ▼
          Máscara            Máscara            Máscara
             │                  │                  │
             ▼                  ▼                  ▼
        Refinamento        Refinamento        Refinamento
             │                  │                  │
             └──────────────────┼──────────────────┘
                                │
                                ▼
                       Avaliação dos métodos
                                │
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Modelo de IA        │
                     │ Segmentação         │
                     └──────────┬──────────┘
                                │
                                ▼
                           Máscara IA
                          /           \
                         /             \
                        ▼               ▼
                  IA pura          IA + refinamento
                        │               │
                        └───────┬───────┘
                                ▼
                       Avaliação final
```

## 2.2 Fluxo da aplicação

Utilizado pelo usuário:

```text
Usuário
   │
   │ imagem + instrução
   ▼
IA Generativa
   │
   │ interpretação
   ▼
Ação estruturada
   │
   ▼
Sistema de processamento
   │
   ▼
Segmentação / refinamento
   │
   ▼
Imagem final
```

---

# 3. Organização dos módulos

Estrutura inicial recomendada:

```text
removedor_fundo/
│
├── main.py
│
├── image_processing.py
├── segmentation.py
├── evaluation.py
├── ai.py
├── pipeline.py
│
├── input/
│   └── imagens/
│
├── ground_truth/
│   └── mascaras/
│
├── output/
│   ├── rgb/
│   ├── hsv/
│   ├── lab/
│   ├── ai/
│   ├── hybrid/
│   └── comparisons/
│
└── tests/
```

A separação poderá ser ampliada posteriormente, mas essa estrutura já mantém as responsabilidades bem definidas.

---

# 4. Módulo `image_processing.py`

## Responsabilidade

Conter as operações tradicionais de processamento de imagens.

Este módulo NÃO deve decidir qual método usar e NÃO deve possuir lógica de IA.

---

## 4.1 Carregamento

### `carregar_imagem()`

Responsável por carregar uma imagem usando OpenCV.

```python
def carregar_imagem(caminho):
    ...
```

Entrada:

```text
caminho da imagem
```

Saída:

```text
imagem BGR
```

---

## 4.2 Conversão para RGB

### `converter_rgb()`

```python
def converter_rgb(imagem):
    ...
```

Responsabilidade:

- converter BGR → RGB;
- permitir visualização correta em Matplotlib;
- fornecer representação RGB para experimentos que utilizem esse espaço.

---

## 4.3 Conversão para HSV

### `converter_hsv()`

```python
def converter_hsv(imagem):
    ...
```

Responsabilidade:

- converter BGR → HSV;
- separar matiz, saturação e brilho;
- fornecer a representação utilizada pelo método HSV.

---

## 4.4 Conversão para LAB

### `converter_lab()`

```python
def converter_lab(imagem):
    ...
```

Responsabilidade:

- converter BGR → LAB;
- separar luminosidade e componentes de cor;
- fornecer a representação utilizada pelo método LAB.

---

## 4.5 Conversão para grayscale

### `converter_grayscale()`

```python
def converter_grayscale(imagem):
    ...
```

Responsabilidade:

- converter a imagem para escala de cinza;
- permitir técnicas baseadas em intensidade;
- futuramente permitir operações como detecção de bordas.

O grayscale será estudado e poderá ser utilizado em técnicas auxiliares, mas não será necessariamente um dos principais métodos de comparação.

---

## 4.6 Operações de máscara

### `aplicar_mascara()`

```python
def aplicar_mascara(imagem, mascara):
    ...
```

Responsabilidade:

Aplicar uma máscara à imagem original.

---

## 4.7 Refinamento morfológico

### `refinar_mascara()`

```python
def refinar_mascara(mascara):
    ...
```

Responsabilidade:

Melhorar máscaras utilizando operações como:

- erosão;
- dilatação;
- abertura;
- fechamento.

O objetivo é:

- remover pequenos ruídos;
- preencher pequenas falhas;
- melhorar regiões fragmentadas;
- suavizar imperfeições da máscara.

---

## 4.8 Criação do Alpha

### `criar_alpha()`

```python
def criar_alpha(mascara):
    ...
```

Responsabilidade:

Transformar a máscara em canal Alpha.

```text
0   → transparente
255 → opaco
```

---

## 4.9 Composição da imagem final

### `criar_imagem_transparente()`

```python
def criar_imagem_transparente(imagem, mascara):
    ...
```

Responsabilidade:

Combinar:

```text
B + G + R + Alpha
```

e produzir uma imagem BGRA.

---

## 4.10 Salvamento

### `salvar_imagem()`

```python
def salvar_imagem(caminho, imagem):
    ...
```

Responsabilidade:

Salvar o resultado, principalmente em PNG quando houver transparência.

---

# 5. Módulo `segmentation.py`

## Responsabilidade

Conter os diferentes métodos de geração de máscaras.

Este módulo utiliza as funções de `image_processing.py`, mas é responsável pela **segmentação**.

---

# 5.1 Segmentação RGB

### `segmentar_rgb()`

```python
def segmentar_rgb(imagem, parametros):
    ...
```

Fluxo:

```text
Imagem BGR
     ↓
RGB
     ↓
Threshold
     ↓
Máscara RGB
```

Objetivo:

Criar uma máscara utilizando os valores dos canais RGB.

---

# 5.2 Segmentação HSV

### `segmentar_hsv()`

```python
def segmentar_hsv(imagem, parametros):
    ...
```

Fluxo:

```text
Imagem BGR
     ↓
HSV
     ↓
Threshold de H/S/V
     ↓
Máscara HSV
```

Esse deverá ser um dos principais métodos tradicionais.

---

# 5.3 Segmentação LAB

### `segmentar_lab()`

```python
def segmentar_lab(imagem, parametros):
    ...
```

Fluxo:

```text
Imagem BGR
     ↓
LAB
     ↓
Threshold / distância entre canais
     ↓
Máscara LAB
```

Objetivo:

Investigar se a separação entre luminosidade e componentes de cor produz uma segmentação melhor.

---

# 5.4 Refinamento das máscaras clássicas

Cada método poderá gerar duas versões:

```text
RGB
 ├── máscara original
 └── máscara refinada

HSV
 ├── máscara original
 └── máscara refinada

LAB
 ├── máscara original
 └── máscara refinada
```

Isso permitirá avaliar o impacto do processamento.

---

# 6. Segmentação baseada em IA

Ainda dentro de `segmentation.py` ou em um módulo separado posteriormente:

```text
ai_segmentation.py
```

## Responsabilidade

Executar um modelo de segmentação pré-treinado.

---

## 6.1 `carregar_modelo()`

```python
def carregar_modelo():
    ...
```

Responsabilidade:

Carregar o modelo de segmentação.

---

## 6.2 `segmentar_ia()`

```python
def segmentar_ia(imagem):
    ...
```

Fluxo:

```text
Imagem
   ↓
Modelo de segmentação
   ↓
Máscara IA
```

A máscara produzida pela IA será preservada antes de qualquer processamento.

Isso é importante para a comparação.

---

# 7. Pipeline IA pura

Criar:

### `executar_ia()`

```python
def executar_ia(imagem):
    ...
```

Fluxo:

```text
Imagem
  ↓
IA
  ↓
Máscara IA
  ↓
Alpha
  ↓
Imagem final
```

Esse resultado será chamado de:

> **IA pura**

Ele será a linha de base para comparar com o método híbrido.

---

# 8. Pipeline IA + processamento

Criar:

### `executar_ia_refinada()`

```python
def executar_ia_refinada(imagem):
    ...
```

Fluxo:

```text
Imagem
  ↓
IA
  ↓
Máscara IA
  ↓
Refinamento
  ↓
Máscara IA refinada
  ↓
Alpha
  ↓
Imagem final
```

Esse resultado será chamado de:

> **IA + processamento**

ou:

> **Segmentação híbrida**

Essa será uma das partes centrais do projeto.

---

# 9. Módulo `evaluation.py`

## Responsabilidade

Medir quantitativamente a qualidade das máscaras.

Para isso será necessária uma **Ground Truth**.

---

# 9.1 Ground Truth

Para cada imagem experimental deverá existir uma máscara considerada correta.

```text
input/
    pessoa01.jpg

ground_truth/
    pessoa01.png
```

A Ground Truth representa:

```text
255 → objeto
0   → fundo
```

Ela será usada somente para avaliação.

---

# 9.2 IoU

### `calcular_iou()`

```python
def calcular_iou(mascara_predita, ground_truth):
    ...
```

Mede a interseção entre a máscara produzida e a máscara correta.

Quanto mais próximo de 1, melhor.

---

# 9.3 Precision

### `calcular_precision()`

```python
def calcular_precision(predicao, ground_truth):
    ...
```

Avalia quantos pixels classificados como objeto realmente pertencem ao objeto.

---

# 9.4 Recall

### `calcular_recall()`

```python
def calcular_recall(predicao, ground_truth):
    ...
```

Avalia quanto do objeto real foi encontrado.

---

# 9.5 F1-score

### `calcular_f1()`

```python
def calcular_f1(precision, recall):
    ...
```

Combina Precision e Recall.

---

# 9.6 Comparação de métodos

### `avaliar_metodo()`

```python
def avaliar_metodo(mascara, ground_truth):
    ...
```

Retorna algo estruturado:

```python
{
    "iou": ...,
    "precision": ...,
    "recall": ...,
    "f1": ...
}
```

---

# 10. Avaliação do ganho do refinamento

Criar:

### `calcular_melhoria()`

```python
def calcular_melhoria(valor_original, valor_refinado):
    ...
```

Principalmente para:

```text
IA
   ↓
IA + processamento
```

Por exemplo:

```text
IoU IA              = 0.89
IoU IA + processamento = 0.94

Melhoria            = +0.05
```

Também poderá ser apresentada como porcentagem.

---

# 11. Experimentos principais

O projeto deverá realizar pelo menos os seguintes experimentos.

## Experimento A — RGB

```text
Imagem
 ↓
RGB
 ↓
Máscara
 ↓
Avaliação
```

---

## Experimento B — RGB + refinamento

```text
Imagem
 ↓
RGB
 ↓
Máscara
 ↓
Morfologia
 ↓
Avaliação
```

---

## Experimento C — HSV

```text
Imagem
 ↓
HSV
 ↓
Máscara
 ↓
Avaliação
```

---

## Experimento D — HSV + refinamento

```text
Imagem
 ↓
HSV
 ↓
Máscara
 ↓
Morfologia
 ↓
Avaliação
```

---

## Experimento E — LAB

```text
Imagem
 ↓
LAB
 ↓
Máscara
 ↓
Avaliação
```

---

## Experimento F — LAB + refinamento

```text
Imagem
 ↓
LAB
 ↓
Máscara
 ↓
Morfologia
 ↓
Avaliação
```

---

## Experimento G — IA

```text
Imagem
 ↓
Modelo IA
 ↓
Máscara IA
 ↓
Avaliação
```

---

## Experimento H — IA + processamento

```text
Imagem
 ↓
Modelo IA
 ↓
Máscara IA
 ↓
Morfologia/refinamento
 ↓
Máscara híbrida
 ↓
Avaliação
```

---

# 12. Tabela final de comparação

O sistema deverá produzir uma tabela semelhante a:

| Método | Refinamento | IoU | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| RGB | Não | — | — | — | — |
| RGB | Sim | — | — | — | — |
| HSV | Não | — | — | — | — |
| HSV | Sim | — | — | — | — |
| LAB | Não | — | — | — | — |
| LAB | Sim | — | — | — | — |
| IA | Não | — | — | — | — |
| IA | Sim | — | — | — | — |

Os valores serão obtidos experimentalmente.

---

# 13. Comparação visual

Além das métricas numéricas, o sistema deverá salvar:

```text
imagem_original.png
mascara_rgb.png
resultado_rgb.png

mascara_hsv.png
resultado_hsv.png

mascara_lab.png
resultado_lab.png

mascara_ia.png
resultado_ia.png

mascara_ia_refinada.png
resultado_ia_refinada.png
```

Isso permitirá analisar visualmente:

- bordas;
- cabelos;
- pequenos objetos;
- sombras;
- regiões internas;
- vazamentos do fundo;
- buracos na máscara;
- ruídos.

---

# 14. Visualização dos resultados

Criar futuramente no módulo `evaluation.py` ou `visualization.py`:

### `gerar_grafico_comparacao()`

Apresentar, por exemplo:

```text
              IoU
               │
               │             █
               │       █     █
               │  █    █     █
               │  █    █     █
               └────────────────
                 HSV  IA  IA+Proc.
```

Os gráficos poderão comparar:

- IoU;
- F1;
- Precision;
- Recall;
- melhoria da IA após refinamento.

---

# 15. Módulo `pipeline.py`

## Responsabilidade

Orquestrar os diferentes experimentos.

Ele não deverá implementar algoritmos de processamento.

Sua função será chamar as funções corretas na ordem correta.

Exemplo conceitual:

```python
def executar_experimento(imagem):
    
    mascara_rgb = segmentar_rgb(imagem)
    mascara_rgb_refinada = refinar_mascara(mascara_rgb)

    mascara_hsv = segmentar_hsv(imagem)
    mascara_hsv_refinada = refinar_mascara(mascara_hsv)

    mascara_lab = segmentar_lab(imagem)
    mascara_lab_refinada = refinar_mascara(mascara_lab)

    mascara_ia = segmentar_ia(imagem)
    mascara_ia_refinada = refinar_mascara(mascara_ia)

    ...
```

Isso mantém a arquitetura organizada.

---

# 16. Módulo `ai.py`

## Responsabilidade

Implementar a IA generativa.

A IA generativa não deverá manipular pixels.

Ela será responsável por interpretar a solicitação do usuário.

Exemplo:

```text
Usuário:

"Remova o fundo dessa imagem e mantenha apenas o cachorro."
```

A IA deverá interpretar:

```json
{
    "acao": "remover_fundo",
    "objeto": "cachorro",
    "metodo": "ia"
}
```

---

# 17. Tool Calling

Posteriormente, a IA poderá selecionar funções reais do sistema.

Por exemplo:

```text
Usuário
   ↓
IA generativa
   ↓
JSON
   ↓
Validação
   ↓
Função Python
   ↓
Resultado
```

A IA poderá escolher:

```text
segmentar_rgb()
segmentar_hsv()
segmentar_lab()
segmentar_ia()
executar_ia_refinada()
```

Isso fará com que a IA generativa realmente participe da aplicação.

---

# 18. Regra de separação das IAs

O projeto deverá manter claramente duas funções diferentes de IA:

### IA generativa

Responsável por:

- interpretar linguagem natural;
- identificar intenção;
- escolher ação;
- chamar funções;
- explicar o resultado.

### IA de segmentação

Responsável por:

- analisar a imagem;
- identificar o objeto;
- produzir a máscara.

### Processamento tradicional

Responsável por:

- manipular pixels;
- tratar máscaras;
- aplicar morfologia;
- criar Alpha;
- gerar PNG transparente.

Portanto:

```text
IA generativa
     ↓
"o que o usuário quer?"

IA de segmentação
     ↓
"onde está o objeto?"

Processamento de imagens
     ↓
"como refinar e produzir a imagem?"
```

---

# 19. Interface final

Na última etapa poderá ser criada uma interface simples utilizando, por exemplo, Streamlit.

A interface deverá permitir:

```text
┌─────────────────────────────────┐
│ Removedor de Fundo Inteligente  │
├─────────────────────────────────┤
│                                 │
│      [Selecionar imagem]        │
│                                 │
│ Comando:                        │
│ "Remova o fundo do cachorro"    │
│                                 │
│          [Processar]            │
│                                 │
├─────────────────────────────────┤
│ Resultado                       │
│                                 │
│       [Imagem transparente]     │
│                                 │
└─────────────────────────────────┘
```

---

# 20. Organização final das responsabilidades

| Módulo | Responsabilidade |
|---|---|
| `main.py` | Entrada e execução |
| `image_processing.py` | Operações tradicionais |
| `segmentation.py` | Segmentações RGB/HSV/LAB |
| `ai_segmentation.py` | Modelo de segmentação |
| `evaluation.py` | Métricas |
| `visualization.py` | Gráficos e comparações |
| `pipeline.py` | Orquestração dos experimentos |
| `ai.py` | IA generativa |
| `input/` | Imagens de entrada |
| `ground_truth/` | Máscaras corretas |
| `output/` | Resultados experimentais |
| `tests/` | Testes |

---

# 21. Ordem de desenvolvimento

## Fase 1 — Fundamentos

Implementar:

- carregamento;
- BGR;
- RGB;
- grayscale;
- HSV;
- LAB;
- canais;
- pixels;
- visualização.

**Objetivo:** entender como uma imagem colorida é representada.

---

## Fase 2 — Segmentação tradicional

Implementar:

- `segmentar_rgb()`;
- `segmentar_hsv()`;
- `segmentar_lab()`;
- geração de máscaras;
- aplicação de máscaras.

**Objetivo:** obter diferentes máscaras para a mesma imagem.

---

## Fase 3 — Refinamento

Implementar:

- erosão;
- dilatação;
- abertura;
- fechamento;
- refinamento das máscaras;
- Alpha;
- PNG transparente.

**Objetivo:** melhorar os resultados produzidos pelos métodos tradicionais.

---

## Fase 4 — Dataset experimental

Criar:

- conjunto de imagens;
- máscaras Ground Truth;
- organização dos arquivos;
- diferentes tipos de imagens e fundos.

É importante não testar somente uma imagem.

---

## Fase 5 — Avaliação

Implementar:

- IoU;
- Precision;
- Recall;
- F1;
- comparação automática;
- geração de tabela;
- gráficos.

**Objetivo:** transformar a comparação visual em uma avaliação quantitativa.

---

## Fase 6 — IA de segmentação

Implementar:

- carregamento do modelo;
- segmentação;
- geração da máscara;
- resultado IA puro.

**Objetivo:** estabelecer a IA como uma nova abordagem de segmentação.

---

## Fase 7 — Pipeline híbrido

Implementar:

```text
IA
 ↓
Máscara
 ↓
Processamento
 ↓
Máscara refinada
```

Comparar:

```text
IA
vs.
IA + processamento
```

Calcular:

- IoU;
- Precision;
- Recall;
- F1;
- ganho absoluto;
- ganho percentual.

**Objetivo:** verificar experimentalmente se o processamento tradicional melhora a saída da IA.

---

## Fase 8 — Comparação completa

Executar:

```text
RGB
RGB + processamento

HSV
HSV + processamento

LAB
LAB + processamento

IA
IA + processamento
```

Produzir:

- máscaras;
- imagens finais;
- métricas;
- gráficos;
- ranking dos métodos.

---

## Fase 9 — IA generativa

Implementar:

- interpretação de comandos;
- JSON estruturado;
- validação;
- tool calling;
- seleção de pipeline;
- explicação do resultado.

---

## Fase 10 — Interface

Criar a interface final com:

- upload da imagem;
- comando em linguagem natural;
- seleção do método, se desejado;
- processamento;
- visualização;
- comparação;
- métricas;
- download do resultado.

---

# 22. Resultado final esperado

Ao final, o projeto deverá conseguir responder experimentalmente:

### Qual espaço de cor apresenta melhor segmentação?

```text
RGB × HSV × LAB
```

### O processamento melhora as máscaras tradicionais?

```text
RGB × RGB + refinamento
HSV × HSV + refinamento
LAB × LAB + refinamento
```

### A IA apresenta resultado superior aos métodos tradicionais?

```text
IA × RGB/HSV/LAB
```

### O processamento tradicional consegue melhorar a máscara produzida pela IA?

```text
IA × IA + processamento
```

### Qual é a melhor combinação?

```text
Métodos clássicos
       ↓
     IA
       ↓
IA + processamento
```

---

# 23. Conceito central do projeto

O projeto não deverá ser apresentado simplesmente como:

> "Um programa que remove o fundo de uma imagem."

A proposta é:

> **Um sistema experimental de remoção de fundo que compara diferentes espaços de cor e métodos de segmentação, avalia quantitativamente seus resultados e investiga a combinação entre segmentação baseada em inteligência artificial e técnicas tradicionais de processamento digital de imagens.**

A principal hipótese experimental será:

> **Uma máscara produzida por um modelo de segmentação baseado em IA pode apresentar melhoria de qualidade quando submetida a técnicas adequadas de processamento digital de imagens.**

Essa hipótese será verificada por meio de Ground Truth e métricas como **IoU, Precision, Recall e F1-score**.

---

# 24. Pipeline conceitual definitivo

```text
                         USUÁRIO
                            │
                            ▼
                    ┌──────────────┐
                    │ IA GENERATIVA│
                    └──────┬───────┘
                           │
                    Interpretação
                           │
                           ▼
                     Ação / JSON
                           │
                           ▼
                  ┌─────────────────┐
                  │     PIPELINE    │
                  └────────┬────────┘
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
          ▼                ▼                 ▼
         RGB              HSV               LAB
          │                │                 │
          ▼                ▼                 ▼
       Máscara          Máscara           Máscara
          │                │                 │
          ▼                ▼                 ▼
     Refinamento       Refinamento      Refinamento
          │                │                 │
          └────────────────┼─────────────────┘
                           │
                           │
                           ▼
                    COMPARAÇÃO
                           ▲
                           │
                  ┌────────┴────────┐
                  │                 │
                  ▼                 ▼
                IA PURA       IA + PROCESSAMENTO
                  │                 │
                  ▼                 ▼
             Máscara IA       Máscara refinada
                  │                 │
                  └────────┬────────┘
                           │
                           ▼
                     AVALIAÇÃO
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
            IoU         Precision        F1
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    RESULTADOS
                           │
                    ┌──────┴──────┐
                    ▼             ▼
                 Gráficos      Comparação
                    │             │
                    └──────┬──────┘
                           ▼
                     CONCLUSÃO
```