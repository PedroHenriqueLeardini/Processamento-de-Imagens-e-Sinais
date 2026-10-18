Integração com a disciplina de IA e pipeline do projeto
O projeto integra IA em duas camadas complementares, apoiadas pelo processamento digital de imagens clássico:
•	IA perceptiva: um modelo de segmentação pré-treinado (aprendizado profundo aplicado à visão computacional) identifica o objeto na imagem.
•	IA generativa: um modelo de linguagem interpreta o pedido do usuário e aciona as funções da aplicação.
O princípio de projeto, presente no documento, é que cada tecnologia tem uma responsabilidade única. A IA generativa decide o que fazer, o modelo de segmentação encontra o objeto e o processamento de imagens refina e compõe o resultado. A IA generativa nunca manipula pixels diretamente.

Pipeline completo
Usuário (imagem + instrução)
•	IA generativa (interpretação → JSON)
•	Validação dos parâmetros
•	Segmentação (modelo pré-treinado → máscara)
•	Refinamento da máscara (morfologia)
•	Canal Alpha (BGR + máscara = BGRA)
•	PNG com fundo transparente

1. Entrada. O usuário envia uma imagem colorida e uma instrução em linguagem natural, por exemplo "Remova o fundo e mantenha apenas o cachorro", pela interface em Streamlit. Internamente, o OpenCV carrega a imagem em BGR como uma matriz NumPy (altura × largura × 3, uint8). É a mesma representação que, mais adiante, alimenta o modelo de segmentação.

2. Interpretação pela IA generativa. O modelo de linguagem converte o texto livre em uma saída estruturada, como {"acao": "remover_fundo", "objeto": "cachorro"}. Em termos de IA, é uma tarefa de extração de intenção e entidades, feita com engenharia de prompt e saída restrita a um esquema. O modelo atua como orquestrador que escolhe qual ferramenta chamar, sem gerar nem alterar a imagem.

3. Validação. Antes de executar, a aplicação verifica se a ação é suportada e se os campos obrigatórios estão presentes. Se faltar informação (por exemplo, o objeto não foi indicado), o sistema pergunta ao usuário em vez de adaptar o pedido silenciosamente. Essa etapa funciona como um guardrail: mantém a IA dentro de um conjunto de operações controladas e torna o comportamento previsível.

4. Segmentação. Um modelo pré-treinado recebe a imagem e produz uma máscara do objeto de interesse. O projeto usa apenas inferência, sem treinar uma rede do zero, o que é um caso prático de reaproveitamento de modelos (transfer learning). A saída do modelo, normalmente em probabilidades, é binarizada por limiar em uma máscara de um canal (255 = manter, 0 = remover). A segmentação por cor em HSV com cv2.inRange, feita na Etapa 2 do roadmap, serve como base didática e como ponto de comparação com a abordagem baseada em aprendizado.

5. Refinamento. Máscaras de modelos podem ter ruídos, buracos e ilhas desconectadas. Operações morfológicas de abertura (remove ruído) e fechamento (preenche buracos) melhoram a qualidade antes da composição. Essa é a parte do pipeline que depende do conhecimento de processamento de imagens, não de IA.

6. Composição com Alpha. Os canais BGR originais são mantidos e a máscara refinada vira o quarto canal (Alpha), gerando uma imagem BGRA. O Alpha define onde o fundo fica transparente.

7. Saída. O resultado é exibido na interface e salvo em PNG, formato que preserva a transparência.

Componente	              |  Conceito da disciplina	                                       | Papel no projeto
---------------------------------------------------------------------------------------------------------------------------------------------------
Imagem como matriz NumPy  |	Representação de dados para visão computacional	               | Entrada comum ao processamento clássico e ao modelo
Modelo de segmentação	  | Redes neurais, segmentação semântica, modelos pré-treinados	   | Identifica o objeto e gera a máscara
Modelo de linguagem	      |  IA generativa, PLN, engenharia de prompt, saída estruturada   |    Traduz linguagem natural em ação e parâmetros
Chamada de funções	      |  Agentes e uso de ferramentas	                               | Orquestra as operações sem tocar nos pixels
Validação	              |  IA responsável, controle e segurança	                       |   Impede ações não suportadas ou ambíguas

Limitações
•	O resultado depende da qualidade da máscara.
•	Objetos com cor parecida com a do fundo são mais difíceis de separar.
•	Cabelos, pelos e bordas finas ficam menos precisos.

4.1. Tratamento de erros da segmentação. O modelo de segmentação pode produzir uma máscara incorreta, incompleta ou com baixa qualidade. Por isso, a aplicação não deve assumir que toda máscara gerada pelo modelo está correta. Antes de prosseguir para o refinamento e a composição da imagem, o sistema realiza verificações sobre o resultado da segmentação, como a existência de uma região válida correspondente ao objeto solicitado e a proporção da área identificada em relação à imagem.
Caso a máscara apresente um resultado claramente inadequado, como nenhum objeto identificado ou uma região que ocupe praticamente toda a imagem quando isso não era esperado, o sistema interrompe o processamento e informa ao usuário que não foi possível identificar o objeto com segurança. A aplicação pode então solicitar uma nova imagem ou uma instrução mais específica.
Esse mecanismo é importante porque a IA não é considerada infalível. A função da validação é impedir que um resultado possivelmente incorreto seja transformado automaticamente em uma imagem final. Dessa forma, o sistema mantém uma separação entre a predição do modelo e a decisão de aceitá-la para o processamento, tornando o comportamento mais previsível e controlado.



## Atulizado a partir daqui

# Documentação Técnica — Projeto de Remoção de Fundo em Imagens Coloridas

## 1. Apresentação do projeto

O projeto tem como objetivo desenvolver uma aplicação para remoção de fundo em imagens coloridas, utilizando técnicas de processamento digital de imagens e inteligência artificial.

A proposta é identificar o objeto principal de uma imagem, separar esse objeto do fundo e gerar uma imagem com transparência, preservando os pixels pertencentes ao objeto selecionado.

O desenvolvimento foi organizado de maneira progressiva. Inicialmente, são estudadas técnicas tradicionais de processamento de imagens, incluindo conversão entre espaços de cores, limiarização, criação de máscaras binárias, operações morfológicas e avaliação quantitativa dos resultados. Essa etapa estabelece uma referência para que, posteriormente, seja possível comparar os resultados tradicionais com os obtidos por técnicas de segmentação baseadas em inteligência artificial.

A inteligência artificial constitui uma parte essencial da arquitetura final do projeto. A proposta prevê a utilização de duas camadas complementares: uma IA perceptiva, responsável pela identificação do objeto na imagem por meio de um modelo de segmentação, e uma IA generativa, responsável por interpretar as instruções do usuário e coordenar as operações da aplicação.

## 2. Arquitetura e fluxo de processamento

O fluxo tradicional desenvolvido até o momento pode ser representado pelas seguintes etapas:

1. Carregamento da imagem colorida.
2. Conversão da imagem para diferentes espaços de cores.
3. Aplicação de limiarização para identificar os pixels que correspondem aos intervalos definidos.
4. Geração de máscaras binárias.
5. Aplicação de operações morfológicas para refinar as máscaras.
6. Criação de imagens com fundo transparente por meio de um canal alfa.
7. Comparação das máscaras com as anotações de referência do dataset.
8. Cálculo das métricas de avaliação.
9. Visualização dos resultados por meio de gráficos e imagens comparativas.

Esse fluxo permite estudar separadamente a influência do espaço de cores e das operações morfológicas sobre a qualidade da segmentação.

Na arquitetura futura, a IA generativa interpretará a solicitação do usuário e produzirá parâmetros estruturados para a aplicação. Após a validação desses parâmetros, um modelo de segmentação identificará o objeto e produzirá uma máscara inicial. As técnicas tradicionais de processamento digital de imagens poderão então refinar essa máscara e gerar a imagem final com transparência.

Cada componente terá uma responsabilidade específica: a IA generativa interpretará a intenção do usuário; o modelo de segmentação identificará o objeto; e o processamento digital de imagens realizará o refinamento e a composição da imagem.

## 3. Organização dos módulos Python

O projeto utiliza módulos separados para organizar as responsabilidades do processamento.

### 3.1. `main.py`

O arquivo principal coordena a execução do experimento. Ele carrega a imagem, realiza as conversões de espaço de cores, cria as máscaras, aplica as operações morfológicas, organiza as visualizações e executa a avaliação dos resultados.

Também é responsável por apresentar as imagens e os gráficos comparativos, facilitando a análise dos resultados de cada configuração.

### 3.2. `image_processing.py`

Esse módulo concentra as funções relacionadas ao carregamento e à preparação das imagens.

Entre as funções utilizadas estão:

- `carregar_imagem()`: carrega a imagem original.
- `converter_rgb()`: converte a imagem para o espaço RGB utilizado na análise.
- `converter_hsv()`: converte a imagem para o espaço HSV.
- `converter_lab()`: converte a imagem para o espaço LAB.
- `converter_grayscale()`: converte a imagem para escala de cinza.
- `criar_imagem_transparente()`: utiliza uma máscara para determinar quais pixels permanecerão visíveis e quais receberão transparência.

Embora a conversão para escala de cinza esteja disponível no módulo, a comparação experimental atual está concentrada nos espaços RGB, HSV e LAB.

O OpenCV normalmente carrega imagens coloridas no formato BGR. Por isso, é necessário controlar as conversões entre BGR, RGB e RGBA para evitar que as cores sejam exibidas incorretamente. A imagem transparente é produzida com quatro canais: azul, verde, vermelho e alfa, no formato BGRA do OpenCV.

### 3.3. `segmentation.py`

Esse módulo reúne as funções de segmentação e refinamento das máscaras.

As funções de criação de máscaras utilizam limiarização por intervalo de valores. Em vez de analisar semanticamente o conteúdo da imagem, o algoritmo seleciona pixels cujos valores estão dentro dos limites estabelecidos para cada canal.

As funções utilizadas são:

- `criar_mascara_rgb()`: gera uma máscara a partir de intervalos de valores RGB.
- `criar_mascara_hsv()`: gera uma máscara a partir de intervalos de matiz, saturação e valor.
- `criar_mascara_lab()`: gera uma máscara a partir dos canais do espaço LAB.
- `refinar_mascara()`: função destinada ao refinamento de máscaras.
- `abertura()`: aplica abertura morfológica.
- `fechamento()`: aplica fechamento morfológico.
- `dilatacao()`: expande as regiões brancas da máscara.
- `erosao()`: reduz as regiões brancas da máscara.
- `gradiente()`: calcula o gradiente morfológico, que pode auxiliar na análise das bordas.

Os limites utilizados nos experimentos são parâmetros que precisam ser ajustados conforme a imagem. Os valores inicialmente empregados são exemplos de configuração, não limites universais para todos os objetos ou fundos.

### 3.4. `dataset_utils.py`

Esse módulo é responsável pelo acesso ao dataset utilizado nos experimentos.

Entre as funções utilizadas estão:

- `listar_imagens()`: identifica e lista as imagens disponíveis no conjunto de dados.
- `carregar_par()`: carrega uma imagem e seu respectivo trimap de referência.

O uso de pares de imagem e anotação permite avaliar as máscaras produzidas pelo algoritmo, em vez de depender exclusivamente da inspeção visual.

### 3.5. `evaluation.py`

O módulo de avaliação recebe uma máscara binária e um trimap de referência. Ele verifica os tipos dos dados, confirma que ambos são arrays NumPy bidimensionais, verifica se as dimensões coincidem e valida se o trimap contém os valores esperados: 1, 2 e 3.

Após essas verificações, calcula as métricas tradicionais de segmentação e as métricas complementares relacionadas à região de incerteza.

## 4. Comparação entre os espaços de cores

Uma das principais etapas do projeto consiste em comparar a segmentação realizada nos espaços RGB, HSV e LAB.

### 4.1. RGB

O espaço RGB representa as cores por meio dos canais vermelho, verde e azul. A limiarização nesse espaço seleciona os pixels que atendem simultaneamente aos intervalos definidos para esses canais.

Sua principal vantagem experimental é a simplicidade. Entretanto, os valores RGB são influenciados pela iluminação e pela combinação entre os canais, o que pode dificultar a separação entre objeto e fundo quando ambos possuem cores semelhantes.

### 4.2. HSV

O espaço HSV representa a cor por meio de matiz (Hue), saturação (Saturation) e valor (Value).

A separação desses componentes permite estabelecer limites com base na tonalidade e na intensidade da cor. Isso pode facilitar a seleção de determinadas cores em situações nas quais os canais RGB não oferecem uma separação conveniente.

No OpenCV, a matiz de imagens HSV de 8 bits geralmente varia de 0 a 179, enquanto a saturação e o valor variam de 0 a 255. Os limites devem respeitar essa representação.

### 4.3. LAB

O espaço LAB utiliza três canais: L*, relacionado à luminosidade, a*, relacionado ao eixo entre verde e vermelho, e b*, relacionado ao eixo entre azul e amarelo.

Essa representação permite estudar separadamente a luminosidade e componentes cromáticos. Entretanto, os valores disponibilizados pelo OpenCV em imagens de 8 bits são codificados em uma faixa de 0 a 255, não correspondendo diretamente aos valores perceptuais nominais de L*, a* e b*.

### 4.4. Objetivo da comparação

O propósito não é assumir antecipadamente que um espaço de cores será superior aos outros, mas verificar experimentalmente como cada representação afeta a seleção dos pixels e a qualidade da máscara.

Para que a comparação seja válida, as três máscaras devem ser produzidas a partir da mesma imagem, avaliadas contra o mesmo trimap e submetidas ao mesmo protocolo de cálculo das métricas.

## 5. Criação das máscaras e transparência

A limiarização gera uma máscara binária na qual os pixels brancos representam as regiões selecionadas como objeto, enquanto os pixels pretos representam as regiões consideradas fundo.

A máscara é então utilizada para criar uma imagem com canal alfa. Os pixels classificados como objeto permanecem visíveis, enquanto os classificados como fundo recebem transparência.

Esse procedimento possibilita gerar arquivos de imagem com fundo transparente, como PNG, sem apagar fisicamente os pixels da imagem original. A máscara determina a visibilidade de cada pixel na composição final.

A qualidade da imagem transparente depende diretamente da qualidade da máscara. Uma máscara que inclua partes do fundo manterá regiões indesejadas visíveis. Por outro lado, uma máscara que exclua partes do objeto removerá detalhes que deveriam ser preservados.

Por esse motivo, o projeto também investiga o refinamento morfológico das máscaras.

## 6. Operações morfológicas

As operações morfológicas atuam sobre as regiões da máscara e permitem modificar sua estrutura. Elas não identificam semanticamente o objeto: apenas transformam as regiões selecionadas de acordo com a operação e o elemento estruturante utilizados.

### 6.1. Abertura

A abertura corresponde à erosão seguida de dilatação. Ela pode remover pequenas regiões brancas isoladas e reduzir ruídos claros, preservando parte da estrutura principal do objeto.

Se for aplicada de maneira excessiva, poderá eliminar partes finas do objeto e detalhes importantes.

### 6.2. Fechamento

O fechamento corresponde à dilatação seguida de erosão. Ele pode preencher pequenos buracos e conectar regiões brancas próximas.

Quando utilizado em excesso, pode unir o objeto ao fundo ou preencher áreas que deveriam permanecer transparentes.

### 6.3. Dilatação

A dilatação expande as regiões brancas da máscara. Na segmentação, isso pode recuperar partes do objeto que ficaram de fora, mas também pode incorporar pixels pertencentes ao fundo.

### 6.4. Erosão

A erosão reduz as regiões brancas da máscara. Ela pode remover pequenas regiões indevidamente selecionadas, mas também pode cortar as bordas e reduzir a área do objeto.

### 6.5. Gradiente morfológico

O gradiente morfológico representa a diferença entre a dilatação e a erosão de uma imagem. Ele pode ser utilizado para evidenciar contornos, auxiliando na análise das bordas da máscara.

A investigação dessas operações permite comparar a máscara original com versões refinadas e determinar se cada transformação melhora ou prejudica os resultados quantitativos.

## 7. Visualização das imagens e máscaras

O arquivo principal foi organizado para apresentar diferentes resultados em painéis comparativos.

As visualizações incluem:

- Imagem original.
- Imagem convertida para RGB.
- Representações da imagem em HSV e LAB.
- Máscara original de cada espaço de cores.
- Máscaras após abertura.
- Máscaras após fechamento.
- Máscaras após dilatação.
- Máscaras após erosão.
- Imagens transparentes produzidas a partir das máscaras.

Essa organização facilita a identificação visual de ruídos, falhas nas bordas, perda de partes do objeto e inclusão indevida de regiões do fundo.

As representações numéricas de HSV e LAB não devem ser interpretadas automaticamente como imagens em cores naturais. Para visualizar esses espaços de maneira fiel, é necessário convertê-los para um formato de exibição adequado. Para analisar as máscaras binárias, a representação em escala de cinza é suficiente.

## 8. Uso do dataset Oxford-IIIT Pet

O projeto utiliza o dataset Oxford-IIIT Pet, que contém imagens de animais acompanhadas de anotações trimap.

O trimap fornece uma referência para distinguir três regiões:

- **Valor 1 — animal:** região identificada como primeiro plano.
- **Valor 2 — fundo:** região identificada como plano de fundo.
- **Valor 3 — região incerta:** região de fronteira cuja classificação não é tratada como definitiva nas métricas tradicionais utilizadas no projeto.

O trimap é importante porque permite comparar a máscara produzida pelo algoritmo com uma anotação de referência independente.

Durante a preparação do dataset, foi necessário considerar a correspondência entre as imagens e seus respectivos trimaps, além de verificar as dimensões dos arquivos. Também foram encontrados arquivos auxiliares com prefixos como `._`, que podem aparecer em diretórios extraídos de sistemas que geram metadados adicionais. Esses arquivos não devem ser tratados como imagens válidas do dataset.

As dimensões das imagens variam. Portanto, a máscara e o trimap utilizados em cada avaliação precisam possuir exatamente as mesmas dimensões. Uma diferença de resolução exige um tratamento explícito e consistente, pois a comparação pixel a pixel não pode ser realizada corretamente entre arrays de tamanhos diferentes.

Outro cuidado importante é preservar os valores originais do trimap. Normalizar os valores 1, 2 e 3 como se fossem uma máscara binária eliminaria a distinção entre animal, fundo e região incerta.

## 9. Métricas tradicionais de avaliação

O módulo `evaluation.py` calcula cinco métricas principais. Para esses cálculos, os pixels da região incerta, identificados pelo valor 3, são excluídos.

Os valores são obtidos a partir de quatro categorias:

- **TP (verdadeiros positivos):** pixels de animal corretamente previstos como animal.
- **FP (falsos positivos):** pixels de fundo previstos incorretamente como animal.
- **FN (falsos negativos):** pixels de animal previstos incorretamente como fundo.
- **TN (verdadeiros negativos):** pixels de fundo corretamente previstos como fundo.

### 9.1. IoU — Intersection over Union

O IoU mede a interseção entre a região prevista como animal e a região de referência, dividida pela união dessas regiões.

Valores maiores indicam uma sobreposição melhor entre a máscara e a referência.

### 9.2. Precisão

A precisão mede a proporção de pixels previstos como animal que realmente pertencem ao animal na referência.

Uma precisão baixa indica que a máscara pode estar incorporando muitas regiões de fundo.

### 9.3. Recall

O recall mede a proporção dos pixels de animal da referência que foram identificados corretamente.

Um recall baixo indica que partes do objeto podem estar sendo removidas ou classificadas como fundo.

### 9.4. F1-score

O F1-score combina precisão e recall por meio da média harmônica entre eles.

Essa métrica ajuda a avaliar o equilíbrio entre evitar a inclusão de fundo e preservar as partes do objeto.

### 9.5. Acurácia

A acurácia representa a proporção de classificações corretas entre os pixels válidos considerados na avaliação.

Ela deve ser interpretada junto com IoU, precisão, recall e F1-score, pois a quantidade de pixels de fundo pode influenciar fortemente seu resultado.

As cinco métricas variam de 0 a 1 no cálculo implementado. Valores próximos de 1 geralmente indicam melhor desempenho. Quando o denominador de uma métrica é zero, a função utiliza divisão segura e retorna 0, evitando uma exceção matemática.

## 10. Avaliação da região de incerteza

Além das métricas tradicionais, a função `avaliar_mascara()` calcula cinco indicadores específicos da região de incerteza.

Esses indicadores não são utilizados como substitutos das métricas tradicionais. Seu propósito é descrever como a máscara se comporta na fronteira entre objeto e fundo.

### 10.1. `pixels_regiao_incerta`

Conta quantos pixels do trimap pertencem à classe 3.

Como esse valor depende do trimap e não da máscara produzida, deve permanecer igual para todas as máscaras avaliadas contra a mesma imagem e referência.

### 10.2. `pixels_previstos_na_incerteza`

Conta quantos pixels foram previstos como animal pela máscara dentro da região incerta.

O valor permite observar quantos pixels da máscara ocupam a região de fronteira, mas, isoladamente, não indica se a segmentação é melhor ou pior.

### 10.3. `proporcao_mascara_na_incerteza`

Representa a fração dos pixels previstos como animal que estão localizados na região incerta.

Essa proporção é calculada dividindo os pixels previstos como animal na região incerta pelo total de pixels previstos como animal na máscara.

Uma proporção elevada indica que uma parcela maior da área selecionada pela máscara está localizada nessa região. Isso não significa necessariamente uma fronteira melhor ou pior.

### 10.4. `proporcao_incerteza_como_animal`

Representa a fração da região incerta que a máscara classificou como animal.

É calculada dividindo os pixels previstos como animal na região incerta pelo total de pixels dessa região.

### 10.5. `proporcao_incerteza_como_fundo`

Representa a fração da região incerta que a máscara classificou como fundo.

É calculada dividindo os pixels da região incerta não previstos como animal pelo total de pixels incertos.

Quando existe pelo menos um pixel na região incerta e a máscara é binária, as proporções classificadas como animal e como fundo somam 1, ressalvados pequenos efeitos de representação numérica.

As métricas de incerteza são indicadores diagnósticos. Uma proporção maior de pixels classificados como animal nessa região não deve ser interpretada automaticamente como uma melhora. A avaliação precisa considerar também as métricas tradicionais e, se necessário, uma análise específica da qualidade das bordas.

## 11. Comparação quantitativa entre máscaras

A etapa de avaliação foi ampliada para comparar não somente os espaços de cores, mas também as diferentes operações morfológicas.

Para isso, as máscaras são organizadas em três grupos:

- RGB: máscara original, abertura, fechamento, dilatação e erosão.
- HSV: máscara original, abertura, fechamento, dilatação e erosão.
- LAB: máscara original, abertura, fechamento, dilatação e erosão.

Considerando essas cinco versões por espaço de cores, são avaliadas até quinze máscaras para uma mesma imagem.

Cada máscara é submetida à função `avaliar_mascara()`, que retorna as métricas tradicionais e os indicadores da região de incerteza. Os resultados são armazenados em uma estrutura identificada pelo espaço de cores e pela operação aplicada.

A comparação pressupõe que todas as máscaras e o trimap pertençam à mesma imagem. Caso contrário, os resultados não representam uma comparação válida.

## 12. Gráficos comparativos

Foram preparados dois conjuntos de gráficos para facilitar a interpretação dos resultados.

### 12.1. Gráfico das métricas tradicionais

O primeiro conjunto apresenta IoU, precisão, recall, F1-score e acurácia.

Cada gráfico permite comparar as diferentes máscaras em uma métrica específica. Essa separação facilita a leitura dos valores e a identificação das máscaras que apresentam melhor desempenho em cada critério.

Os gráficos utilizam uma escala de 0 a 1, permitindo comparar os resultados das métricas tradicionais.

### 12.2. Gráfico das métricas da região de incerteza

O segundo conjunto apresenta as cinco métricas relacionadas à região de fronteira.

As duas métricas expressas em quantidade de pixels são apresentadas como contagens. As três métricas de proporção são apresentadas em uma escala de 0 a 1.

Essa separação evita que as contagens de pixels tornem as proporções visualmente insignificantes. Os valores numéricos podem ser exibidos sobre as barras para facilitar a comparação.

A quantidade total de pixels incertos permanece constante para todas as máscaras de uma mesma imagem. Por isso, essa métrica serve como referência, mas não diferencia o desempenho dos métodos.

Os gráficos constituem uma ferramenta de análise experimental: sua finalidade é apresentar os resultados calculados, não determinar antecipadamente qual espaço de cores ou operação será superior.

## 13. Análise dos histogramas de cor

Também foi planejada a análise de histogramas para investigar a distribuição dos valores de cor da imagem inteira e das regiões selecionadas pelas máscaras.

O objetivo é comparar os canais dos espaços RGB, HSV e LAB e observar como a distribuição dos valores muda quando são considerados todos os pixels ou apenas os pixels selecionados por uma máscara.

Os histogramas podem ajudar a compreender por que determinados limites de limiarização selecionam regiões específicas. Por exemplo, a distribuição dos canais pode revelar concentrações de valores em determinadas faixas.

No entanto, histogramas não identificam, por si só, a posição espacial dos pixels nem constituem uma assinatura única da imagem. Imagens diferentes podem apresentar distribuições semelhantes. Eles são instrumentos complementares de análise, e não substituem as métricas calculadas com o trimap.

A implementação e a execução efetiva desses gráficos devem ser verificadas no código do projeto para distinguir as visualizações já integradas daquelas que ainda estão em fase de preparação.

## 14. Limitações da abordagem atual

A segmentação por limiarização apresenta limitações importantes.

Primeiramente, os limites são definidos manualmente e podem não funcionar da mesma maneira para imagens com iluminação, fundos ou cores diferentes. Um intervalo que funciona bem para determinado animal pode falhar em outra imagem.

Em segundo lugar, as operações morfológicas alteram a forma da máscara, mas não compreendem o conteúdo semântico da imagem. Elas não conseguem, sozinhas, distinguir uma parte do animal de uma região do fundo quando ambas apresentam características semelhantes.

Além disso, as métricas tradicionais ignoram a classe 3 do trimap. Isso significa que a avaliação atual não mede diretamente a qualidade da segmentação nas regiões de fronteira. Os indicadores complementares descrevem como a máscara ocupa essas regiões, mas não estabelecem se cada pixel incerto foi classificado corretamente.

Por fim, os resultados obtidos em uma única imagem não são suficientes para concluir que determinado método é superior de maneira geral. Uma análise mais robusta exige a avaliação de várias imagens e a comparação consistente dos resultados.

## 15. Integração futura com inteligência artificial

A etapa tradicional estabelece uma referência experimental para a integração posterior de inteligência artificial.

A arquitetura proposta contempla duas camadas:

**IA perceptiva:** um modelo de segmentação pré-treinado identifica o objeto principal e produz uma máscara inicial.

**IA generativa:** interpreta a instrução do usuário, determina quais operações devem ser solicitadas e organiza os parâmetros necessários para a execução.

O processamento digital de imagens permanece responsável pelo refinamento morfológico, pela aplicação do canal alfa e pela composição do resultado final.

O fluxo futuro será:

Usuário envia uma imagem e uma instrução → IA generativa interpreta a solicitação → parâmetros são validados → modelo de segmentação produz a máscara → processamento digital de imagens refina a máscara → canal alfa é aplicado → imagem PNG transparente é gerada.

A validação dos parâmetros é importante para evitar operações incompatíveis ou interpretações incorretas das instruções. A IA generativa deve coordenar as funções da aplicação, sem manipular diretamente os pixels da imagem.

Posteriormente, será possível comparar três situações: máscaras obtidas por limiarização tradicional, máscaras produzidas pelo modelo de segmentação e máscaras geradas pelo modelo de segmentação após refinamento por processamento digital de imagens.

Essa comparação permitirá investigar se a combinação entre segmentação baseada em IA e operações tradicionais melhora os resultados em relação às abordagens isoladas.

## 16. Estado atual do desenvolvimento

Até o momento, o trabalho está estruturado em torno da implementação e análise da abordagem tradicional de remoção de fundo.

As atividades desenvolvidas ou preparadas no código apresentado incluem:

- Organização do processamento em módulos Python.
- Carregamento de imagens coloridas.
- Conversão entre os espaços RGB, HSV e LAB.
- Criação de máscaras por limiarização.
- Geração de imagens com transparência.
- Aplicação de abertura, fechamento, dilatação e erosão.
- Utilização de trimaps do dataset Oxford-IIIT Pet como referência.
- Implementação de métricas tradicionais de segmentação.
- Implementação de indicadores para a região de incerteza.
- Preparação de comparações entre máscaras originais e refinadas.
- Preparação de gráficos para visualizar e comparar as métricas.
- Investigação de histogramas para analisar a distribuição dos canais de cor.

A execução dos experimentos, a coleta dos resultados e a análise comparativa final devem ser registradas conforme forem realizadas. A existência do código de avaliação não implica, por si só, que um espaço de cores ou uma operação morfológica tenha demonstrado superioridade.

A integração do modelo de segmentação e da IA generativa permanece como uma etapa posterior da arquitetura proposta.

## 17. Conclusão

O projeto estabelece uma base experimental para estudar a remoção de fundo em imagens coloridas por meio da combinação de espaços de cores, limiarização, operações morfológicas e avaliação quantitativa.

A utilização do dataset Oxford-IIIT Pet permite comparar as máscaras geradas com anotações de referência. As métricas tradicionais avaliam a correspondência entre a segmentação e as regiões conhecidas de animal e fundo, enquanto os indicadores da região de incerteza ajudam a caracterizar o comportamento das máscaras nas fronteiras.

A visualização das imagens, dos histogramas e dos gráficos comparativos contribui para compreender as vantagens e limitações de cada configuração.

Essa base permite avançar para a integração de inteligência artificial de maneira fundamentada. Ao comparar as técnicas tradicionais com um modelo de segmentação e com o refinamento posterior das máscaras, o projeto poderá avaliar experimentalmente a contribuição de cada componente e desenvolver uma aplicação mais flexível para a remoção de fundo em imagens coloridas.