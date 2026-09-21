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
