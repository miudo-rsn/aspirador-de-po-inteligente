Descrição do Projeto
Este projeto utiliza um perceptron simples para controlar um aspirador de pó inteligente. O modelo recebe como entrada:
- Tipo de piso (carpete, cerâmica, madeira)
- Quantidade de poeira (0 a 9)
- Distância de obstáculos (0 a 5 metros)

E retorna como saída:
- Potência de aspiração (1 a 3)
- Velocidade de movimento (1 a 5)

Como Executar
1. Instale o Python 3 e as bibliotecas numpy e matplotlib.
2. Execute o arquivo .py no terminal ou em um ambiente como Jupyter Notebook.
3. O programa treina o perceptron e mostra a previsão para uma entrada de teste.
4. Um gráfico será exibido mostrando a evolução do erro durante o treinamento.

Justificativa da Função de Ativação
Foi utilizada a função sigmoide, pois ela permite prever saídas contínuas entre 0 e 1, que são depois convertidas para os valores reais de potência e velocidade. A função degrau não seria adequada para esse tipo de saída.

Análise do Treinamento
O gráfico gerado mostra que o erro total diminui ao longo das épocas, indicando que o modelo está aprendendo corretamente com os dados fornecidos.

Equipe
- Nome: Fabricio Rocha RA:125111399975
- Nome: Maycon Soares RA:125111404445
- Nome: Matheus RA:125111402366

