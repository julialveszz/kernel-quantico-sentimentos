# parametros do experimento, tudo fica aqui pra nao espalhar numero magico pelo codigo

# dados
TAMANHO_POR_CLASSE = 500          # igual a Yelp: 500 positivas e 500 negativas
SEED_AMOSTRA_REPRO = 0            # seed fixa so pra sortear a amostra da RePro
PERCENTIS_TAMANHO = (5, 95)       # faixa de tamanho (em palavras) das frases da Yelp usada pra filtrar a RePro

# embeddings
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# compressao e qubits
COMPRESSOES = ["pca", "nca"]
DIMENSOES = [4, 8, 12]

# validacao
SEEDS = list(range(10))
N_FOLDS = 5
N_FOLDS_INTERNOS = 3              # busca em grade dentro de cada fold de treino

# grades de hiperparametros
GRADE_C = [0.1, 1, 10, 100]
GRADE_GAMMA = [0.001, 0.01, 0.1, 1, 10]
GRADE_C_LOGISTICA = [0.01, 0.1, 1, 10, 100]
# rodada 2: na rodada 1 a largura escolhida foi quase sempre 0,05, o menor valor da grade,
# entao a grade foi estendida pra baixo pra ver se o melhor valor estava fora dela
GRADE_LARGURA_BANDA = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 1.0]
LARGURA_FIXA = 1.0                # versao do QSVM sem ajuste de largura de banda

# SVM padrao, igual ao usado em Cajas Ordonez et al.
SVM_PADRAO = {"C": 1.0, "gamma": "scale"}

# cenarios de calculo do kernel
SHOTS = 1000
# erro por porta de dois qubits no modelo de despolarizacao global
RUIDOS = {"ruido_baixo": 0.001, "ruido_alto": 0.01}
DIMENSOES_COM_RUIDO = [4, 8]

# analise de concentracao (H4)
QUBITS_CONCENTRACAO = list(range(2, 17))
LARGURAS_CONCENTRACAO = [0.01, 0.05, 0.1, 0.5, 1.0]
N_PONTOS_CONCENTRACAO = 300

# estatistica
ALFA = 0.05
N_BOOTSTRAP = 10000
