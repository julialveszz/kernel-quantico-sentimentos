# simulacao do kernel quantico com o ZZ feature map (Havlicek et al., 2019)
# o estado e |phi(x)> = U(x) H U(x) H |0>, com 2 repeticoes e emaranhamento entre todos os pares
# U(x) aplica fase x_i em cada qubit e (pi - x_i)(pi - x_j) em cada par (termos ZZ)
# na versao sem emaranhamento os termos ZZ saem e o estado vira um produto de qubits independentes

import numpy as np
from itertools import combinations

_cache_sinais = {}

def sinais(n):
    # matriz (2^n x n) com +1/-1 pra cada qubit em cada estado da base, e o produto pra cada par
    if n not in _cache_sinais:
        indices = np.arange(2 ** n)
        s = 1 - 2 * ((indices[:, None] >> np.arange(n)) & 1)
        pares = list(combinations(range(n), 2))
        s_pares = np.stack([s[:, i] * s[:, j] for i, j in pares], axis=1) if pares else np.zeros((2 ** n, 0))
        _cache_sinais[n] = (s.astype(np.float64), s_pares.astype(np.float64), pares)
    return _cache_sinais[n]

def hadamard_em_todos(psi):
    # transformada de Walsh-Hadamard rapida no ultimo eixo, normalizada
    psi = psi.copy()
    n_linhas, tamanho = psi.shape
    passo = 1
    while passo < tamanho:
        psi = psi.reshape(n_linhas, tamanho // (2 * passo), 2, passo)
        a = psi[:, :, 0, :].copy()
        b = psi[:, :, 1, :].copy()
        psi[:, :, 0, :] = a + b
        psi[:, :, 1, :] = a - b
        psi = psi.reshape(n_linhas, tamanho)
        passo *= 2
    return psi / np.sqrt(tamanho)

_cache_hadamard = {}

def _hadamard_matriz(k):
    if k not in _cache_hadamard:
        H = np.array([[1.0]])
        for _ in range(k):
            H = np.kron(H, np.array([[1.0, 1.0], [1.0, -1.0]]))
        _cache_hadamard[k] = (H / np.sqrt(2 ** k)).astype(np.complex128)
    return _cache_hadamard[k]

def _hadamard_rapido(psi, n):
    # H em todos os qubits = H^{n1} x H^{n2}, aplicado com duas multiplicacoes de matriz
    n1 = n // 2
    n2 = n - n1
    H1, H2 = _hadamard_matriz(n1), _hadamard_matriz(n2)
    X = psi.reshape(len(psi), 2 ** n1, 2 ** n2)
    X = np.matmul(H1, X)
    X = np.matmul(X, H2.T)
    return X.reshape(len(psi), 2 ** n)

def estados(angulos, emaranhamento=True):
    # angulos: matriz (N x n) ja multiplicada pela largura de banda
    angulos = np.asarray(angulos, dtype=np.float64)
    N, n = angulos.shape
    s, _, _ = sinais(n)
    fase = angulos @ s.T                                       # parte de 1 qubit
    if emaranhamento and n > 1:
        # soma dos pares (pi - x_i)(pi - x_j) s_i s_j = ((pi - x).s)^2 / 2 - |pi - x|^2 / 2, ja que s_i^2 = 1
        c = np.pi - angulos
        fase = fase + 0.5 * ((c @ s.T) ** 2 - np.sum(c ** 2, axis=1, keepdims=True))
    # complex128 porque, com larguras de banda muito pequenas, os valores do kernel ficam muito perto de 1
    giro = np.exp(1j * fase)
    psi = giro / np.sqrt(2 ** n)                               # U(x) H|0>
    psi = _hadamard_rapido(psi, n)                             # H
    return giro * psi                                          # U(x)

def kernel_exato(estados_a, estados_b):
    sobreposicao = estados_a.conj() @ estados_b.T
    return np.clip(np.abs(sobreposicao) ** 2, 0.0, 1.0).astype(np.float64)

def n_cnots(n, emaranhamento=True):
    # circuito compute-uncompute U(x) U(y)^dagger: cada termo ZZ usa 2 CNOTs,
    # sao n(n-1)/2 pares, 2 repeticoes, e o circuito tem U e U^dagger
    if not emaranhamento:
        return 0
    return 4 * n * (n - 1)

def aplicar_ruido(K, n, erro_por_cnot, emaranhamento=True):
    # despolarizacao global: K_ruido = F K + (1 - F) / 2^n, com F = (1 - p)^(numero de CNOTs)
    F = (1 - erro_por_cnot) ** n_cnots(n, emaranhamento)
    return F * K + (1 - F) / (2 ** n)

def amostrar_shots(K, shots, gerador, simetrica=False):
    # estima cada entrada contando quantas vezes saiu |0...0> em 'shots' medicoes
    if simetrica:
        # a diagonal tambem e medida, entao com ruido ela fica abaixo de 1, igual ao modelo
        K_est = np.diag(gerador.binomial(shots, np.clip(np.diag(K), 0, 1)) / shots)
        i, j = np.triu_indices(K.shape[0], 1)
        valores = gerador.binomial(shots, np.clip(K[i, j], 0, 1)) / shots
        K_est[i, j] = valores
        K_est[j, i] = valores
        return K_est
    return gerador.binomial(shots, np.clip(K, 0, 1)) / shots

def kernel_separavel(angulos_a, angulos_b):
    # sem os termos ZZ o estado e um produto de qubits independentes, entao o kernel
    # e o produto das sobreposicoes de cada qubit. da o mesmo valor que estados() + kernel_exato(),
    # mas sem montar o vetor de 2^n posicoes (bem mais rapido com 8 e 12 qubits)
    a = np.asarray(angulos_a, dtype=np.float64)
    b = np.asarray(angulos_b, dtype=np.float64)
    def qubit(x):
        # U(x) H U(x) H |0> num qubit so: [cos(x) e^{ix}, i sen(x) e^{-ix}]
        return np.stack([np.cos(x) * np.exp(1j * x), 1j * np.sin(x) * np.exp(-1j * x)], axis=-1)
    Qa, Qb = qubit(a), qubit(b)
    produto = np.ones((len(a), len(b)), dtype=np.complex128)
    for i in range(a.shape[1]):
        produto *= Qa[:, i, :].conj() @ Qb[:, i, :].T
    return np.clip(np.abs(produto) ** 2, 0.0, 1.0)
