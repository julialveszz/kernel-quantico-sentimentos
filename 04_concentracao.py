# analise da concentracao do kernel (H4): variancia dos valores fora da diagonal
# e alinhamento kernel-alvo, variando o numero de qubits de 2 a 16
import json, time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import config, quantico

linhas = []
for base in ["yelp", "repro"]:
    dados = np.load(f"dados/embeddings_{base}.npz")
    X, y = dados["X"], dados["y"]
    gerador = np.random.default_rng(0)
    idx = gerador.choice(len(X), size=config.N_PONTOS_CONCENTRACAO, replace=False)
    Xs = StandardScaler().fit_transform(X)
    alvo = np.where(y[idx] == 1, 1.0, -1.0)
    Y = np.outer(alvo, alvo)
    for n in config.QUBITS_CONCENTRACAO:
        Z = StandardScaler().fit_transform(PCA(n_components=n, random_state=0).fit_transform(Xs))[idx]
        for emaranhado in [True, False]:
            for b in config.LARGURAS_CONCENTRACAO:
                inicio = time.time()
                E = quantico.estados(Z * b, emaranhamento=emaranhado)
                K = quantico.kernel_exato(E, E)
                fora = K[np.triu_indices(len(K), 1)]
                alinhamento = float(np.sum(K * Y) / (np.linalg.norm(K) * np.linalg.norm(Y)))
                linhas.append({"base": base, "qubits": n, "emaranhamento": emaranhado, "largura_banda": b,
                               "media": float(fora.mean()), "variancia": float(fora.var()),
                               "alinhamento": alinhamento})
                del E, K
        print(base, n, "ok", flush=True)

json.dump(linhas, open("resultados/concentracao.json", "w"), indent=1)
