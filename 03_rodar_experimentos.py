# experimento principal: QSVM x modelos classicos sobre as mesmas features comprimidas
# rodada 3: busca de hiperparametros totalmente aninhada. a padronizacao e a compressao (PCA ou NCA)
# sao refeitas dentro de cada fold interno, entao a validacao interna nunca ve uma projecao
# ajustada com ela mesma.
# uso: python 03_rodar_experimentos.py
#      python 03_rodar_experimentos.py --seeds 0,1 --bases yelp --saida resultados/parte1.csv

import sys, os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
import warnings
warnings.filterwarnings("ignore")
import config, quantico

def argumento(nome, padrao):
    if nome in sys.argv:
        return sys.argv[sys.argv.index(nome) + 1]
    return padrao

seeds = [int(s) for s in argumento("--seeds", ",".join(map(str, config.SEEDS))).split(",")]
bases = argumento("--bases", "yelp,repro").split(",")
saida = argumento("--saida", "resultados/resultados_por_fold.csv")
dimensoes = [int(x) for x in argumento("--dimensoes", ",".join(map(str, config.DIMENSOES))).split(",")]

grade_svm = [{"C": C, "gamma": g} for C in config.GRADE_C for g in config.GRADE_GAMMA]
grade_log = [{"C": C, "max_iter": 2000} for C in config.GRADE_C_LOGISTICA]

def f1(y_real, y_prev):
    return f1_score(y_real, y_prev, average="macro")

def comprimir(Xa, ya, Xb, compressao, d, seed):
    # tudo ajustado so em Xa e depois aplicado em Xb
    escala = StandardScaler().fit(Xa)
    A, B = escala.transform(Xa), escala.transform(Xb)
    if compressao == "pca":
        red = PCA(n_components=d, random_state=seed).fit(A)
    else:
        red = NeighborhoodComponentsAnalysis(n_components=d, init="pca", max_iter=config.NCA_MAX_ITER,
                                             tol=config.NCA_TOL, random_state=seed).fit(A, ya)
    pad = StandardScaler().fit(red.transform(A))
    return pad.transform(red.transform(A)), pad.transform(red.transform(B))

def kernels_exatos(Za, Zb, emaranhado, b):
    if not emaranhado:
        # sem emaranhamento da pra calcular qubit a qubit, mesmo resultado e bem mais rapido
        return quantico.kernel_separavel(Za * b, Za * b), quantico.kernel_separavel(Zb * b, Za * b)
    Ea = quantico.estados(Za * b, emaranhamento=emaranhado)
    Eb = quantico.estados(Zb * b, emaranhamento=emaranhado)
    return quantico.kernel_exato(Ea, Ea), quantico.kernel_exato(Eb, Ea)

def kernel_do_cenario(K_aa, K_ba, d, emaranhado, cenario, gerador):
    if cenario == "exato":
        return K_aa, K_ba
    if cenario != "shots":
        K_aa = quantico.aplicar_ruido(K_aa, d, config.RUIDOS[cenario], emaranhado)
        K_ba = quantico.aplicar_ruido(K_ba, d, config.RUIDOS[cenario], emaranhado)
    return (quantico.amostrar_shots(K_aa, config.SHOTS, gerador, simetrica=True),
            quantico.amostrar_shots(K_ba, config.SHOTS, gerador))

def alinhamento_centrado(K1, K2):
    n = len(K1)
    H = np.eye(n) - np.ones((n, n)) / n
    A, B = H @ K1 @ H, H @ K2 @ H
    return float(np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B)))

def melhor(notas, filtro):
    candidatos = {k: np.mean(v) for k, v in notas.items() if filtro(k)}
    return max(candidatos, key=candidatos.get)

ja_feitos = set()
if os.path.exists(saida):
    antigo = pd.read_csv(saida)
    ja_feitos = set(zip(antigo["base"], antigo["seed"]))

for base in bases:
    dados = np.load(f"dados/embeddings_{base}.npz")
    X, y = dados["X"], dados["y"]
    for seed in seeds:
        if (base, seed) in ja_feitos:
            print("pulando", base, seed, flush=True)
            continue
        inicio = time.time()
        linhas = []
        externos = StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(externos.split(X, y)):
            gerador = np.random.default_rng(1000 * seed + fold)
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            for compressao in config.COMPRESSOES:
                for d in dimensoes:
                    cenarios = ["exato", "shots"] + (list(config.RUIDOS) if d in config.DIMENSOES_COM_RUIDO else [])
                    notas = {}

                    # ---- busca interna: compressao refeita em cada fold interno ----
                    internos = StratifiedKFold(config.N_FOLDS_INTERNOS, shuffle=True, random_state=seed)
                    for a, v in internos.split(Xtr, ytr):
                        Za, Zv = comprimir(Xtr[a], ytr[a], Xtr[v], compressao, d, seed)
                        for p in grade_log:
                            m = LogisticRegression(**p).fit(Za, ytr[a])
                            notas.setdefault(("logistica", p["C"]), []).append(f1(ytr[v], m.predict(Zv)))
                        for p in grade_svm:
                            m = SVC(kernel="rbf", **p).fit(Za, ytr[a])
                            notas.setdefault(("svm", p["C"], p["gamma"]), []).append(f1(ytr[v], m.predict(Zv)))
                        for emaranhado in [True, False]:
                            for b in config.GRADE_LARGURA_BANDA:
                                K_aa, K_va = kernels_exatos(Za, Zv, emaranhado, b)
                                for cenario in cenarios:
                                    Kc_aa, Kc_va = kernel_do_cenario(K_aa, K_va, d, emaranhado, cenario, gerador)
                                    for C in config.GRADE_C:
                                        m = SVC(kernel="precomputed", C=C).fit(Kc_aa, ytr[a])
                                        notas.setdefault(("q", cenario, emaranhado, b, C), []).append(f1(ytr[v], m.predict(Kc_va)))

                    # ---- treino final no fold externo inteiro ----
                    Ztr, Zte = comprimir(Xtr, ytr, Xte, compressao, d, seed)
                    base_linha = {"base": base, "seed": seed, "fold": fold, "compressao": compressao, "d": d}

                    def registrar(cenario, modelo, prev_teste, prev_treino, hiper, alinhamento=np.nan):
                        linhas.append({**base_linha, "cenario": cenario, "modelo": modelo,
                                       "f1": f1(yte, prev_teste), "acuracia": accuracy_score(yte, prev_teste),
                                       "f1_treino": f1(ytr, prev_treino), "hiper": json.dumps(hiper),
                                       "alinhamento_rbf": alinhamento})

                    k = melhor(notas, lambda k: k[0] == "logistica")
                    m = LogisticRegression(C=k[1], max_iter=2000).fit(Ztr, ytr)
                    registrar("classico", "logistica", m.predict(Zte), m.predict(Ztr), {"C": k[1]})
                    m = SVC(kernel="rbf", **config.SVM_PADRAO).fit(Ztr, ytr)
                    registrar("classico", "svm_padrao", m.predict(Zte), m.predict(Ztr), config.SVM_PADRAO)
                    k = melhor(notas, lambda k: k[0] == "svm")
                    gamma_svm = k[2]
                    m = SVC(kernel="rbf", C=k[1], gamma=gamma_svm).fit(Ztr, ytr)
                    registrar("classico", "svm_ajustado", m.predict(Zte), m.predict(Ztr), {"C": k[1], "gamma": gamma_svm})
                    K_rbf = rbf_kernel(Ztr, gamma=gamma_svm)

                    escolhas = {}
                    for cenario in cenarios:
                        escolhas[(cenario, "qsvm_ajustado")] = melhor(notas, lambda k: k[0] == "q" and k[1] == cenario and k[2])
                        escolhas[(cenario, "qsvm_largura_fixa")] = melhor(notas, lambda k: k[0] == "q" and k[1] == cenario and k[2] and k[3] == config.LARGURA_FIXA)
                        escolhas[(cenario, "qsvm_sem_emaranhamento")] = melhor(notas, lambda k: k[0] == "q" and k[1] == cenario and not k[2])
                    precisa = {(k[2], k[3]) for k in escolhas.values()}
                    exatos = {(emaranhado, b): kernels_exatos(Ztr, Zte, emaranhado, b) for emaranhado, b in precisa}
                    for cenario in cenarios:
                        for modelo in ["qsvm_ajustado", "qsvm_largura_fixa", "qsvm_sem_emaranhamento"]:
                            _, _, emaranhado, b, C = escolhas[(cenario, modelo)]
                            K_tr, K_te = exatos[(emaranhado, b)]
                            alinhamento = alinhamento_centrado(K_tr, K_rbf) if cenario == "exato" else np.nan
                            Kc_tr, Kc_te = kernel_do_cenario(K_tr, K_te, d, emaranhado, cenario, gerador)
                            m = SVC(kernel="precomputed", C=C).fit(Kc_tr, ytr)
                            registrar(cenario, modelo, m.predict(Kc_te), m.predict(Kc_tr),
                                      {"C": C, "largura_banda": b}, alinhamento)
        df = pd.DataFrame(linhas)
        df.to_csv(saida, mode="a", header=not os.path.exists(saida), index=False)
        print(f"{base} seed {seed} pronta em {time.time() - inicio:.0f}s", flush=True)
