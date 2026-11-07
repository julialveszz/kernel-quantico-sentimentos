# experimento principal: QSVM x modelos classicos sobre as mesmas features comprimidas
# uso: python 03_rodar_experimentos.py            (tudo)
#      python 03_rodar_experimentos.py --seeds 0,1 --bases yelp   (teste rapido)

import sys, os, json, time, itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
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

def metricas(y_real, y_prev):
    return f1_score(y_real, y_prev, average="macro"), accuracy_score(y_real, y_prev)

def busca_kernel(K_tr, y_tr, grade_C, seed):
    # busca em grade do C com validacao cruzada interna, usando o kernel ja pronto
    internos = StratifiedKFold(config.N_FOLDS_INTERNOS, shuffle=True, random_state=seed)
    melhor_C, melhor = None, -1
    for C in grade_C:
        notas = []
        for a, b in internos.split(K_tr, y_tr):
            m = SVC(kernel="precomputed", C=C).fit(K_tr[np.ix_(a, a)], y_tr[a])
            notas.append(f1_score(y_tr[b], m.predict(K_tr[np.ix_(b, a)]), average="macro"))
        if np.mean(notas) > melhor:
            melhor, melhor_C = np.mean(notas), C
    return melhor_C, melhor

def busca_classica(modelo_base, grade, Z_tr, y_tr, seed):
    internos = StratifiedKFold(config.N_FOLDS_INTERNOS, shuffle=True, random_state=seed)
    melhor_param, melhor = None, -1
    for param in grade:
        notas = []
        for a, b in internos.split(Z_tr, y_tr):
            m = modelo_base(**param).fit(Z_tr[a], y_tr[a])
            notas.append(f1_score(y_tr[b], m.predict(Z_tr[b]), average="macro"))
        if np.mean(notas) > melhor:
            melhor, melhor_param = np.mean(notas), param
    return melhor_param

grade_svm = [{"C": C, "gamma": g} for C in config.GRADE_C for g in config.GRADE_GAMMA]
grade_log = [{"C": C, "max_iter": 2000} for C in config.GRADE_C_LOGISTICA]

ja_feitos = set()
if os.path.exists(saida):
    antigo = pd.read_csv(saida)
    ja_feitos = set(zip(antigo["base"], antigo["seed"]))

for base in bases:
    dados = np.load(f"dados/embeddings_{base}.npz")
    X, y = dados["X"], dados["y"]
    for seed in seeds:
        if (base, seed) in ja_feitos:
            print("pulando", base, seed); continue
        inicio = time.time()
        linhas = []
        externos = StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(externos.split(X, y)):
            gerador = np.random.default_rng(1000 * seed + fold)
            escala = StandardScaler().fit(X[tr])
            Xtr, Xte = escala.transform(X[tr]), escala.transform(X[te])
            for compressao in config.COMPRESSOES:
                for d in config.DIMENSOES:
                    # compressao ajustada so no treino (sem vazamento)
                    if compressao == "pca":
                        red = PCA(n_components=d, random_state=seed).fit(Xtr)
                    else:
                        red = NeighborhoodComponentsAnalysis(n_components=d, init="pca",
                                                             max_iter=50, random_state=seed).fit(Xtr, y[tr])
                    pad = StandardScaler().fit(red.transform(Xtr))
                    Ztr, Zte = pad.transform(red.transform(Xtr)), pad.transform(red.transform(Xte))
                    ytr, yte = y[tr], y[te]
                    base_linha = {"base": base, "seed": seed, "fold": fold, "compressao": compressao, "d": d}

                    # ---- classicos (nao dependem do cenario do kernel) ----
                    p = busca_classica(LogisticRegression, grade_log, Ztr, ytr, seed)
                    f1, acc = metricas(yte, LogisticRegression(**p).fit(Ztr, ytr).predict(Zte))
                    linhas.append({**base_linha, "cenario": "classico", "modelo": "logistica", "f1": f1, "acuracia": acc, "hiper": json.dumps(p)})
                    f1, acc = metricas(yte, SVC(kernel="rbf", **config.SVM_PADRAO).fit(Ztr, ytr).predict(Zte))
                    linhas.append({**base_linha, "cenario": "classico", "modelo": "svm_padrao", "f1": f1, "acuracia": acc, "hiper": json.dumps(config.SVM_PADRAO)})
                    p = busca_classica(lambda **k: SVC(kernel="rbf", **k), grade_svm, Ztr, ytr, seed)
                    f1, acc = metricas(yte, SVC(kernel="rbf", **p).fit(Ztr, ytr).predict(Zte))
                    linhas.append({**base_linha, "cenario": "classico", "modelo": "svm_ajustado", "f1": f1, "acuracia": acc, "hiper": json.dumps(p)})

                    # ---- quanticos ----
                    cenarios = ["exato", "shots"] + (list(config.RUIDOS) if d in config.DIMENSOES_COM_RUIDO else [])
                    Zall = np.vstack([Ztr, Zte])
                    n_tr = len(Ztr)
                    # kernels exatos pra cada largura de banda e cada tipo de circuito
                    kernels = {}
                    for emaranhado in [True, False]:
                        for b in config.GRADE_LARGURA_BANDA:
                            E = quantico.estados(Zall * b, emaranhamento=emaranhado)
                            kernels[(emaranhado, b)] = quantico.kernel_exato(E, E)
                    for cenario in cenarios:
                        # monta o kernel de cada cenario (exato, com shots ou com ruido + shots)
                        K_cen = {}
                        for (emaranhado, b), K in kernels.items():
                            if cenario == "exato":
                                Ktr, Kte = K[:n_tr, :n_tr], K[n_tr:, :n_tr]
                            else:
                                Kr = K if cenario == "shots" else quantico.aplicar_ruido(K, d, config.RUIDOS[cenario], emaranhado)
                                Ktr = quantico.amostrar_shots(Kr[:n_tr, :n_tr], config.SHOTS, gerador, simetrica=True)
                                Kte = quantico.amostrar_shots(Kr[n_tr:, :n_tr], config.SHOTS, gerador)
                            K_cen[(emaranhado, b)] = (Ktr, Kte)
                        variantes = {
                            "qsvm_ajustado": (True, config.GRADE_LARGURA_BANDA),
                            "qsvm_largura_fixa": (True, [config.LARGURA_FIXA]),
                            "qsvm_sem_emaranhamento": (False, config.GRADE_LARGURA_BANDA),
                        }
                        for nome, (emaranhado, larguras) in variantes.items():
                            melhor = (-1, None, None)
                            for b in larguras:
                                C, nota = busca_kernel(K_cen[(emaranhado, b)][0], ytr, config.GRADE_C, seed)
                                if nota > melhor[0]:
                                    melhor = (nota, b, C)
                            _, b, C = melhor
                            Ktr, Kte = K_cen[(emaranhado, b)]
                            m = SVC(kernel="precomputed", C=C).fit(Ktr, ytr)
                            f1, acc = metricas(yte, m.predict(Kte))
                            linhas.append({**base_linha, "cenario": cenario, "modelo": nome, "f1": f1, "acuracia": acc,
                                           "hiper": json.dumps({"C": C, "largura_banda": b})})
        df = pd.DataFrame(linhas)
        df.to_csv(saida, mode="a", header=not os.path.exists(saida), index=False)
        print(f"{base} seed {seed} pronta em {time.time() - inicio:.0f}s", flush=True)
