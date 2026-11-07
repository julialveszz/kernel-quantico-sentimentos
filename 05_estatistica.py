# testes estatisticos
# as 10 seeds reaproveitam as mesmas 1000 frases, entao os resultados nao sao independentes.
# por isso o teste usado e o t corrigido para validacao cruzada repetida (Nadeau e Bengio, 2003),
# aplicado sobre as 50 diferencas pareadas (10 seeds x 5 folds, mesmas particoes para os dois modelos):
#   t = media(d) / sqrt((1/k + n_teste/n_treino) * var(d)),  com k = 50 e 49 graus de liberdade
# o IC de 95% usa a mesma variancia corrigida. Holm e aplicado dentro de cada hipotese e cenario,
# juntando as configuracoes das duas bases (12 no cenario exato e com shots, 8 nos cenarios com ruido).
# continua sendo uma aproximacao: a correcao reduz, mas nao elimina, o efeito da dependencia entre folds.
import sys, json
import numpy as np
import pandas as pd
from scipy import stats
import config

arquivo = sys.argv[1] if len(sys.argv) > 1 else "resultados/resultados_por_fold.csv"
df = pd.read_csv(arquivo)
CLASSICOS = {"logistica", "svm_padrao", "svm_ajustado"}
RAZAO_TESTE_TREINO = (1 / config.N_FOLDS) / (1 - 1 / config.N_FOLDS)   # 200/800 = 0,25

def folds(base, comp, d, cenario, modelo):
    cen = "classico" if modelo in CLASSICOS else cenario
    s = df[(df.base == base) & (df.compressao == comp) & (df.d == d) & (df.cenario == cen) & (df.modelo == modelo)]
    if s.empty:
        return None
    return s.sort_values(["seed", "fold"])["f1"].values

def t_corrigido(dif):
    k = len(dif)
    media = float(dif.mean())
    var = float(dif.var(ddof=1))
    erro = np.sqrt((1 / k + RAZAO_TESTE_TREINO) * var)
    if erro == 0:
        return media, (media, media), 0.0, 1.0
    t = media / erro
    p = float(2 * stats.t.sf(abs(t), k - 1))
    q = stats.t.ppf(0.975, k - 1)
    return media, (media - q * erro, media + q * erro), float(t), p

def holm(pvalores):
    pvalores = np.asarray(pvalores, dtype=float)
    ordem = np.argsort(pvalores)
    m = len(pvalores)
    ajustado = np.empty(m)
    maior = 0.0
    for k, i in enumerate(ordem):
        maior = max(maior, (m - k) * pvalores[i])
        ajustado[i] = min(1.0, maior)
    return ajustado

COMPARACOES = {
    "H1_qsvm_vs_svm_ajustado": ("qsvm_ajustado", "svm_ajustado"),
    "H2_svm_ajustado_vs_padrao": ("svm_ajustado", "svm_padrao"),
    "H2b_qsvm_vs_svm_padrao": ("qsvm_ajustado", "svm_padrao"),
    "H5_qsvm_vs_sem_emaranhamento": ("qsvm_ajustado", "qsvm_sem_emaranhamento"),
    "H4b_largura_ajustada_vs_fixa": ("qsvm_ajustado", "qsvm_largura_fixa"),
    "extra_qsvm_vs_logistica": ("qsvm_ajustado", "logistica"),
}

linhas = []
cenarios = [c for c in ["exato", "shots"] + list(config.RUIDOS) if c in df["cenario"].unique()]
for cenario in cenarios:
    for nome, (m1, m2) in COMPARACOES.items():
        grupo = []
        for base in ["yelp", "repro"]:
            for comp in config.COMPRESSOES:
                for d in config.DIMENSOES:
                    a, b = folds(base, comp, d, cenario, m1), folds(base, comp, d, cenario, m2)
                    if a is None or b is None:
                        continue
                    media, ic, t, p = t_corrigido(a - b)
                    grupo.append({"base": base, "cenario": cenario, "comparacao": nome, "compressao": comp, "d": d,
                                  "media_1": float(a.mean()), "media_2": float(b.mean()), "diferenca": media,
                                  "ic_inf": ic[0], "ic_sup": ic[1], "t": t, "p": p, "n_pares": len(a)})
        for g, pa in zip(grupo, holm([g["p"] for g in grupo])):
            g["p_holm"] = float(pa)
        linhas += grupo
    # H3: a diferenca QSVM - SVM ajustado muda entre PCA e NCA? (pareado por seed e fold, com sinal)
    grupo = []
    for base in ["yelp", "repro"]:
        for d in config.DIMENSOES:
            s = {c: (folds(base, c, d, cenario, "qsvm_ajustado"), folds(base, c, d, cenario, "svm_ajustado")) for c in config.COMPRESSOES}
            if any(v[0] is None or v[1] is None for v in s.values()):
                continue
            dif_pca = s["pca"][0] - s["pca"][1]
            dif_nca = s["nca"][0] - s["nca"][1]
            media, ic, t, p = t_corrigido(dif_pca - dif_nca)
            grupo.append({"base": base, "cenario": cenario, "comparacao": "H3_dif_pca_vs_nca", "compressao": "pca-nca", "d": d,
                          "media_1": float(dif_pca.mean()), "media_2": float(dif_nca.mean()), "diferenca": media,
                          "ic_inf": ic[0], "ic_sup": ic[1], "t": t, "p": p, "n_pares": len(dif_pca)})
    for g, pa in zip(grupo, holm([g["p"] for g in grupo])):
        g["p_holm"] = float(pa)
    linhas += grupo

testes = pd.DataFrame(linhas)
testes.to_csv("resultados/testes_estatisticos.csv", index=False)

# medias e desvios por configuracao (entre as 10 medias por seed, so pra descrever a variacao)
por_seed = df.groupby(["base", "compressao", "d", "cenario", "modelo", "seed"])["f1"].mean()
resumo = por_seed.groupby(["base", "compressao", "d", "cenario", "modelo"]).agg(["mean", "std"]).reset_index()
resumo.columns = ["base", "compressao", "d", "cenario", "modelo", "f1_media", "f1_dp"]
resumo.to_csv("resultados/resumo_f1.csv", index=False)

# hiperparametros escolhidos (estabilidade do ajuste)
q = df[df["modelo"] == "qsvm_ajustado"].copy()
q["largura"] = q["hiper"].apply(lambda h: json.loads(h)["largura_banda"])
q.groupby(["base", "cenario", "d", "largura"]).size().rename("vezes").reset_index().to_csv("resultados/frequencia_largura_banda.csv", index=False)
print(testes.groupby(["cenario", "comparacao"]).apply(lambda g: int((g.p_holm < config.ALFA).sum())).rename("significativos").to_string())
