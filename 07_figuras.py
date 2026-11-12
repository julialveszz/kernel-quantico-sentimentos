# figuras do artigo (rodada 3)
# escolhas: fonte serifada igual a do artigo, 8,5 pt, sem grade, virgula decimal, tons de cinza,
# largura de uma coluna (3,5 pol) e exportacao vetorial em PDF
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "lines.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})

def virgula(casas):
    return FuncFormatter(lambda v, _: f"{v:.{casas}f}".replace(".", ","))

# figura 1: tamanho das frases nas duas bases
y = pd.read_csv("dados/yelp_preparada.csv")
r = pd.read_csv("dados/repro_preparada.csv")
fig, ax = plt.subplots(figsize=(3.5, 1.7))
faixas = np.arange(0.5, 33.5, 1)
ax.hist(r.palavras, bins=faixas, histtype="stepfilled", color="0.8", label="RePro")
ax.hist(y.palavras, bins=faixas, histtype="step", color="black", lw=0.9, label="Yelp")
ax.set_xlabel("número de palavras"); ax.set_ylabel("frases")
ax.legend(frameon=False, loc="upper right")
fig.savefig("figuras/fig_tamanhos.pdf")

# figura 2: F1-macro do QSVM e do SVM-RBF ajustados, com PCA
resumo = pd.read_csv("resultados/resumo_f1.csv")
def media(base, d, modelo):
    cen = "classico" if modelo.startswith("svm") else "exato"
    l = resumo[(resumo.base == base) & (resumo.compressao == "pca") & (resumo.d == d) & (resumo.cenario == cen) & (resumo.modelo == modelo)]
    return l.f1_media.iloc[0]
casos = [(b, d) for b in ["yelp", "repro"] for d in [4, 8, 12]]
rotulos = [f"{'Yelp' if b == 'yelp' else 'RePro'}, {d} qubits" for b, d in casos]
pos = np.arange(len(casos))[::-1].astype(float)
pos[3:] -= 0.6
fig, ax = plt.subplots(figsize=(3.5, 1.9))
for k, (p, (b, d)) in enumerate(zip(pos, casos)):
    s, q = media(b, d, "svm_ajustado"), media(b, d, "qsvm_ajustado")
    ax.plot([s, q], [p, p], color="0.7", lw=1.6, zorder=1)
    ax.plot(s, p, "o", ms=4.4, mfc="white", mec="black", mew=0.8, zorder=2, label="SVM-RBF ajustado" if k == 0 else None)
    ax.plot(q, p, "o", ms=4.4, color="black", zorder=3, label="QSVM ajustado" if k == 0 else None)
valores = [media(b, d, m) for b, d in casos for m in ["svm_ajustado", "qsvm_ajustado"]]
ax.set_xlim(np.floor(min(valores) * 100) / 100 - 0.005, np.ceil(max(valores) * 100) / 100 + 0.005)
ax.set_yticks(pos); ax.set_yticklabels(rotulos); ax.tick_params(axis="y", length=0)
ax.xaxis.set_major_formatter(virgula(2)); ax.set_xlabel("F1-macro")
ax.legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2)
fig.savefig("figuras/fig_principal.pdf")

# figura 3: variancia do kernel com e sem emaranhamento (RePro)
conc = pd.DataFrame(json.load(open("resultados/concentracao.json")))
fig, eixos = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
for ax, emar, titulo in zip(eixos, [True, False], ["com emaranhamento", "sem emaranhamento"]):
    for b, cor, marca in [(0.01, "0.65", "s"), (0.1, "0.4", "^"), (1.0, "black", "o")]:
        s = conc[(conc.base == "repro") & (conc.largura_banda == b) & (conc.emaranhamento == emar)].sort_values("qubits")
        ax.semilogy(s.qubits, s.variancia, color=cor, marker=marca, ms=2.6, label=f"$b$ = {str(b).replace('.', ',')}")
    ax.set_title(titulo, fontsize=8.5); ax.set_xticks([2, 6, 10, 14]); ax.set_xlabel("qubits")
eixos[0].set_ylabel("variância do kernel")
handles, labels = eixos[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.55, 0.98), ncol=3)
fig.tight_layout()
fig.savefig("figuras/fig_concentracao.pdf")
print("figuras geradas")
