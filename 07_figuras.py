# figuras do artigo
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight"})
NOMES_BASE = {"yelp": "Yelp (inglês)", "repro": "RePro (português)"}
ESTILO = {"logistica": ("Regressão logística", "#7f7f7f", "s"),
          "svm_padrao": ("SVM-RBF padrão", "#ff7f0e", "v"),
          "svm_ajustado": ("SVM-RBF ajustado", "#d62728", "^"),
          "qsvm_sem_emaranhamento": ("QSVM sem emaranhamento", "#9467bd", "D"),
          "qsvm_largura_fixa": ("QSVM largura fixa", "#8c564b", "x"),
          "qsvm_ajustado": ("QSVM ajustado", "#1f77b4", "o")}
resumo = pd.read_csv("resultados/resumo_f1.csv")

def pegar(base, comp, cen, modelo):
    cen = "classico" if modelo in ("logistica", "svm_padrao", "svm_ajustado") else cen
    s = resumo[(resumo.base == base) & (resumo.compressao == comp) & (resumo.cenario == cen) & (resumo.modelo == modelo)].sort_values("d")
    return s.d.values, s.f1_media.values, s.f1_dp.values

# figura 1: F1 x dimensao, cenario exato
fig, eixos = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
for i, base in enumerate(["yelp", "repro"]):
    for j, comp in enumerate(config.COMPRESSOES):
        ax = eixos[i, j]
        for modelo, (nome, cor, marca) in ESTILO.items():
            if modelo == "qsvm_largura_fixa":
                continue   # fica perto de 0,50 e achataria o eixo, vai so na tabela
            d, m, s = pegar(base, comp, "exato", modelo)
            if len(d):
                ax.errorbar(d + (list(ESTILO).index(modelo) - 2.5) * 0.12, m, yerr=s, label=nome, color=cor, marker=marca, ms=3.5, lw=1, capsize=2)
        ax.set_title(f"{NOMES_BASE[base]}, {comp.upper()}")
        ax.set_xticks(config.DIMENSOES)
        ax.set_ylim(0.86, 0.97)
        if i == 1: ax.set_xlabel("dimensões = qubits")
        if j == 0: ax.set_ylabel("F1-macro")
eixos[0, 0].legend(fontsize=6, ncol=2, loc="lower left")
fig.savefig("figuras/fig_f1_dimensoes.pdf")

# figura 2: concentracao do kernel
try:
    conc = pd.DataFrame(json.load(open("resultados/concentracao.json")))
    fig, eixos = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, base in zip(eixos, ["yelp", "repro"]):
        cores = ["#9467bd", "#ff7f0e", "#2ca02c", "#1f77b4", "#d62728", "#8c564b", "#7f7f7f"]
        for b, cor in zip(config.LARGURAS_CONCENTRACAO, cores):
            for emaranhado, estilo in [(True, "-"), (False, ":")]:
                s = conc[(conc.base == base) & (conc.largura_banda == b) & (conc.emaranhamento == emaranhado)].sort_values("qubits")
                if len(s):
                    ax.semilogy(s.qubits, s.variancia, estilo, color=cor, marker="o" if emaranhado else None, ms=2.5,
                                label=f"b = {b}" + ("" if emaranhado else " (sem emar.)"))
        ax.set_title(NOMES_BASE[base]); ax.set_xlabel("qubits")
        ax.set_xticks(range(2, 17, 2))
    eixos[0].set_ylabel("variância do kernel fora da diagonal")
    eixos[1].legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig("figuras/fig_concentracao.pdf")
except FileNotFoundError:
    print("sem concentracao.json ainda")

# figura 3: cenarios do kernel para o QSVM ajustado
cens = [c for c in ["exato", "shots"] + list(config.RUIDOS) if c in resumo.cenario.unique()]
nomes = {"exato": "exato", "shots": "1.000 shots", "ruido_baixo": "ruído p=0,001", "ruido_alto": "ruído p=0,01"}
fig, eixos = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
largura = 0.8 / len(cens)
for ax, base in zip(eixos, ["yelp", "repro"]):
    rotulos = [f"{c.upper()} d={d}" for c in config.COMPRESSOES for d in config.DIMENSOES_COM_RUIDO]
    for k, cen in enumerate(cens):
        medias, dps = [], []
        for comp in config.COMPRESSOES:
            for d in config.DIMENSOES_COM_RUIDO:
                l = resumo[(resumo.base == base) & (resumo.compressao == comp) & (resumo.d == d) & (resumo.cenario == cen) & (resumo.modelo == "qsvm_ajustado")]
                medias.append(l.f1_media.iloc[0] if len(l) else np.nan); dps.append(l.f1_dp.iloc[0] if len(l) else 0)
        ax.bar(np.arange(len(rotulos)) + k * largura, medias, largura, yerr=dps, label=nomes[cen], capsize=2)
    ax.set_xticks(np.arange(len(rotulos)) + largura * (len(cens) - 1) / 2); ax.set_xticklabels(rotulos, fontsize=7)
    ax.set_title(NOMES_BASE[base])
eixos[0].set_ylabel("F1-macro (QSVM ajustado)"); eixos[0].legend(fontsize=6, loc="lower left")
for ax in eixos: ax.set_ylim(0.80, 0.97)
fig.savefig("figuras/fig_cenarios.pdf")

# figura 4: tamanho das frases nas duas bases
y = pd.read_csv("dados/yelp_preparada.csv"); r = pd.read_csv("dados/repro_preparada.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.2))
bins = np.arange(0, 34, 1)
ax.hist(y.palavras, bins=bins, alpha=0.6, label="Yelp", density=True)
ax.hist(r.palavras, bins=bins, alpha=0.6, label="RePro (selecionada)", density=True)
ax.set_xlabel("número de palavras"); ax.set_ylabel("densidade"); ax.legend(fontsize=6)
fig.savefig("figuras/fig_tamanhos.pdf")

# figura 5: fluxo do experimento (caixas medidas pelo texto, setas entre as bordas)
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
etapas = ["Frases\nYelp e RePro", "Embeddings\nMiniLM multilíngue\n(384 dim.)", "Padronização\n+ PCA ou NCA\n(só no treino)",
          "d = 4, 8 ou 12\nfeatures", "Kernel ZZ (exato,\nshots, ruído) ou\nmodelos clássicos", "Busca em grade\ninterna\n(3 folds)",
          "F1-macro\n10 seeds x 5 folds\nt corrigido + Holm"]
fig = plt.figure(figsize=(7.0, 0.95)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
r = fig.canvas.get_renderer()
textos = [ax.text(0, 0.5, e, ha="center", va="center", fontsize=6.3, linespacing=1.15) for e in etapas]
larguras = [t.get_window_extent(renderer=r).width / fig.bbox.width for t in textos]
alturas = [t.get_window_extent(renderer=r).height / fig.bbox.height for t in textos]
folga = 0.012
espaco = (1 - 2 * 0.005 - sum(l + 2 * folga for l in larguras)) / (len(etapas) - 1)
x = 0.005
bordas = []
for t, l, h in zip(textos, larguras, alturas):
    w = l + 2 * folga
    t.set_position((x + w / 2, 0.5))
    ax.add_patch(FancyBboxPatch((x, 0.5 - max(alturas) / 2 - 0.08), w, max(alturas) + 0.16,
                                boxstyle="round,pad=0,rounding_size=0.02", fc="#eef3fb", ec="#1f77b4", lw=0.9))
    bordas.append((x, x + w)); x += w + espaco
for (a0, a1), (b0, b1) in zip(bordas[:-1], bordas[1:]):
    ax.add_patch(FancyArrowPatch((a1 + 0.003, 0.5), (b0 - 0.003, 0.5), arrowstyle="-|>", mutation_scale=7, color="#444", lw=0.8))
fig.savefig("figuras/fig_pipeline.pdf")
print("figuras geradas")
