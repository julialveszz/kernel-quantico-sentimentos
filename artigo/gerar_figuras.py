from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

raiz = Path(__file__).resolve().parent.parent
saida = Path(__file__).resolve().parent / 'figuras'
resumo = pd.read_csv(raiz / 'resultados' / 'resumo_f1.csv')
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 0.7, 'pdf.fonttype': 42})

# comparação entre QSVM e SVM ajustados
fig, ax = plt.subplots(figsize=(6.9, 3.9))
largura = 0.16
for i, base in enumerate(['yelp', 'repro']):
    dados = resumo[resumo.base == base]
    for deslocamento, modelo, cenario, cor, nome in [
        (-largura / 2, 'qsvm_ajustado', 'exato', '#353535', 'QSVM ajustado'),
        (largura / 2, 'svm_ajustado', 'classico', 'white', 'SVM-RBF ajustado'),
    ]:
        valor = dados[(dados.modelo == modelo) & (dados.cenario == cenario)].f1_media.mean()
        ax.bar(i + deslocamento, valor - 0.89, bottom=0.89, width=largura * 0.9,
               color=cor, edgecolor='#353535', linewidth=0.8, label=nome if i == 0 else None)
        ax.text(i + deslocamento, valor + 0.001, f'{valor:.4f}'.replace('.', ','), ha='center', fontsize=9)
    q = dados[(dados.modelo == 'qsvm_ajustado') & (dados.cenario == 'exato')].f1_media.mean()
    s = dados[(dados.modelo == 'svm_ajustado') & (dados.cenario == 'classico')].f1_media.mean()
    ax.text(i, 0.960, 'Δ ' + f'{100 * (q - s):+.2f}'.replace('.', ',') + ' p.p.', ha='center', fontsize=9)
ax.set_ylim(0.89, 0.965)
ax.set_xlim(-0.55, 1.55)
ax.set_xticks([0, 1], ['Yelp', 'RePro'])
ax.set_yticks([0.89, 0.90, 0.92, 0.94, 0.96])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.2f}'.replace('.', ',')))
ax.set_ylabel('F1-macro médio no teste')
ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.01), fontsize=9)
fig.tight_layout()
fig.savefig(saida / 'fig_principal.pdf', bbox_inches='tight')
plt.close(fig)

# comparação entre largura fixa e ajustada
fig, ax = plt.subplots(figsize=(7.2, 2.8))
for y, base in [(1, 'yelp'), (0, 'repro')]:
    dados = resumo[(resumo.base == base) & (resumo.cenario == 'exato')]
    fixa = dados[dados.modelo == 'qsvm_largura_fixa'].f1_media.mean()
    ajustada = dados[dados.modelo == 'qsvm_ajustado'].f1_media.mean()
    ax.plot([fixa, ajustada], [y, y], color='#b9b9b1', lw=1.5, zorder=1)
    ax.scatter([fixa], [y], s=70, facecolor='white', edgecolor='#263b4a', lw=1.3, zorder=3)
    ax.scatter([ajustada], [y], s=70, color='#263b4a', zorder=3)
    for valor, nome in [(fixa, 'Fixa'), (ajustada, 'Ajustada')]:
        ax.annotate(nome + '  ' + f'{valor:.2f}'.replace('.', ','), (valor, y),
                    xytext=(0, 12), textcoords='offset points', ha='center', fontsize=10)
ax.spines['left'].set_visible(False)
ax.set_xlim(0.4, 1.0)
ax.set_ylim(-0.42, 1.5)
ax.set_yticks([1, 0], ['Yelp', 'RePro'])
ax.tick_params(axis='y', length=0)
ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.1f}'.replace('.', ',')))
ax.set_xlabel('F1-macro no teste', labelpad=9)
fig.tight_layout()
fig.savefig(saida / 'fig_f1_treino_teste.pdf', bbox_inches='tight')
plt.close(fig)
