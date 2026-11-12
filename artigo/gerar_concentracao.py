from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

raiz = Path(__file__).resolve().parent.parent
with open(raiz / 'resultados' / 'concentracao.json') as arquivo:
    dados = json.load(arquivo)
plt.rcParams.update({'font.family': 'STIXGeneral', 'font.size': 9, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, eixos = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
for ax, base, titulo in zip(eixos, ['yelp', 'repro'], ['Yelp', 'RePro']):
    for largura, estilo, cor in [(1.0, '-', 'black'), (0.01, '--', '0.45')]:
        linhas = sorted((x for x in dados if x['base'] == base and x['emaranhamento']
                         and x['largura_banda'] == largura), key=lambda x: x['qubits'])
        ax.semilogy([x['qubits'] for x in linhas], [x['variancia'] for x in linhas],
                    linestyle=estilo, color=cor, linewidth=1.1,
                    label='$b=1$' if largura == 1.0 else '$b=0{,}01$')
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel('Número de qubits')
    ax.set_xticks([2, 6, 10, 14, 16])
eixos[0].set_ylabel('Variância fora da diagonal')
eixos[1].legend(frameon=False, fontsize=9, loc='lower left')
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parent / 'figuras' / 'fig_concentracao_final.pdf', bbox_inches='tight')
