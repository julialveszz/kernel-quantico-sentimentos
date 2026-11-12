# gera as tabelas em LaTeX que o artigo inclui com \input
import pandas as pd
import config

resumo = pd.read_csv("resultados/resumo_f1.csv")
testes = pd.read_csv("resultados/testes_estatisticos.csv")
NOMES_BASE = {"yelp": "Yelp", "repro": "RePro"}
MODELOS = [("logistica", "Log."), ("svm_padrao", "SVM pad."), ("svm_ajustado", "SVM aj."),
           ("qsvm_sem_emaranhamento", "QSVM sep."), ("qsvm_largura_fixa", "QSVM fixa"), ("qsvm_ajustado", "QSVM aj.")]

def f1(base, comp, d, cen, modelo):
    cen = "classico" if modelo in ("logistica", "svm_padrao", "svm_ajustado") else cen
    linha = resumo[(resumo.base == base) & (resumo.compressao == comp) & (resumo.d == d) & (resumo.cenario == cen) & (resumo.modelo == modelo)]
    if linha.empty:
        return "--"
    return f"{linha.f1_media.iloc[0]:.3f} $\\pm$ {linha.f1_dp.iloc[0]:.3f}".replace(".", ",")

# tabela 1: F1 de todos os modelos no cenario exato
saida = ["\\begin{table*}[t]", "\\centering",
         "\\caption{F1-macro (média $\\pm$ desvio padrão entre as médias das 10 seeds) no cenário de estado exato. Todos os modelos recebem as mesmas features comprimidas. O $\\pm$ só descreve a variação entre seeds e não serve como teste: as comparações estão na Tabela~\\ref{tab:testes}.}",
         "\\label{tab:f1_exato}", "\\scriptsize", "\\setlength{\\tabcolsep}{3.5pt}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llc" + "c" * len(MODELOS) + "}", "\\hline",
         "Base & Compressão & $d$ & " + " & ".join(n for _, n in MODELOS) + " \\\\", "\\hline"]
for base in ["yelp", "repro"]:
    for comp in config.COMPRESSOES:
        for d in config.DIMENSOES:
            celulas = [f1(base, comp, d, "exato", m) for m, _ in MODELOS]
            saida.append(f"{NOMES_BASE[base]} & {comp.upper()} & {d} & " + " & ".join(celulas) + " \\\\")
    saida.append("\\hline")
saida += ["\\end{tabular}}", "\\end{table*}"]
open("tabelas/tab_f1_exato.tex", "w").write("\n".join(saida))

# tabela 2: testes principais (H1, H2, H5) no cenario exato
def fmt(x, casas=3):
    texto = f"{x:.{casas}f}".replace(".", ",")
    return texto[1:] if texto.startswith("-") and set(texto[1:]) <= set("0,") else texto
def pfmt(p):
    return "<0,001" if p < 0.001 else fmt(p)
comparacoes = [("H1_qsvm_vs_svm_ajustado", "H\\textsubscript{1}: QSVM aj. $-$ SVM aj."),
               ("H2_svm_ajustado_vs_padrao", "H\\textsubscript{2}: SVM aj. $-$ SVM pad."),
               ("H5_qsvm_vs_sem_emaranhamento", "H\\textsubscript{5}: QSVM aj. $-$ QSVM sep.")]
saida = ["\\begin{table*}[t]", "\\centering",
         "\\caption{Testes pareados no cenário exato, sobre as 50 diferenças por fold: diferença média de F1-macro com intervalo de confiança de 95\\%, estatística $t$ corrigida de Nadeau e Bengio (49 graus de liberdade), p-valor com correção de Holm sobre as 12 configurações de cada comparação e diferença com cerca de 80\\% de chance de ser detectada (DMD).}",
         "\\label{tab:testes}", "\\footnotesize", "\\begin{tabular}{lllccccc}", "\\hline",
         "Comparação & Base & Compr. & $d$ & $\\Delta$ F1 [IC 95\\%] & $t$ & $p_{\\mathrm{Holm}}$ & DMD \\\\", "\\hline"]
for chave, rotulo in comparacoes:
    t = testes[(testes.comparacao == chave) & (testes.cenario == "exato")]
    primeira = True
    for base in ["yelp", "repro"]:
        for comp in config.COMPRESSOES:
            for d in config.DIMENSOES:
                l = t[(t.base == base) & (t.compressao == comp) & (t.d == d)]
                if l.empty:
                    continue
                l = l.iloc[0]
                nome = rotulo if primeira else ""
                primeira = False
                saida.append(f"{nome} & {NOMES_BASE[base]} & {comp.upper()} & {d} & {fmt(l.diferenca)} [{fmt(l.ic_inf)}; {fmt(l.ic_sup)}] & {fmt(l.t, 2)} & {pfmt(l.p_holm)} & {fmt(l.dmd)} \\\\")
    saida.append("\\hline")
saida += ["\\end{tabular}", "\\end{table*}"]
open("tabelas/tab_testes.tex", "w").write("\n".join(saida))

# tabela 3: QSVM ajustado em cada cenario (d = 4 e 8)
cens = [c for c in ["exato", "shots"] + list(config.RUIDOS) if c in resumo.cenario.unique()]
nomes_cen = {"exato": "Exato", "shots": "1.000 shots", "ruido_baixo": "Ruído $p=0{,}001$", "ruido_alto": "Ruído $p=0{,}01$"}
saida = ["\\begin{table}[t]", "\\centering",
         "\\caption{F1-macro do QSVM ajustado em cada cenário de cálculo do kernel. Os cenários com ruído foram rodados só com 4 e 8 qubits.}",
         "\\label{tab:cenarios}", "\\footnotesize", "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{llc" + "c" * len(cens) + "}", "\\hline",
         "Base & Compr. & $d$ & " + " & ".join(nomes_cen[c] for c in cens) + " \\\\", "\\hline"]
for base in ["yelp", "repro"]:
    for comp in config.COMPRESSOES:
        for d in config.DIMENSOES:
            saida.append(f"{NOMES_BASE[base]} & {comp.upper()} & {d} & " + " & ".join(f1(base, comp, d, c, "qsvm_ajustado").split(" $")[0].replace("--", "n.a.") for c in cens) + " \\\\")
    saida.append("\\hline")
saida += ["\\end{tabular}}", "\\end{table}"]
open("tabelas/tab_cenarios.tex", "w").write("\n".join(saida))

# tabela 4: H3
t = testes[testes.comparacao == "H3_dif_pca_vs_nca"]
saida = ["\\begin{table}[t]", "\\centering",
         "\\caption{H\\textsubscript{3}: diferença média de F1-macro entre QSVM ajustado e SVM ajustado com PCA e com NCA, no cenário exato, e p-valor do $t$ corrigido para a mudança dessa diferença entre as duas compressões, com correção de Holm sobre as 6 configurações.}",
         "\\label{tab:h3}", "\\footnotesize", "\\begin{tabular}{lcccc}", "\\hline",
         "Base & $d$ & $\\Delta$ PCA & $\\Delta$ NCA & $p_{\\mathrm{Holm}}$ \\\\", "\\hline"]
for base in ["yelp", "repro"]:
    for d in config.DIMENSOES:
        l = t[(t.base == base) & (t.d == d) & (t.cenario == "exato")]
        if l.empty:
            continue
        l = l.iloc[0]
        saida.append(f"{NOMES_BASE[base]} & {d} & {fmt(l.media_1)} & {fmt(l.media_2)} & {pfmt(l.p_holm)} \\\\")
saida += ["\\hline", "\\end{tabular}", "\\end{table}"]
open("tabelas/tab_h3.tex", "w").write("\n".join(saida))
# tabela 5: F1 no treino com largura fixa e alinhamento com o kernel RBF (cenario exato)
saida = ["\\begin{table}[t]", "\\centering",
         "\\caption{F1-macro no treino e no teste do QSVM com largura de banda fixa ($b = 1$) e alinhamento centrado entre o kernel quântico escolhido e o kernel RBF do \\svm{} ajustado, no cenário exato (médias entre seeds e folds).}",
         "\\label{tab:treino_alinhamento}", "\\footnotesize", "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{llccccc}", "\\hline",
         "& & & \\multicolumn{2}{c}{QSVM largura fixa} & \\multicolumn{2}{c}{Alinhamento com RBF} \\\\",
         "Base & Compr. & $d$ & treino & teste & QSVM aj. & QSVM sep. \\\\", "\\hline"]
def valor(base, comp, d, modelo, coluna):
    l = resumo[(resumo.base == base) & (resumo.compressao == comp) & (resumo.d == d) & (resumo.cenario == "exato") & (resumo.modelo == modelo)]
    return fmt(l[coluna].iloc[0])
for base in ["yelp", "repro"]:
    for comp in config.COMPRESSOES:
        for d in config.DIMENSOES:
            saida.append(f"{NOMES_BASE[base]} & {comp.upper()} & {d} & {valor(base, comp, d, 'qsvm_largura_fixa', 'f1_treino')} & {valor(base, comp, d, 'qsvm_largura_fixa', 'f1_media')} & {valor(base, comp, d, 'qsvm_ajustado', 'alinhamento_rbf')} & {valor(base, comp, d, 'qsvm_sem_emaranhamento', 'alinhamento_rbf')} \\\\")
    saida.append("\\hline")
saida += ["\\end{tabular}}", "\\end{table}"]
open("tabelas/tab_treino_alinhamento.tex", "w").write("\n".join(saida))
# booktabs_nas_tabelas: mesma cara das tabelas escritas a mao no artigo
import glob
for arquivo in glob.glob("tabelas/*.tex"):
    linhas = open(arquivo).read().split("\n")
    pos = [i for i, l in enumerate(linhas) if l.strip() == "\\hline"]
    for k, i in enumerate(pos):
        linhas[i] = "\\toprule" if k == 0 else ("\\bottomrule" if k == len(pos) - 1 else "\\midrule")
    open(arquivo, "w").write("\n".join(linhas))
print("tabelas geradas")
