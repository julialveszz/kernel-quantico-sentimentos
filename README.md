# Kernel quântico e SVM clássico em análise de sentimentos

Experimento do artigo de Júlia Alves, que compara um classificador com kernel quântico a modelos clássicos sobre os mesmos embeddings comprimidos. As bases são a Yelp, em inglês, e a RePro, em português.

## Experimento

Os textos são representados pelo modelo multilíngue `paraphrase-multilingual-MiniLM-L12-v2` e comprimidos com PCA ou NCA para 4, 8 ou 12 dimensões. A comparação usa regressão logística, SVM-RBF e variantes do QSVM com mapa ZZ. O kernel é simulado em estado exato e, nas configurações indicadas no artigo, com shots e ruído despolarizante global.

A avaliação usa 10 sementes de validação cruzada estratificada com 5 folds. A busca interna de hiperparâmetros usa 3 folds; padronização e compressão são reajustadas dentro de cada fold interno. Os parâmetros estão em [config.py](config.py).

## Resultados e artigo

- [Resultados por fold](resultados/resultados_por_fold.csv): métricas e hiperparâmetros escolhidos na rodada 3.
- [Testes estatísticos](resultados/testes_estatisticos.csv): diferenças pareadas, intervalos, p-valores e DMD.
- [Rodadas anteriores](resultados/): subpastas `rodada1/` e `rodada2/`.
- [Artigo em LaTeX](artigo/artigo.tex): fonte, figuras e tabelas.
- [Suplemento](artigo/suplemento/): tabela completa e CSV dos testes.

As figuras da versão do artigo são geradas a partir de `resultados/resumo_f1.csv` pelo script `artigo/gerar_figuras.py`. O script `07_figuras.py` gera as figuras da versão anterior do experimento, na pasta `figuras/` da raiz. As tabelas incluídas no artigo estão em `artigo/tabelas/`; as da raiz são geradas por `06_tabelas_latex.py`.

## Instalação e execução

Crie um ambiente Python e instale as dependências:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Para refazer o fluxo completo, execute os scripts numerados na ordem ou use `sh rodar_tudo.sh`. A geração dos embeddings baixa o modelo na primeira execução. O experimento principal é a etapa mais demorada e pode ser retomado por base e seed. Para gerar apenas as figuras do artigo com os resultados já salvos:

```sh
python artigo/gerar_figuras.py
```

`rodar_rodada3.sh` prepara uma execução paralela no macOS. Ele move os resultados atuais para `resultados/rodada2/` e pode sobrescrever arquivos nessa pasta; faça uma cópia do histórico antes de usá-lo. Não é necessário executá-lo para consultar os resultados incluídos.

As versões registradas na execução estão em [resultados/versoes_ambiente.txt](resultados/versoes_ambiente.txt). A revisão específica do modelo de embeddings não foi fixada; uma execução futura pode não reproduzir os vetores originais exatamente.

## Dados

- Yelp: Kotzias et al. (2015), Sentiment Labelled Sentences, UCI, CC BY 4.0.
- RePro: dos Santos Silva et al. (2024), [repositório da base](https://github.com/lucasnil/repro), CC BY-NC-SA 4.0.
