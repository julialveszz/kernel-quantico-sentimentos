# Kernel quântico x SVM clássico em embeddings comprimidos

Código do artigo "Kernel quântico versus SVM clássico ajustado sobre embeddings comprimidos: classificação de sentimentos em português e inglês" (Inteli).

A ideia é comparar um QSVM com kernel quântico (ZZ feature map) com modelos clássicos usando exatamente as mesmas features comprimidas, nas bases Yelp (inglês) e RePro (português).

## Como rodar

```
pip install -r requirements.txt
python 01_preparar_dados.py
python 02_gerar_embeddings.py
python 03_rodar_experimentos.py
python 04_concentracao.py
python 05_estatistica.py
python 06_tabelas_latex.py
python 07_figuras.py
```

No Mac dá pra rodar tudo de uma vez, em paralelo, com `zsh rodar_mac.sh`.

Os parâmetros ficam todos no `config.py`. Os resultados que estão no artigo são os de `resultados/`; os da primeira rodada (com a grade de largura de banda menor) ficaram em `resultados/rodada1/`.

## Dados

- Yelp: Kotzias et al. (2015), Sentiment Labelled Sentences, UCI, CC BY 4.0
- RePro: dos Santos Silva et al. (2024), https://github.com/lucasnil/repro, CC BY-NC-SA 4.0 (uso não comercial)
