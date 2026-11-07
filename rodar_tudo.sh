#!/bin/sh
# roda o pipeline inteiro, na ordem
set -e
python3 01_preparar_dados.py
python3 02_gerar_embeddings.py          # baixa o modelo do Hugging Face na primeira vez
python3 validar_qiskit.py
python3 03_rodar_experimentos.py        # parte mais demorada, pode ser retomada se parar
python3 04_concentracao.py
python3 05_estatistica.py
python3 06_tabelas_latex.py
python3 07_figuras.py
