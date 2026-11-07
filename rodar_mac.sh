#!/bin/zsh
# roda tudo no Mac: instala, prepara os dados, gera os embeddings e roda o experimento em paralelo
cd "$(dirname "$0")"

echo "== 1/5 criando ambiente e instalando pacotes =="
python3 -m venv .venv || exit 1
source .venv/bin/activate
pip install --upgrade pip -q
pip install -q sentence-transformers pandas scikit-learn scipy matplotlib qiskit || exit 1

mkdir -p resultados figuras tabelas
echo "== 2/5 preparando os dados =="
python 01_preparar_dados.py || exit 1

echo "== 3/5 gerando os embeddings (baixa o modelo na primeira vez) =="
python 02_gerar_embeddings.py || exit 1
python validar_qiskit.py || exit 1

echo "== 4/5 rodando o experimento em 8 processos + concentracao (uns 10 a 15 min) =="
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p resultados
rm -f resultados/parte*.csv resultados/log_parte*.txt
python 03_rodar_experimentos.py --bases yelp  --seeds 0,1,2 --saida resultados/parte1.csv > resultados/log_parte1.txt 2>&1 &
python 03_rodar_experimentos.py --bases yelp  --seeds 3,4,5 --saida resultados/parte2.csv > resultados/log_parte2.txt 2>&1 &
python 03_rodar_experimentos.py --bases yelp  --seeds 6,7   --saida resultados/parte3.csv > resultados/log_parte3.txt 2>&1 &
python 03_rodar_experimentos.py --bases yelp  --seeds 8,9   --saida resultados/parte4.csv > resultados/log_parte4.txt 2>&1 &
python 03_rodar_experimentos.py --bases repro --seeds 0,1,2 --saida resultados/parte5.csv > resultados/log_parte5.txt 2>&1 &
python 03_rodar_experimentos.py --bases repro --seeds 3,4,5 --saida resultados/parte6.csv > resultados/log_parte6.txt 2>&1 &
python 03_rodar_experimentos.py --bases repro --seeds 6,7   --saida resultados/parte7.csv > resultados/log_parte7.txt 2>&1 &
python 03_rodar_experimentos.py --bases repro --seeds 8,9   --saida resultados/parte8.csv > resultados/log_parte8.txt 2>&1 &
python 04_concentracao.py > resultados/log_concentracao.txt 2>&1 &
wait

if grep -l "Traceback" resultados/log_*.txt > /dev/null 2>&1; then
  echo "erro em algum processo. confira os arquivos:"
  grep -l "Traceback" resultados/log_*.txt
  exit 1
fi

echo "== 5/5 juntando os resultados =="
python -c "import pandas as pd,glob; pd.concat([pd.read_csv(f) for f in sorted(glob.glob('resultados/parte*.csv'))]).to_csv('resultados/resultados_por_fold.csv', index=False)"

echo ""
echo "pronto. resultados gerados:"
echo "  $(pwd)/resultados/resultados_por_fold.csv"
echo "  $(pwd)/resultados/concentracao.json"
