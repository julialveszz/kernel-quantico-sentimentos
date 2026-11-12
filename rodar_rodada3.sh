#!/bin/zsh
# rodada 3: busca aninhada, grade de largura maior, Yelp sem frases repetidas e diagonal com ruido
cd "$(dirname "$0")"
source .venv/bin/activate || exit 1
setopt NULL_GLOB
mkdir -p resultados/rodada2
mv -f resultados/resultados_por_fold.csv resultados/concentracao.json resultados/rodada2/ 2>/dev/null
rm -f resultados/parte*.csv resultados/log_parte*.txt

echo "preparando os dados e gerando os embeddings de novo (a Yelp mudou)..."
python 01_preparar_dados.py > /dev/null || exit 1
python 02_gerar_embeddings.py > /dev/null || exit 1
pip freeze > resultados/versoes_ambiente.txt

# um processo por nucleo do M5 (10), cada um com 2 seeds, e a concentracao junto
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "rodando a rodada 3 (uns 15 a 30 min)..."
k=1
for base in yelp repro; do
  for seeds in 0,1 2,3 4,5 6,7 8,9; do
    python 03_rodar_experimentos.py --bases $base --seeds $seeds --saida resultados/parte$k.csv > resultados/log_parte$k.txt 2>&1 &
    k=$((k+1))
  done
done
python 04_concentracao.py > resultados/log_concentracao.txt 2>&1 &
wait
if grep -l "Traceback" resultados/log_*.txt > /dev/null 2>&1; then
  echo "DEU ERRO. Me manda este arquivo:"; grep -l "Traceback" resultados/log_*.txt; exit 1
fi
python -c "import pandas as pd,glob; pd.concat([pd.read_csv(f) for f in sorted(glob.glob('resultados/parte*.csv'))]).to_csv('resultados/resultados_por_fold.csv', index=False)"
echo ""
echo "PRONTO. Me manda estes tres arquivos:"
echo "  $(pwd)/resultados/resultados_por_fold.csv"
echo "  $(pwd)/resultados/concentracao.json"
echo "  $(pwd)/resultados/versoes_ambiente.txt"
