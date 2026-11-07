# prepara as duas bases: Yelp (Kotzias et al.) e RePro
# a RePro tem 4 rotulos, entao ficam so as avaliacoes so positivas ou so negativas
# e, como a Yelp tem frases curtas, a RePro e filtrada pela faixa de tamanho das frases da Yelp

import json, os
os.makedirs("resultados", exist_ok=True)
import numpy as np
import pandas as pd
import config

def contar_palavras(texto):
    return len(str(texto).split())

# --- Yelp ---
linhas = []
with open("dados/yelp_labelled.txt", encoding="utf-8") as f:
    for linha in f:
        linha = linha.rstrip("\n")
        if not linha.strip():
            continue
        texto, rotulo = linha.rsplit("\t", 1)
        linhas.append({"texto": texto.strip(), "rotulo": int(rotulo)})
yelp = pd.DataFrame(linhas)
yelp["palavras"] = yelp["texto"].apply(contar_palavras)

# faixa de tamanho que vai ser usada na RePro
minimo, maximo = np.percentile(yelp["palavras"], config.PERCENTIS_TAMANHO)
minimo, maximo = int(np.floor(minimo)), int(np.ceil(maximo))

# --- RePro ---
repro_bruta = pd.read_csv("dados/RePro.csv")
repro = repro_bruta[repro_bruta["polarity"].isin(["['POSITIVO']", "['NEGATIVO']"])].copy()
repro["texto"] = repro["review_text"].astype(str).str.strip()
repro["rotulo"] = (repro["polarity"] == "['POSITIVO']").astype(int)
repro["palavras"] = repro["texto"].apply(contar_palavras)
repro = repro[(repro["palavras"] >= minimo) & (repro["palavras"] <= maximo)]
repro = repro.drop_duplicates(subset="texto")

gerador = np.random.default_rng(config.SEED_AMOSTRA_REPRO)
partes = []
for rotulo in [0, 1]:
    candidatos = repro[repro["rotulo"] == rotulo]
    escolhidos = gerador.choice(len(candidatos), size=config.TAMANHO_POR_CLASSE, replace=False)
    partes.append(candidatos.iloc[escolhidos])
repro_final = pd.concat(partes).reset_index(drop=True)[["texto", "rotulo", "palavras"]]

yelp.to_csv("dados/yelp_preparada.csv", index=False)
repro_final.to_csv("dados/repro_preparada.csv", index=False)

# estatisticas de tamanho pra reportar no artigo
def resumo(df):
    p = df["palavras"]
    return {"n": int(len(df)), "positivas": int(df["rotulo"].sum()),
            "mediana": float(p.median()), "q1": float(p.quantile(.25)), "q3": float(p.quantile(.75)),
            "media": float(p.mean()), "min": int(p.min()), "max": int(p.max())}

info = {
    "faixa_palavras_usada": [minimo, maximo],
    "repro_original": int(len(repro_bruta)),
    "repro_so_pos_ou_neg": int(repro_bruta["polarity"].isin(["['POSITIVO']", "['NEGATIVO']"]).sum()),
    "repro_dentro_da_faixa": int(len(repro)),
    "yelp": resumo(yelp),
    "repro": resumo(repro_final),
}
json.dump(info, open("resultados/estatisticas_dados.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(info, indent=2, ensure_ascii=False))
