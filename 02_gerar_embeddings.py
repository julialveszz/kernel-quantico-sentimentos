# gera os embeddings das duas bases com o MESMO modelo multilingue
# uso normal:   python 02_gerar_embeddings.py
# so pra testar o pipeline sem baixar o modelo: python 02_gerar_embeddings.py --substituto
# (o substituto NAO e usado nos resultados do artigo)

import sys, json
import numpy as np
import pandas as pd
import config

usar_substituto = "--substituto" in sys.argv
bases = {"yelp": pd.read_csv("dados/yelp_preparada.csv"),
         "repro": pd.read_csv("dados/repro_preparada.csv")}

if usar_substituto:
    # TF-IDF + SVD com 384 dimensoes, so pra conferir se o pipeline roda do comeco ao fim
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    nome_modelo = "tfidf+svd384 (SUBSTITUTO, so teste de pipeline)"
    def codificar(textos):
        tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2).fit_transform(textos)
        return TruncatedSVD(384, random_state=0).fit_transform(tf).astype(np.float32)
else:
    from sentence_transformers import SentenceTransformer
    modelo = SentenceTransformer(config.MODELO_EMBEDDING)
    nome_modelo = config.MODELO_EMBEDDING
    def codificar(textos):
        return modelo.encode(list(textos), batch_size=64, show_progress_bar=True,
                             convert_to_numpy=True).astype(np.float32)

for nome, df in bases.items():
    X = codificar(df["texto"])
    np.savez_compressed(f"dados/embeddings_{nome}.npz", X=X, y=df["rotulo"].values)
    print(nome, X.shape)

json.dump({"modelo": nome_modelo, "substituto": usar_substituto},
          open("dados/embeddings_info.json", "w"), indent=2, ensure_ascii=False)
