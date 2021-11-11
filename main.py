import spacy
from dataclasses import dataclass
import json

import plotly.express as px
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans

nlp_fr = spacy.load("fr_core_news_md")

@dataclass
class TokenizerFR():
    modele : spacy.lang.fr.French
    fonctions : list

    def tokenize(self, texte):
        texte_tokens = self.modele(texte)
        return " ".join([token.lemma_ for token in texte_tokens if token.pos_ in self.fonctions and not token.lemma_.isdigit()])
            
        
## Inputs

data_path = "DATA_PFE_11_01_2021.json"
n_resumes = 100
n_clusters_max = n_resumes//5
n_dimensions = 50
n_termes_pertinents = 3
max_df = n_resumes//10
min_df = 2


## Lecture des données d'ArchiRès
with open(data_path,"r",encoding="utf-8") as file:
    data = json.load(file)
    file.close()

data = pd.DataFrame(data)
print(f"Nombre de documents : {len(data)}")
data = data[~data["resume"].isna()]
print(f"Nombre de documents avec résumé : {len(data)}")
data = data.head(n_resumes)
print(f"Nombre de documents à clusteriser : {len(data)}")


## Ajustement des variables lieu_projet et zone_geo_projet
def corriger_localisation(lieux):
    if len(lieux) != 0:
        return lieux[0]
    else:
        return "Inconnu"

data["lieu_projet"] = data["lieu_projet"].apply(corriger_localisation)
data["zone_geo_projet"] = data["zone_geo_projet"].apply(corriger_localisation)


## Tokenisation des résumés
TK = TokenizerFR(nlp_fr,["NOUN"])#,"VERB"
corpus_tokenized = data["resume"].apply(TK.tokenize)

## Calcul de l'indice TFIDF de chaque terme
vectorizer = TfidfVectorizer(max_df = max_df, min_df = min_df, decode_error = "ignore")#ngram_range=(1,3),
X = vectorizer.fit_transform(corpus_tokenized)

## Réduction dimensionnelle <https://scikit-learn.org/0.18/auto_examples/text/document_clustering.html>
svd = TruncatedSVD(n_dimensions)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)

## Clustering avec l'algorithme KMeans, en déterminant le nombre optimal de cluster grâce au coefficient silhouette

resultats = []

for n_clusters in range(2,n_clusters_max+1):
    
    algo_clustering = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1)#
    Y = algo_clustering.fit_predict(X)
    coeff_silhouette = metrics.silhouette_score(X, algo_clustering.labels_, sample_size=1000)
    print(f"{n_clusters} clusters : coefficient silhouette de {coeff_silhouette}")
    resultats.append({
        "n_clusters" : n_clusters,
        "algo" : algo_clustering,
        "labels" : Y,
        "coeff_silhouette" : coeff_silhouette
        })

## Sélection de l'algorithme avec un nombre optimal de clusters
resultats = pd.DataFrame(resultats)
resultats = resultats.sort_values(by=["coeff_silhouette"],ascending=False).reset_index()
n_clusters = resultats.at[0,"n_clusters"]
algo_clustering = resultats.at[0,"algo"]
Y = resultats.at[0,"labels"]

print(f"NOMBRE DE CLUSTERS OPTIMAL : {n_clusters}")

## Récupération des termes les plus pertinents par cluster
centroides_originaux = svd.inverse_transform(algo_clustering.cluster_centers_)
ordre_centroides = centroides_originaux.argsort()[:, ::-1]
termes = vectorizer.get_feature_names_out()
print(f"NOMBRES DE TERMES : {len(termes)}")

termes_par_cluster = {i : ", ".join([termes[ind] for ind in ordre_centroides[i, :n_termes_pertinents]]) for i in range(n_clusters)}

[print(k,v) for k,v in termes_par_cluster.items()]


## Affectation des résultats dans les documents de départ
data["cluster_index"] = Y
data["cluster_termes"] = data["cluster_index"].map(termes_par_cluster)


## Réduction dimensionnelle à trois dimensions pour le traçage
pca = PCA(3)
X_2d = pca.fit_transform(X)

data["x"] = [x for x,y,z in X_2d]
data["y"] = [y for x,y,z in X_2d]
data["z"] = [z for x,y,z in X_2d]

## Traçage du graphique
import plotly.express as px

fig = px.scatter_3d(
    data, 
    x='x', 
    y='y', 
    z='z',
    color='cluster_termes',
    opacity=0.9,
    hover_name="titre",
    hover_data={
        "x" : False,
        "y" : False,
        "z" : False,
        "annee" : True,
        "lieu_projet" : True,
        "zone_geo_projet" : True},
    template="simple_white",
    )
fig.update_layout(hoverlabel={"font_size" : 14})
fig.write_html("carte_sémantique.html")
