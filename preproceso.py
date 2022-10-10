import json
import re

import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import ToktokTokenizer
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary
from sklearn.decomposition import LatentDirichletAllocation
import random
import numpy as np
import matplotlib.pyplot as plt


STOPWORDS = set(stopwords.words("english"))
wnl = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

def limpiar_texto(texto):
    # Eliminamos los caracteres especiales
    texto = re.sub(r'\W', ' ', str(texto))
    # Eliminado las palabras que tengo un solo caracter
    texto = re.sub(r'\s+[a-zA-Z]\s+', ' ', texto)
    # Sustituir los espacios en blanco en uno solo
    texto = re.sub(r'\s+', ' ', texto, flags=re.I)
    # Convertimos textos a minusculas
    texto = texto.lower()
    return texto

def eliminar_stopwords(tokens):
    return [token for token in tokens if token not in STOPWORDS and not token.isdigit()]

def lematizar(tokens):
    return [wnl.lemmatize(token) for token in tokens]

def estemizar(tokens):
    return [stemmer.stem(token) for token in tokens]

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print("Topic %d:" % (topic_idx))
        print(''.join([' ' +feature_names[i] + ' ' + str(round(topic[i], 5)) #y esto también
                for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        docProbArray=np.argsort(W[:,topic_idx])
        print(docProbArray)
        howMany=len(docProbArray);
        print("How Many");
        print(howMany);
        for doc_index in top_doc_indices:
            print(documents[doc_index])

# ---> Parte 1: https://elmundodelosdatos.com/topic-modeling-gensim-fundamentos-preprocesamiento-textos/
#ruta = str(input("Introduce el path relativo (EJ: ./datasets/nombre.csv) :"))
ruta = "./datasets/train.csv"
df = pd.read_csv(ruta)
df = df[["open_response"]]

# 1.- Limpiamos (quitar caracteres especiaes, minúsculas...)
df["Tokens"] = df.open_response.apply(limpiar_texto)

# 2.- Tokenizamos
tokenizer= ToktokTokenizer()
df["Tokens"] = df.Tokens.apply(tokenizer.tokenize)

# 3.- Eliminar stopwords y digitos
df["Tokens"] = df.Tokens.apply(eliminar_stopwords)

# 4.- ESTEMIZAR / LEMATIZAR ???
df["Tokens"] = df.Tokens.apply(estemizar)
print(df.Tokens[0][0:10])


# ---> Parte 2: https://elmundodelosdatos.com/topic-modeling-gensim-asignacion-topicos/
# Cargamos en el diccionario la lista de palabras que tenemos de las reviews
diccionario = Dictionary(df.Tokens)
print(f'Número de tokens: {len(diccionario)}') #mostrar el numero se palabras

# Reducimos el diccionario filtrando las palabras mas raras o demasiado frecuentes
# no_below = mantener tokens que se encuentran en el a menos x documentos
# no_above = mantener tokens que se encuentran en no mas del 80% de los documentos
diccionario.filter_extremes(no_below=2, no_above = 0.8)
print(f'Número de tokens: {len(diccionario)}')

# Creamos el corpus (por cada roken en el df) QUE ES UN ARRAY BOW
corpus = [diccionario.doc2bow(review) for review in df.Tokens]

# BOW de una review
# print(corpus[5])

from sklearn.feature_extraction.text import CountVectorizer
documents = df["open_response"]
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
tf = tf_vectorizer.fit_transform(documents.values.astype(str))
tf_feature_names = tf_vectorizer.get_feature_names()

lda = LatentDirichletAllocation(corpus=corpus, id2word=diccionario,
               num_topics=5, random_state=42,
               chunksize=1000, passes=1,
               alpha=2, eta=2)

# Imprimimos los topicos creados con las 5 palabras que más contribuyen a ese tópico y sus pesos
lda_W = lda.transform(tf)
lda_H = lda.components_

# Imprimimos los tópicos necesarios
print("LDA Topics:")
terms = tf_vectorizer.get_feature_names()
for index, component in enumerate(lda_H):
    zipped = zip(terms, component)
    top_terms_key=sorted(zipped, key = lambda t: t[1], reverse=True)[:7]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(index)+": ",top_terms_list)
lda_H=lda.components_ /lda.components_.sum(axis=1)[:, np.newaxis]  #esto cambia
print("LDA Topics")
display_topics(lda_H, lda_W, tf_feature_names, documents, 10, 10)

# Aqui imprimimos una review aleatoria para comprobar la eficacia de nuestro modelo!
indice_review = random.randint(0,len(df))
review = df.iloc[indice_review]

import pickle
pickle.dump(lda, open("./modelos/lda.sav", "wb"))

print("***********************")
print("\nReview: " + review[0] + "\n")
print("***********************")

'''
# Obtenemos el BOW de la review
# Obtenemos la distribucion de topicos
bow_review = corpus[indice_review]
distribucion_review = lda[bow_review]

# Indices de los topicos mas significativos
dist_indices = [topico[0] for topico in lda[bow_review]]
# Contribución de los topicos mas significativos
dist_contrib = [topico[1] for topico in lda[bow_review]]

# Representacion grafica de los topicos mas significativos
distribucion_topicos = pd.DataFrame({'Topico':dist_indices,
                                     'Contribucion':dist_contrib })
distribucion_topicos.sort_values('Contribucion',
                                 ascending=False, inplace=True)
ax = distribucion_topicos.plot.bar(y='Contribucion',x='Topico',
                                   rot=0, color="orange",
                                   title = 'Tópicos mas importantes'
                                   'de review ' + str(indice_review))

# Imprimimos las palabras mas significativas de los topicos
cabeceras = ["num_topics", "alpha", "beta"]
nombreTXT = "Gnsm-" + "nTopics_" + str(lda.num_topics) + "-alpha_" + str(lda.alpha[0]).split(".")[0] + "-beta_" + str(lda.eta[0]).split(".")[0] + ".txt"
print(nombreTXT)
archivo = open(nombreTXT, "w")
writer = csv.writer(archivo)
print("PREPARANDO ARCHIVO .TXT PARA VOLCAR DISTRIBUCION PALABRAS EN LOS TOPICOS...")


for ind, topico in distribucion_topicos.iterrows():

    print("*** Tópico: " + str(int(topico.Topico)) + " ***")
    writer.writerow("*** Tópico: " + str(int(topico.Topico)) + " ***\n")

    palabras = [palabra[0] for palabra in lda.show_topic(
        topicid=int(topico.Topico))]
    palabras = ', '.join(palabras)
    print(palabras, "\n")
    writer.writerow(str(palabras)+ "\n\n\n")

archivo.close()
'''