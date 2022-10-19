import re
import pickle
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import ToktokTokenizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

nltk.download("stopwords")

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

def topicosReview(cuerpo, indice_review):
    # Cargo el modelo lda
    file = open("./modelos/lda.sav", "rb")
    lda = pickle.load(file)
    file.close()

    bow_review = cuerpo[indice_review]

    # Indices de los topicos mas significativos
    #dist_indices = [topico[0] for topico in lda[bow_review]]
    # Contribución de los topicos mas significativos
    dist_contrib = [topico[1] for topico in lda[bow_review]]

    return dist_contrib

def topicosTest(review, diccionario):
    dfOld = review
    df = review[["open_response"]]

    # 1.- Limpiamos (quitar caracteres especiaes, minúsculas...)
    df["Tokens"] = df.open_response.apply(limpiar_texto)

    # 2.- Tokenizamos
    tokenizer = ToktokTokenizer()
    df["Tokens"] = df.Tokens.apply(tokenizer.tokenize)

    # 3.- Eliminar stopwords y digitos
    df["Tokens"] = df.Tokens.apply(eliminar_stopwords)

    # 4.- ESTEMIZAR / LEMATIZAR ???
    df["Tokens"] = df.Tokens.apply(estemizar)

    diccionario.filter_extremes(no_below=2, no_above = 0.8)
    cuerpo = [diccionario.doc2bow(review) for review in df.Tokens]

    documents = df["open_response"]
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
    tf_vectorizer.fit_transform(documents.values.astype(str))

    topicos = []
    for i in range (len(documents)):
        topicos.append(topicosReview(cuerpo, i))

    df["Topicos"] = topicos
    df["newid"] = dfOld["newid"]  # guardamos los ids

    return df

def topicosTrain(df, num_Topics):
    # ---> Parte 1: https://elmundodelosdatos.com/topic-modeling-gensim-fundamentos-preprocesamiento-textos/
    #ruta = str(input("Introduce el path relativo (EJ: ./datasets/nombre.csv) :"))
    dfOld = df
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
    #print(df.Tokens[0][0:10])


    # ---> Parte 2: https://elmundodelosdatos.com/topic-modeling-gensim-asignacion-topicos/
    # Cargamos en el diccionario la lista de palabras que tenemos de las reviews
    diccionario = Dictionary(df.Tokens)
    #print(f'Número de tokens: {len(diccionario)}') #mostrar el numero se palabras

    # Reducimos el diccionario filtrando las palabras mas raras o demasiado frecuentes
    # no_below = mantener tokens que se encuentran en el a menos x documentos
    # no_above = mantener tokens que se encuentran en no mas del 80% de los documentos
    diccionario.filter_extremes(no_below=2, no_above = 0.8)
    #print(f'Número de tokens: {len(diccionario)}')

    # Creamos el corpus (por cada token en el df) QUE ES UN ARRAY BOW
    cuerpo = [diccionario.doc2bow(review) for review in df.Tokens]

    # BOW de una review
    # print(corpus[5])

    documents = df["open_response"]
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
    tf_vectorizer.fit_transform(documents.values.astype(str))

    lda = LdaModel(corpus=cuerpo, id2word=diccionario,
               num_topics=num_Topics, random_state=42,
               chunksize=1000, passes=1,
               alpha=2, eta=2)

    # Guardo el modelo
    file = open("./modelos/lda.sav", "wb")
    pickle.dump(lda, file)
    file.close()
    topicos = []
    for i in range (len(documents)):
        topicos.append(topicosReview(cuerpo, i))
    df["Topicos"] = topicos

    df["newid"] = dfOld["newid"]    #guardamos los ids

    return df, diccionario
    # Para esta review random sacamos el array de contribuciones de cada topico
