from multiprocessing import cpu_count
from itertools import groupby
import pandas as pd
import string
import spacy
import time


# Spacy permette di escludere parti della pipeline non necessarie per migliorare le prestazioni
nlp = spacy.load('en_core_web_sm', exclude=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])


# Numero di core disponibili
CPUs = cpu_count()


# Normalizza i testi di un corpus di coppie: [(testo, contesto),...]
# Possono essere implementate diverse funzioni in futuro per esperimento
def spacy_text_norm(corpus):

    converted = []

    for doc, context in nlp.pipe(corpus, as_tuples=True, n_process=CPUs):
        tokens = [t.ent_type_ if t.ent_type_ else t.text.lower() for t in doc]
        tokens = [token for token in tokens if token.isprintable() and token.isalpha()]
        tokens = [g[0] for g in groupby(tokens)] # Collassa ripetizioni e.g. DATE, DATE, DATE -> DATE        
        norm_phr = ' '.join(tokens)
        converted.append((norm_phr, context))

    return converted


# Applica la normalizzazione del testo ad una tabella
# nel formato: [(phrase, e1, t1, e2, t2, ...), ...]
# la phrase o il testo da normalizzare deve essere nel primo campo di ogni record
def text_triples_norm(table):

    # Creazione corpus (phrase, contesto)
    corpus = [(record[0], record[1:]) for record in table]

    # Applica text normalization alle phrases in table
    norm_corpus = spacy_text_norm(corpus) 

    # Ricostruisce la tabella in output
    return [(record[0], *record[1]) for record in norm_corpus]


# Utility per preprocessare un dataframe e salvarlo su disco
# Se nel file tsv la prima riga non contiene i nomi delle colonne
# è possibile specificarli con field_names
def preprocess_df(df):

    tup_list = [tuple(r) for r in df.to_records(index=False)]
    norm_tup_list = text_triples_norm(tup_list)
    n_df = pd.DataFrame(norm_tup_list, columns=df.columns)
    
    return n_df



## TEST AREA ##
if __name__ == '__main__':

    print('some test')
