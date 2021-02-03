from multiprocessing import Pool
import pandas as pd
import spacy
import nltk

nlp = spacy.load('en_core_web_sm')

# Normalizza una phrase con la libreria spacy
def spacy_phrase_normalizer(phr):

    doc = nlp(phr.lower())
    tokens = [t.text if not t.ent_type_ else t.ent_type_ for t in doc]
    tokens = [token for token in tokens if token.isalpha()]
    phr = ' '.join(tokens)

    return phr


# Applica normalizzazione ad un solo record
# Un record deve essere una tupla: (phrase, entity1, type1, entity2, type2)
# BUG: Una funzione che viene parallelizzata non può contenere tuple unpacking
# def record_normalizer_bug(record):

#     phr, e1, t1, e2, t2 = record # Tuple unpacking
#     pos = ['CD', 'JJ', 'PRP', 'NNP']
#     tokens = nltk.word_tokenize(phr)
#     #tokens = [token for token in tokens if token.isalpha()]
#     pos_tags = nltk.pos_tag(tokens)
#     nphr = [w if t not in pos else t for w, t in pos_tags]

#     return (' '.join(nphr), e1, t1, e2, t2)


# Normalizza una phrase con la libreria NLTK
def nltk_phrase_normalizer(phr):
    pos = ['CD', 'JJ', 'JJR', 'JJS', 'PRP', 'PRP$', 'NN', 'NNS', 'NNP', 'NNPS']
    tokens = nltk.word_tokenize(phr)
    tokens = [token for token in tokens if token.isalpha()]
    pos_tags = nltk.pos_tag(tokens)
    nphr = [w if t not in pos else t for w, t in pos_tags]

    return ' '.join(nphr) 


# Applica normalizzazione a tutti i record di una tabella nel formato usato da SELector
# Sfrutta tutti i core disponibili per una normalizzazione in parallelo
def parallel_phrases_normalizer(tb, phrase_normalizer):

    phrases = [record[0] for record in tb]

    with Pool() as pool:
        phrases = pool.map(phrase_normalizer, phrases)

    tb_norm = [(phr, tb[i][1], tb[i][2], tb[i][3], tb[i][4]) for i, phr in enumerate(phrases)]        

    return tb_norm



## TEST AREA ##
if __name__ == '__main__':

    # Carica dati
    print('Lettura da disco...', end='', flush=True)
    df = pd.read_csv('data/lector_example/train/text_triples.tsv', sep='\t', names=['phr', 'a', 'b', 'c', 'd'])
    print('Fatto.', flush=True)

    # Seleziona alcuni record e converte la struttura dati
    print('Scelta casuale di qualche record...', end='', flush=True)
    tlist = list(df.sample(100).to_records(index=False))
    print('Fatto.', flush=True)

    # Mostra alcuni record prima
    print('\nRecords originali:\n')
    for t in tlist[:30]:
        print(t)

    # Normalizzazione
    tlist = parallel_phrases_normalizer(tlist, nltk_phrase_normalizer)

    # Mostra alcuni record dopo
    print('\nRecords normalizzati:\n')
    for t in tlist[:30]:
        print(t)

