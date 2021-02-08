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
def preprocess_tsv(file_path, field_names=None):

    df = pd.read_csv(file_path, sep='\t', names=field_names)
    tup_list = [tuple(r) for r in df.to_records(index=False)]
    norm_tup_list = text_triples_norm(tup_list)
    n_df = pd.DataFrame(norm_tup_list, columns=field_names)
    n_df.to_csv(file_path + '_text_preprocessed.tsv', sep='\t', index=False, header=False)
    assert df.shape[0] == n_df.shape[0], 'Row count mismatch!'



## TEST AREA ##
if __name__ == '__main__':

    print('Spacy pipeline:', nlp.pipe_names)

    # Reading data for test
    print('Lettura da disco...')
    df = pd.read_csv('text_triples.tsv', sep='\t', names=['phr', 'a', 'b', 'c', 'd'])
    tlist = list(df.sample(100000).to_records(index=False))
    tlist = [tuple(x) for x in tlist] # Conversione da lista di np.record a lista di tuple
    
    # Print before
    print('\nPrima conversione')
    for t in tlist[:100]:
        print(t)

    tic = time.perf_counter()
    tlist = text_triples_norm(tlist)
    toc = time.perf_counter()

    # Print after
    print('\nDopo conversione')
    for t in tlist[:100]:
        print(t)

    print('\nTempo esecuzione:', toc - tic)
    






# print('\n\nNLTK Performance')
# # NLTK Performance
# tic = time.perf_counter()
# phrase_normalizer_nltk(txt)
# toc = time.perf_counter()
# print('Tempo NLTK esecuzione singola:', toc - tic)
# # Esecuzione in serie

#tlist = [(phr, tlist[i][1], tlist[i][2], tlist[i][3], tlist[i][4])  for i, phr in enumerate(nlp.pipe(phr_list))]



# # Spacy performance
# print('\n\nSpacy Performance')
# tic = time.perf_counter()
# phrase_normalizer_spacy(txt)
# toc = time.perf_counter()
# print('Tempo Spacy esecuzione singola:', toc - tic)
# # Esecuzione in serie
# spacy_result = []
# tic = time.perf_counter()
# for phr, e1, t1, e2, t2 in tlist:
#     phr = phrase_normalizer_spacy(phr)
#     spacy_result.append((phr, e1, t1, e2, t2))
# toc = time.perf_counter()
# print('Tempo Spacy su serie da 1000', toc - tic)
# print(spacy_result[:10])


