# Dimostrazione addestramento ed estrazione relazioni da test set esterno.
# Training set di dati reali (KG con 120.000 fatti e TT con 1.200.000 testi + entità e tipi)
# Test set ricavato da FewRel (40.177 frasi estratte dal testo con entità e tipi)

from type_regularization import types_remap
from selector import SELector
import pandas as pd
import json

# Files esempio con dati reali estratti da Lector
lector_KG = 'data/lector_example/train/knowledge_graph.tsv'
lector_TT = 'data/lector_example/train/text_triples.tsv'

# Files di test ricavati da FewRel (entità e tipi di DBPedia)
fewrel_TT = 'data/lector_example/test/fewrel_translated.tsv'
fewrel_GT = 'data/lector_example/test/fewrel_translated_gt.tsv'

# File di traduzione property wikidata
pid2name_file = 'data/wikidata_stuff/pid2name.json'

# Parametri addestramento
unlabeled_sub = 0.6
no_types = False
text_norm = False
type_remapping = types_remap('data/dbpedia_stuff/types_hierarchy_fixed.tsv', -1, ['[Person]'])

# Mostra informazioni
print('Avvio demo con parametri:')
print(f'Percentuale sottocampionamento unlabeled: {unlabeled_sub*100}%')
print(f'Utilizzo tipi come parte del pattern: {"Disabilitato" if no_types else "Abilitato"}')
print(f'Normalizzazione automatica delle phrases: {"Abilitata" if text_norm else "Disabilitata"}')
print(f'Riassegnazione tipi a grana grossa: {"Abilitata" if type_remapping else "Disabilitata"}')

# Creazione modello 
slc = SELector(unlabeled_sub=unlabeled_sub, no_types=no_types, text_norm=text_norm, type_remapping=type_remapping)

# Addestramento
slc.train(lector_TT, lector_KG)

# Informazioni stato del modello
print('- Model triples:', len(slc.model_triples))
print('- Labeled triples:', len(slc.labeled_triples))
print('- Unlabeled triples:', len(slc.unlabeled_triples))

# Estrazione relazioni dal dataset di test
harvested = slc.harvest(fewrel_TT, keep_unknown=True)

# Costruzione dataframe per visualizzazione
harv_df = pd.DataFrame(harvested, columns=['Subject', 'Relation', 'Object'])

# Lettura JSON con il mapping pid -> nome property wikidata
with open(pid2name_file) as f:
    pid2name = json.load(f)

# Lettura e preparazione ground truth
ground_truth = pd.read_csv(fewrel_GT, sep='\t', names=['PID'])
ground_truth['name'] = ground_truth['PID'].apply(lambda x: pid2name[x][0])
ground_truth['desc'] = ground_truth['PID'].apply(lambda x: pid2name[x][1])

# Aggiunta delle descrizioni ai fatti estratti per ispezione manuale
harv_df['Separator'] = '--->'
harv_df['True Property'] = ground_truth['name']
#harv_df['Description'] = ground_truth['desc']
harv_df['PID'] = ground_truth['PID']

# Mostra statistiche
print('\nRelazioni totali nel test set:', harv_df.shape[0])
print('Relazioni NON estratte dal modello:', harv_df[harv_df['Relation'] == 'unknown'].shape[0])
print('Relazioni estratte dal modello:', harv_df[harv_df['Relation'] != 'unknown'].shape[0])

# Mostra dataset delle relazioni estratte per ispezione manuale
pd.set_option('display.max_rows', None)    
#pd.set_option('display.max_columns', 6)
#pd.set_option('display.width', 10)
harv_df_no_unknown = harv_df[harv_df['Relation'] != 'unknown']
answ = input(f'Vuoi mostrare le relazioni estratte? ({harv_df_no_unknown.shape[0]} righe) y/n > ')
if answ.lower() == 'y' or answ.lower() == 'yes':
    answ = input('Quante righe vuoi mostrare? > ')
    try:
        answ = int(answ)
        print(harv_df_no_unknown.sample(answ))
        print('Fine demo.')
    except:
        print('Immissione invalida, termino la demo.')
        pass
