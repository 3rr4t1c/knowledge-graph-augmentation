# Dimostrazione addestramento ed estrazione relazioni da test set esterno.
# Training set di dati reali (KG con 120.000 fatti e TT con 1.200.000 testi + entità e tipi)
# Test set ricavato da FewRel (40.177 frasi estratte dal testo con entità e tipi)

from type_controller import types_remap
from selector import SELector
import pandas as pd
import json


# Files esempio con dati reali estratti da Lector
lector_KG = 'data/lector_example_train/knowledge_graph.tsv'
#lector_KG = 'data/KG500k_train/knowledge_graph.tsv'  # Test su KG con 500.000 fatti non sul repo
lector_TT = 'data/lector_example_train/text_triples.tsv'
#lector_TT = 'data/KG500k_train/text_triples_text_preprocessed.tsv' # Test su KG con 500.000 fatti non sul repo

# File di test ricavato da FewRel (entità e tipi tradotti in quelli di DBPedia)
fewrel_GT = 'data/fewrel_test/fewrel_translated_gt.tsv'
#fewrel_GT = 'data/fewrel_test/fewrel_translated_gt_text_preprocessed.tsv' # Test su KG con 500.000 fatti non sul repo

# File di descrizione property wikidata
pid2name_file = 'data/wikidata_stuff/pid2name.json'


# Parametri addestramento
unlabeled_sub = 0.5
no_types = False
text_norm = False
type_remapping = None #types_remap('data/dbpedia_stuff/types_hierarchy_fixed.tsv', -3) #, ['[Person]'])


# Specifica parametri interattivamente
answ = input(f'\n* Vuoi impostare i parametri manualmente? y / n (usa default) > ')
if answ[0].lower() == 'y':
    
    try:
        unlabeled_sub_custom = input('Inserisci percentuale di unlabeled da utilizzare [0-1] (default = 0.5) > ')
        unlabeled_sub = float(unlabeled_sub_custom)
    except:
        pass
    
    no_types_opt = input('Includere i tipi delle entità nei pattern? y / n (default = Yes) > ')
    no_types = no_types_opt[0].lower() == 'n' if no_types_opt is not '' else no_types

    text_norm_opt = input('Normalizzare le phrases? (Multiprocessing, potrebbe volerci qualche minuto)  y / n (default = No) > ')
    text_norm = text_norm_opt[0].lower() == 'y' if text_norm_opt is not '' else text_norm
    
    type_remapping_opt = input('Riassegnare i tipi a grana larga (se non si usano i tipi è ininfluente) ? y / n (default = No) > ')
    if type_remapping_opt is not '' and type_remapping_opt[0].lower() == 'y':
        type_remapping = types_remap('data/dbpedia_stuff/types_hierarchy_fixed.tsv', -2)
        

# Mostra informazioni
print('* Avvio demo con parametri:')
print(f'Percentuale sottocampionamento unlabeled: {unlabeled_sub*100}%')
print(f'Utilizzo tipi come parte del pattern: {"Disabilitato" if no_types else "Abilitato"}')
print(f'Normalizzazione automatica delle phrases: {"Abilitata" if text_norm else "Disabilitata"}')
print(f'Riassegnazione tipi a grana grossa: {"Abilitata" if type_remapping else "Disabilitata"}')


# Creazione modello 
slc = SELector(rseed=42, 
               unlabeled_sub=unlabeled_sub, 
               no_types=no_types, 
               text_norm=text_norm, 
               type_remapping=type_remapping)


# Addestramento
print('\n* Avvio addestramento modello:')
slc.train(lector_TT, lector_KG)


# Informazioni stato del modello
print('- Labeled triples:', len(slc.labeled_triples))
print('- Unlabeled triples:', len(slc.unlabeled_triples))
print('- Model triples:', len(slc.model_triples))


# Caricamento ground truth di FewRel
fewrel_df = pd.read_csv(fewrel_GT, sep='\t', names=['phr', 's', 'st', 'o', 'ot', 'PID'])


# Lettura JSON con il mapping pid -> nome property wikidata
with open(pid2name_file) as f:
    pid2name = json.load(f)


# Aggiunta campi informazione al dataframe di FewRel
fewrel_df['name'] = fewrel_df['PID'].apply(lambda x: pid2name[x][0])
fewrel_df['desc'] = fewrel_df['PID'].apply(lambda x: pid2name[x][1])


# Estrazione fatti senza considerare la ground truth
print('\n* Estrazione fatti dal dataset FewRel:')
fewrel_tt = fewrel_df.drop(['PID', 'name', 'desc'], axis=1).to_records(index=False)
harvested = slc.harvest([tuple(r) for r in fewrel_tt], keep_unknown=True)


# Costruzione dataframe per visualizzazione
harv_df = pd.DataFrame(harvested, columns=['Subject', 'Relation', 'Object'])


# Aggiunta delle descrizioni ai fatti estratti per ispezione manuale
harv_df['TRUE'] = '--->'
harv_df['True Property'] = fewrel_df['name']
#harv_df['Description'] = ground_truth['desc']
harv_df['PID'] = fewrel_df['PID']


# Mostra statistiche
print('\n* Statistiche sui fatti estratti')
print('- Relazioni totali nel test set:', harv_df.shape[0])
print('- Relazioni NON estratte dal modello:', harv_df[harv_df['Relation'] == 'unknown'].shape[0])
print('- Relazioni estratte dal modello:', harv_df[harv_df['Relation'] != 'unknown'].shape[0])


# Mostra dataset delle relazioni estratte per ispezione manuale
pd.set_option('display.max_rows', None)    
#pd.set_option('display.max_columns', 6)
#pd.set_option('display.width', 10)
harv_df_no_unknown = harv_df[harv_df['Relation'] != 'unknown']
answ = input(f'\n* Vuoi mostrare le relazioni estratte? ({harv_df_no_unknown.shape[0]} righe) y/n > ')
if answ.lower() == 'y' or answ.lower() == 'yes':
    answ = input('Quante righe vuoi mostrare? > ')
    try:
        answ = int(answ)
        print(harv_df_no_unknown.sample(answ))        
    except:
        print('Immissione non valida.')
        pass


# Distribuzione estrazioni
#pd.set_option('display.max_rows', None)
dis_df = harv_df[['Relation', 'True Property', 'PID']].copy()
dis_df['id'] = dis_df.index
dis_df = dis_df.groupby(['Relation', 'True Property', 'PID'])['id'].count().reset_index(name="count")
answ = input(f'\n* Vuoi mostrare tutte le associazioni aggregate per conteggio? ({dis_df.shape[0]} righe) y/n > ')
if answ.lower() == 'y' or answ.lower() == 'yes':
    print(dis_df.sort_values(by=['count'], ascending=False))


# Valutazione
from mappings import perfect_alignment
print('\n* Valutazione modello:')
p, r, f = slc.evaluate(fewrel_GT, perfect_alignment)
print(f'- Precision: {p}')
print(f'- Recall: {r}')
print(f'- F1-Score: {f}')
print('\n* Fine demo.')