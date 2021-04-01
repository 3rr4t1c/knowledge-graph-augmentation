import pandas as pd
from sys import argv
from text_preprocessing import preprocess_df
from evtools import custom_stats

labeled_triples_fields = ['wikid', 'section', 'phrase_original', 
                          'phrase_placeholder', 'phrase_pre', 
                          'phrase_post', 'subject', 'wiki_subject', 
                          'type_subject', 'object', 'wiki_object', 
                          'type_object', 'relation']


target_fields = ['phrase_original', 'wiki_subject', 
                 'type_subject', 'wiki_object', 
                 'type_object', 'relation']


target_relations = ['birthPlace', 'deathPlace', 'nationality'
                    'team', 'almaMater', 'spouse', 'parent',
                    'child', 'religion', 'award', 'party',
                    'notableWork', 'recordLabel', 'bandMember']


# Questo script prende come parametro il path del file csv
# da usare come sorgente dati, nel formato prodotto da Lector originale
DATA_SOURCE = argv[1]


print('Reading data...', end='', flush=True)
main_df = pd.read_csv(DATA_SOURCE, names=labeled_triples_fields, usecols=target_fields)
n_rows, n_cols = main_df.shape
print(f'Done. [{n_rows} rows x {n_cols} columns]', flush=True)


print('\nRecreating the full Knowledge Graph...', end='', flush=True)
full_KG = main_df[['wiki_subject', 'relation', 'wiki_object']].drop_duplicates(ignore_index=True)
#full_KG = full_KG[full_KG['relation'].isin(target_relations)]  # Seleziona solo relazioni target
full_KG = full_KG[full_KG['relation'].str.contains('-1') == False]  # Seleziona solo relazioni non inverse
#full_KG = full_KG.sample(2000000) # Taglio del KG, commenta per usare tutti i fatti
print(f'Done. [Total Facts: {full_KG.shape[0]}]', flush=True)


# Indagine su cardinalità nodi (entità)
from tqdm import tqdm

print('\nBuilding cardinality index...')
card_index = dict()
for h, r, t in tqdm(full_KG.to_records(index=False)):
    
    try:
        card_index[h][0] += 1
    except:
        card_index[h] = [1, 0]

    try:
        card_index[t][1] +=1
    except:
        card_index[t] = [0, 1]

# Preview
print('\nPreview')
[print(k, '>', v) for k, v in list(card_index.items())[:10]]

print('\nFiltering low cardinality nodes...')
card_thresh = 5 # Soglia di cardinalità minima per nodo
target_entities = list()
for entity, cards in tqdm(card_index.items()):
    if sum(cards) >= card_thresh:
        target_entities.append(entity)
    
print('\nSelected entities:', len(target_entities), 'of', len(card_index))


# WARNING: Alcune entità target vengono perse se in tutti i fatti che le riguardano 
# sono connesse ad entità con bassa cardinalità. Basta cambiare l'operatore & in | per 
# mantenere anche altre eventuali entità a bassa cardinalità connesse ad entità ad alta cardinalità
print('\nSelecting all facts with known entities...', end='', flush=True)
full_KG = full_KG[full_KG['wiki_subject'].isin(target_entities) & full_KG['wiki_object'].isin(target_entities)]
print(f'Done. [Total facts: {full_KG.shape[0]}]', flush=True)


# print('\nSelecting target relations...', end='', flush=True)
# target_relations = full_KG.groupby('relation').size().sort_values(ascending=False).reset_index()
# target_relations = target_relations.iloc[:64] # Seleziona le relazioni target (64, modifica a piacere)
# target_relations['weights'] = target_relations[0].iloc[-1]
# target_relations['weights'] = target_relations['weights'] / target_relations[0] # Aggiunge pesi per un sampling pesato
# full_KG = full_KG[full_KG['relation'].isin(target_relations['relation'])] # Seleziona solo relazioni target
# rel2weight = dict(zip(target_relations.relation, target_relations.weights))
# full_KG['weights'] = full_KG['relation'].apply(lambda x: rel2weight[x])
# full_KG = full_KG.sample(1000000, weights=full_KG.weights).drop(['weights'], axis=1)
# print(f'Done. [Total Facts: {full_KG.shape[0]}]', flush=True)

# Usati per salvare il kg completo per l'addestramento
#full_KG.to_csv('full_KG.tsv', sep='\t', index=False, header=False)
#exit()


# Questo passo è necessario perchè alcune target entities potrebbero essersi perse
print('\nFinding all KGs entities...', end='', flush=True)
known_entities = pd.unique(full_KG[['wiki_subject', 'wiki_object']].values.ravel('K'))
print(f'Done. [Found {len(known_entities)} entities]', flush=True)


# Solo per informazione
print('\nFinding all KGs relations...', end='', flush=True)
known_relations = full_KG['relation'].drop_duplicates()
print(f'Done. [Found {known_relations.size} relations]', flush=True)


print('\nSelecting all patterns...', end='', flush=True)
all_patterns = main_df.drop('relation', axis=1).drop_duplicates(ignore_index=True)
print(f'Done. [Total patterns: {all_patterns.shape[0]}]', flush=True)


print('\nPreprocessing all patterns (long time)...', end='', flush=True)
all_patterns = preprocess_df(all_patterns)
print(f'Done.', flush=True)


print('\nSelecting all patterns with known entities...', end='', flush=True)
degree_patterns = all_patterns[all_patterns['wiki_subject'].isin(known_entities) & all_patterns['wiki_object'].isin(known_entities)]
print(f'Done. [Total patterns: {degree_patterns.shape[0]}]', flush=True)


print('\nSelecting all remaining patterns...', end='', flush=True)
diff_patterns = pd.concat([all_patterns, degree_patterns]).drop_duplicates(keep=False)
print(f'Done. [Total patterns: {diff_patterns.shape[0]}]', flush=True)

#exit()
# Usato per salvare anche tutti i pattern
#pattern_df.to_csv('patterns.tsv', sep='\t', index=False, header=False)


# print()
# print(full_KG)
# print(pattern_df)
# exit()


# print('\nCreating test and train KGs...', end='', flush=True)
# test_size = int(full_KG.shape[0]*0.2) # 20% of full KG
# full_KG = full_KG.sample(frac=1) # Shuffle full KG
# test_KG = full_KG.iloc[:test_size]
# train_KG = full_KG.iloc[test_size:]
# print(f'Done. [Test KG: {test_KG.shape[0]}, Train KG: {train_KG.shape[0]}]', flush=True)


# print('\nSplitting patterns in test and train...', end='', flush=True)
# test_KG_pair_set = {(rec[0], rec[2]) for rec in test_KG.to_records(index=False)} 
# pattern_df_records = pattern_df.to_records(index=False)
# pattern_df_test = list()
# pattern_df_train = list()
# for rec in pattern_df_records:
#     pair = (rec[1], rec[3])
#     if pair in test_KG_pair_set:
#         pattern_df_test.append(rec)
#     else:
#         pattern_df_train.append(rec)
# pattern_df_test = pd.DataFrame.from_records(pattern_df_test, columns=pattern_df.columns)
# pattern_df_train = pd.DataFrame.from_records(pattern_df_train, columns=pattern_df.columns)
# print(f'Done. [Test patterns: {pattern_df_test.shape[0]}, Train patterns: {pattern_df_train.shape[0]}]', flush=True)


# print('\nPreprocessing phrases, adding placeholders...', end='', flush=True)
# pattern_df_test = preprocess_df(pattern_df_test)
# pattern_df_train = preprocess_df(pattern_df_train)
# pattern_df = preprocess_df(pattern_df)
# print(f'Done.', flush=True)

# print(pattern_df_test)
# print(test_KG)
# print(pattern_df_train)
# print(train_KG)

print()
print(full_KG)
print(degree_patterns)
print(diff_patterns)
degree_patterns.to_csv('pattern_degree_build.tsv', sep='\t', index=False, header=False)

full_KG.to_csv('kg_degree_build.tsv', sep='\t', index=False, header=False)

diff_patterns.to_csv('diffp_degree_build.tsv', sep='\t', index=False, header=False)

# answ = input('\nSave data on disk? (will overwrite existing files) y/n? > ')
# if answ[0].lower() == 'y':
#     pattern_df_train.to_csv('dataset/train/pattern_triples_train.tsv', sep='\t', index=False, header=False)
#     train_KG.to_csv('dataset/train/knowledge_graph_train.tsv', sep='\t', index=False, header=False)
    
#     pattern_df_test.to_csv('dataset/test/pattern_triples_test.tsv', sep='\t', index=False, header=False)
#     test_KG.to_csv('dataset/test/knowledge_graph_test.tsv', sep='\t', index=False, header=False)