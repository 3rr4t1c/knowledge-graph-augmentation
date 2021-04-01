from sys import argv
import pandas as pd
import random

file_path = argv[1]


# Costruisce un dizionario con i degree di ogni nodo
def make_node2degree(df):

    records = df.to_records(index=False)
    node2degree = dict()
    for h, r, t in records:
        try:
            node2degree[h] += 1
        except:
            node2degree[h] = 1
        try:
            node2degree[t] += 1
        except:
            node2degree[t] = 1

    return node2degree


def custom_splitter(df, n):
    
    degree_map = make_node2degree(df)
    h_deg = df['h'].apply(lambda x: degree_map[x])
    t_deg = df['t'].apply(lambda x: degree_map[x]) 
    degree_df = pd.DataFrame({'h_deg': h_deg, 't_deg': t_deg})    
    weights = 2/(1/degree_df['h_deg'] + 1/degree_df['t_deg'])
    weights = weights.apply(lambda x: 0 if x <= 10 else x) # Zero prob to extract nodes with weight less or equal 10
    sampled_df = df.sample(n, weights=weights)
    remainder_df = pd.concat([df, sampled_df]).drop_duplicates(keep=False)

    return remainder_df, sampled_df    
    

# Read data
df = pd.read_csv(file_path, sep='\t', names=['h', 'r', 't'])

# kg_degree_build cardinality: [742303 rows x 3 columns]
train_set_df, test_set_df = custom_splitter(df, 85000)
train_set_df, valid_set_df = custom_splitter(train_set_df, 5000)


# # Test Set
# test_set_size = 5000  #int(df.shape[0]*0.2) # 20% dell'intero KG
# df = df.sample(frac=1) # Shuffle facts
# test_set_df = df.iloc[:test_set_size]
# df = df.iloc[test_set_size:]

# # Validation Set
# valid_set_size = 5000 #int(df.shape[0]*0.2) # 20% del KG senza test set
# df = df.sample(frac=1) # Shuffle facts
# valid_set_df = df.iloc[:valid_set_size]

# # Training Set
# train_set_df = df.iloc[valid_set_size:]


# Check for entities
print('\nChecking entities...', end='', flush=True)
train_set_entities = pd.unique(train_set_df[['h', 't']].values.ravel('K'))
valid_check = valid_set_df[valid_set_df.h.isin(train_set_entities) & valid_set_df.t.isin(train_set_entities)]
test_check = test_set_df[test_set_df.h.isin(train_set_entities) & test_set_df.t.isin(train_set_entities)]
assert valid_check.shape[0] == valid_set_df.shape[0], f'Ritenta, sarai più fortunato! {valid_check.shape[0]}'
assert test_check.shape[0] == test_set_df.shape[0], f'Ritenta, sarai più fortunato! {test_check.shape[0]}'
print(f'Done.', flush=True)

print('\nTraining Set:')
print(train_set_df)

print('\nValidation Set:')
print(valid_set_df)

print('\nTest Set:')
print(test_set_df)

print('\nSaving...', end='', flush=True)
train_set_df.to_csv('train.tsv', sep='\t', index=False, header=False)
valid_set_df.to_csv('valid.tsv', sep='\t', index=False, header=False)
test_set_df.to_csv('test.tsv', sep='\t', index=False, header=False)
print(f'Done.', flush=True)