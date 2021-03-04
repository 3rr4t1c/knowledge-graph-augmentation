from complexPackage.complexModel import ComplEx
from complexPackage.dataset import Dataset
import numpy as np
from sklearn.preprocessing import minmax_scale

import torch

# Init dataset
print('Loading ComplEx dataset...', end='', flush=True)
#DATASET = Dataset(name='FB15k-237') # Imposta nome dataset
DATASET = Dataset(name='split_degree_build_v2') # Imposta nome dataset
print('Done.')  

# Init ComplEx model
#hyperparameters = {'dimension': 1000, 'init_scale': 1e-3} # Imposta iperparametri complex FB15k-237
hyperparameters = {'dimension': 500, 'init_scale': 1e-3} # Imposta iperparametri complex 2MKGLectorNoInv
print("Initializing ComplEx model...", end='', flush=True)
COMPLEX = ComplEx(dataset=DATASET, hyperparameters=hyperparameters, init_random=True)   # type: ComplEx
COMPLEX.to('cuda')
#COMPLEX.load_state_dict(torch.load('complexPackage/stored_models/ComplEx_FB15k-237.pt')) # Imposta file di modello da caricare
COMPLEX.load_state_dict(torch.load('complexPackage/stored_models/ComplEx_split_degree_build_d500_bs1000.pt')) # Imposta file di modello da caricare
COMPLEX.eval()
print('Done.')

# The ComplEx model wrapper
def complex_wrapper(list_of_pairs):        
       
    # Structures to handle possible unknown entities
    pair2complex = dict()
    pair2unknown = dict()
    for pair in list_of_pairs:
        try:        
            h_id = DATASET.get_id_for_entity_name(pair[0])
            r_id = 0  # Relation id will be ignored in predictions
            t_id = DATASET.get_id_for_entity_name(pair[1])        
            pair2complex[pair] = (h_id, r_id, t_id)
        except:
            pair2unknown[pair] = set(['unknown'])

    # Build ComplEx input
    complex_input = list(pair2complex.values())
    
    # Compute all scores (output is a 3D tensor with dim (npairs, nrelations, 1))
    all_scores = COMPLEX.all_scores_relations(complex_input)[:, :, 0]

    # Normalize scores to range [0, 1]
    minmax_scale(all_scores, axis=1, copy=False)

    # Array 2d con le posizioni [[riga, colonna]] dei valori che superano la soglia
    best_rels = np.argwhere(all_scores > 0.9) 
        
    # Build wrapper output
    complex_output = [set() for _ in complex_input]
    for row in best_rels:
        rel = DATASET.get_name_for_relation_id(row[1])
        # Exclude inverse relations
        if 'INVERSE' not in rel:
            complex_output[row[0]].add(rel)

    # Re-associate pairs to complex outputs
    pair2complex = dict(zip(pair2complex.keys(), complex_output))

    # Build wrapper output
    output = list()
    for pair in list_of_pairs:
        try:
            output.append(pair2complex[pair])
        except:
            output.append(pair2unknown[pair])

    # Add unknown prediction to empty sets
    for pred in output:
        if not pred:
            pred.add('unknown')
    # Empty sets may occurr when min(x) == max(x) in minMax scaling

    return output



if __name__ == '__main__':

    import time

    # Legge un TSV come lista di tuple
    import csv
    def read_tsv(file_path):
        out = list()
        with open(file_path, 'r', encoding='utf8') as tsv_file:
            rd = csv.reader(tsv_file, delimiter='\t')
            for line in rd:                
                out.append(tuple(line))
        return out
    
    # plist_FB15k237 = [("/m/08966", "/m/05lf_"),
    #                 ("/m/01hww_", "/m/01q99h"),
    #                 ("/m/09v3jyg", "/m/0f8l9c"),
    #                 ("/m/02jx1", "/m/013t85"),
    #                 ("/m/02jx1", "/m/0m0bj"),
    #                 ("/m/02bfmn", "/m/04ghz4m"),
    #                 ("/m/05zrvfd", "/m/04y9mm8"),
    #                 ("/m/060bp", "/m/04j53"),
    #                 ("/m/07l450", "/m/082gq"),
    #                 ("/m/07h1h5", "/m/029q3k") ]

    plist_Lector = read_tsv('complexPackage/data/split_degree_build/train.txt')
    plist_input = [(x[0], x[2]) for x in plist_Lector[:7000]]
    plist_input.append(('pippo', 'pluto'))

    # tic = time.perf_counter()
    # result = complex_model_wrapper(plist_input)
    # toc = time.perf_counter()
    # print(f"Old complex wrapper: {toc - tic:0.4f} seconds")
    
    # print()
    # [print(x) for x in result]

    # true_positive = 0
    # for i, fact in enumerate(plist_Lector[:100]):
    #     h, r, t = fact
    #     if r in result[i]:
    #         true_positive += 1

    # print(true_positive/100)

    tic = time.perf_counter()
    result = complex_wrapper(plist_input)
    toc = time.perf_counter()
    print(f"\nComplex wrapper: {toc - tic:0.4f} seconds")
    
    # print()
    # [print(x) for x in result]
    
