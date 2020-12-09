# Import section
import numpy as np
from stub_Lector import StubLector
from complex import ComplEx
from dataset import Dataset
import torch


# Ranking  must take in account relations with high cardinality.
# Given a tail scores matrix (facts x scores), the row index as a map and the fact
# return the fact ranking over all the possible tail scores
def fRank(factsXscores, row_index, fact, decimals=3):
    # Read the scores for this fact per each tails and move data on RAM
    tails_scores = factsXscores[row_index[fact], :].cpu().numpy()
    # Read the score for current fact and tail
    fact_score = tails_scores[fact[2]]
    # Round all the scores to required decimals
    tails_scores = np.around(tails_scores, decimals)
    # Filter all unique scores (and sort ascending)
    tails_scores = np.unique(tails_scores)
    # Evaluate ranking normalized to [0, 1] interval
    return np.searchsorted(tails_scores, fact_score) / tails_scores.size
    # NOTE: searchsorted returns the index where fact_score should be inserted to maintain order


# Take a list (with duplicates) and return
# a dict with each element mapped on an integer
# corresponding to the row index of the score matrix
def create_row_index(seq):
    index = dict()
    row = 0
    for x in seq:
        if x not in index:
            index[x] = row
            row += 1
    
    return index


# Call this function to filter a lector map selecting best phrases with link prediction
# May I suggest name SELector? (It select best phrases and SE == Selection Enhanced)
# Note: In this way Lector and LP can be first trained in parallel with Python multiprocessing (to deepen in future)
# Pre: Same dataset (KG) must be used 
def proof_lector(lector_model, lp_model, rank_threshold=0.8, phrase_threshold=0.7):

    # Run phrase tracking mode onto corpus text 
    phrase2facts = lector_model.phrase_tracking_mode('fake text corpus')
    # NOTE: phrase tracking mode should be executed out of this function and checked if map exist

    # Map: fact as a sample -> row index (for quick row resolution)
    sample2row = create_row_index(sum(phrase2facts.values(), []))
    
    # Build the input matrix for link prediction
    # NOTE: Dict maintain insertion order: https://mail.python.org/pipermail/python-dev/2017-December/151283.html
    all_samples_matrix = np.array(list(sample2row.keys()))
    # Compute scores for each tail of KG # TODO: implement fact inversion to add heads scores in the process (?)
    all_scores_matrix = lp_model.all_scores(all_samples_matrix).detach()
    
    # For each phrase evaluate the corresponding facts list with LP
    for curr_phrase, facts_list in phrase2facts.items():
        # Evaluate rank for each fact predicted by the current phrase # NOTE: decimals are pre-setted to 1        
        rank_scores = [fRank(all_scores_matrix, sample2row, sample, 1) for sample in facts_list]        
        # Evaluate phrase fitness by some policy (accuracy overall)
        phrase_fitness = sum([0 if x <= rank_threshold else 1 for x in rank_scores]) / len(rank_scores)       
        # Filter Lector map phrase -> relation, using some policy/threshold
        if phrase_fitness < phrase_threshold:
            # Remove phrase changing Lector model state
            lector_model.phrase2relation.pop(curr_phrase) 

    return lector_model # Ready to work in Fact Harvesting Mode
# NOTE: This function is intended both to modify lector_model passed as parameter
# and return the object himself for general usage. 


## Evaluate precision and recall
# precision: remaining good phrases after proof lector run / all phrases after proof lector run 
# recall: remaining good phrase after proof lector run / all phrases before proof lector
def evaluate_proof_lector(before_state, after_state):
    # Relevant retrieved phrases cardinality
    RR_card = len([v for _, v in after_state.items() if v[0] == 1])
    # All retrieved phrases cardinality
    AR_card = len(after_state)
    # Total relevant phrases cardinality
    TR_card = len([v for _, v in before_state.items() if v[0] == 1])
    # Return precision, recall
    return RR_card/AR_card, RR_card/TR_card 


# TODO: evaluate precsion and recall per each relation
# def ...



## TEST AREA ### 
# The following code will be executed only if this module is not imported but executed as a script
if __name__ == "__main__":
    
    # Pretty printer for dict
    import pprint
    pp = pprint.PrettyPrinter()
    
    print('\nRunning some test...')
    
    # Init dataset
    print('\nLoading dataset...', end='', flush=True)
    kg = Dataset(name='FB15k-237')
    print('Done.')  
    
    # Init ComplEx model
    hyperparameters = {'dimension': 1000, 'init_scale': 1e-3}
    print("\nInitializing ComplEx model...", end='', flush=True)
    lp_model = ComplEx(dataset=kg, hyperparameters=hyperparameters, init_random=True)   # type: ComplEx
    lp_model.to('cuda')
    lp_model.load_state_dict(torch.load('stored_models/ComplEx_FB15k-237.pt'))
    lp_model.eval()
    print('Done.')
    
    # Init Lector (stub) model
    print('\nInitializing Lector model...', end='', flush=True)
    lector_model = StubLector(kg)
    lector_model.train_mode('fake corpus text')
    
    # Test Proof Lector
    print('\nLector initial state:')
    pp.pprint(lector_model.phrase2relation)
    lector_before_state = lector_model.phrase2relation.copy() # Save a copy for further evaluation
    print('\nRunning Proof Lector...', end='', flush=True)
    proof_lector(lector_model, lp_model)
    print('Done.')
    print('\nLector after state:')
    pp.pprint(lector_model.phrase2relation)

    # Precision and recall test
    precision, recall = evaluate_proof_lector(lector_before_state, lector_model.phrase2relation)
    f_score = 2 * ((precision * recall) / (precision + recall))
    print(f'\nPrecsion: {np.round(precision, 1)},', end='')
    print(f' Recall: {np.round(recall, 1)},', end='')
    print(f' F-Score: {np.round(f_score, 1)}')
