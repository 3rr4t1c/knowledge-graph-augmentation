# Import section
import numpy as np
from stub_Lector import StubLector
from complex import ComplEx
from dataset import Dataset
import torch
from sklearn.preprocessing import MinMaxScaler
mmscaler = MinMaxScaler()


# Ranking must take in account relations with high cardinality.
# Given a tail scores matrix (facts x scores), a fact index and a tail index return the rank
def fRank(factsXscores, fact_index, tail_index, decimals=3):
    # The scalar score corresponding to a single fact (h, r, t)
    fact_score = factsXscores[fact_index, tail_index]
    # All the scores for all the possible tails (h, t, ?), rounded to choosen number of decimals
    tail_scores = np.around(factsXscores[fact_index, :], decimals)
    # All the scores with duplicates removed
    tail_scores = np.unique(tail_scores)
    # Sort all the scores from minimum to maximum
    tail_scores.sort()
    # searchsorted returns the index where fact_score should be inserted to maintain order
    return np.searchsorted(tail_scores, fact_score) / tail_scores.size # Normalized to [0,1] interval


# Call this function to filter a lector map selecting best phrases with link prediction
# May I suggest name SELector? (It select best phrases and SE == Second Evaluation)
# Note: In this way Lector and LP can be first trained in parallel with Python multiprocessing (to deepen in future)
# Pre: Same dataset (KG) must be used 
def proof_lector(lector_model, lp_model, rank_threshold=0.7, phrase_threshold=0.7):

    # Run phrase tracking mode onto corpus text
    phrase2facts = lector_model.phrase_tracking_mode('fake text corpus')
    
    # For each phrase evaluate the corresponding facts list with LP
    for curr_phrase, facts_list in phrase2facts.items():
        #print(curr_phrase, lector_model.phrase2relation[curr_phrase]) #debug
        # Compute scores for each tail of KG # TODO: implement fact inversion to add heads scores in the process
        samples_matrix = np.array(facts_list)        
        all_scores_matrix = lp_model.all_scores(samples_matrix).detach().cpu().numpy()
        #print(all_scores_matrix) #debug       
        # Evaluate rank for each fact predicted by the current phrase # NOTE: decimals are pre-setted to 1        
        rank_scores = [fRank(all_scores_matrix, i, sample[2], 1) for i, sample in enumerate(facts_list)]
        #print(rank_scores) #debug
        # Evaluate phrase fitness by some policy (accuracy overall)
        phrase_fitness = sum([0 if x <= rank_threshold else 1 for x in rank_scores]) / len(rank_scores) 
        #print(phrase_fitness) #debug 
        # Filter Lector map phrase -> relation, using some policy/threshold
        if phrase_fitness < phrase_threshold:
            # Remove phrase changing Lector model state
            lector_model.phrase2relation.pop(curr_phrase) 

    return lector_model # Ready to work in Fact Harvesting Mode
# Warning: This function is intended both to modify lector_model passed as parameter
# and return the object himself for general usage. 



## BEGIN TEST AREA ### 
# The following code will be executed only if this module is not imported but executed as a script
if __name__ == "__main__":
    print('Running some test...')
    # Init dataset
    print('Loading dataset...', end='', flush=True)
    kg = Dataset(name='FB15k-237')
    print('Done.')  
    # Init ComplEx model
    hyperparameters = {'dimension': 1000, 'init_scale': 1e-3}
    print("Initializing ComplEx model...")
    lp_model = ComplEx(dataset=kg, hyperparameters=hyperparameters, init_random=True)   # type: ComplEx
    lp_model.to('cuda')
    lp_model.load_state_dict(torch.load('stored_models/ComplEx_FB15k-237.pt'))
    lp_model.eval()
    # Init Lector model
    print('Initializing Lector model...')
    lector_model = StubLector(kg)
    lector_model.train_mode('fake corpus text')
    # Test function
    print('Lector initial state: \n', lector_model.phrase2relation)
    proof_lector(lector_model, lp_model)
    print('Lector after state: \n', lector_model.phrase2relation)
