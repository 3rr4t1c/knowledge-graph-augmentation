# Import section
import numpy as np
from stub_Lector import StubLector
from complex import ComplEx
from dataset import Dataset
import torch


# Call this function to filter a lector map selecting best phrases with link prediction
# May I suggest name SELector? (It select best phrases and SE == Second Evaluation)
# Note: In this way Lector and LP can be first trained in parallel with Python multiprocessing (to deepen in future)
# Pre: Same dataset (KG) must be used 
def proof_lector(lector_model, lp_model):

    # Run phrase tracking mode onto corpus text
    phrase2facts = lector_model.phrase_tracking_mode('fake text corpus')
    
    # For each phrase evaluate the corresponding facts list with LP
    for curr_phrase, facts_list in phrase2facts.items():
        # Compute scores for each tail of KG (Is this mandatory? Why?)
        samples_matrix = np.array(facts_list)        
        phrase_all_tails_matrix = lp_model.all_scores(samples_matrix).detach() # TODO: Normalize matrix rows to unit norm
        # Naive policy, compute mean with corresponding tail values
        mean_score = np.mean([phrase_all_tails_matrix[i, sample[2]].item() for i, sample in enumerate(facts_list)])
        # Filter Lector map phrase -> relation, using some policy/threshold
        threshold = 0.5 # TODO: to set with a parameter
        if mean_score < threshold:
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
    kg = Dataset(name='FB15K-237', separator="\t", load=True)
    print('Dataset loaded.')
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
