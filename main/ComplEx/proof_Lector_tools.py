import numpy as np
import random
from dataset import Dataset
import requests
from bs4 import BeautifulSoup
import string

# Choosen relations (must match with existing ones)
EXAMPLE_TARGET_RELATIONS = ['/people/person/spouse_s./people/marriage/spouse',
                            '/base/locations/continents/countries_within' ]

# Example map. phrase -> (0 if bad phrase / 1 if good phrase, index of target relation)
EXAMPLE_PHRASE2RELATIONS = {'married on': (1, 0),
                            'met': (0, 0),
                            'married to': (1, 0),
                            'become engaged': (0, 0), 
                            'was born in': (1, 1),
                            'worked in': (0, 1),
                            'lived in': (0, 1),
                            'born in': (1, 1)}
# This map act like a hint just for build the phrase2facts map. 
# In real cases I don't know if a phrase is good or bad.


# Load from file all facts or only facts relative to specified relations
def load_relation2instances(facts_file, relations=None):
    # Read facts file selecting all target relations facts
    relation2instances = dict() # Low redundancy structure for storing facts by relations
    with open(facts_file, 'r') as f:
        for line in f:
            h, r, t = line.strip().split()
            # If relations is empty list will keep all relations
            if not relations or r in relations:
                try:
                    relation2instances[r].append((h, t))
                except:
                    relation2instances[r] = []
                    relation2instances[r].append((h, t))

    return relation2instances


# Generate phrase2relation map
# List with all relations as a string
# Minimum number of phrases per relation
# Maximum number of phrases per relation
# Minimum possible noise probability
# Maximum possible noise probability
def generate_phrase2relation(relations, min_phrases=10, max_phrases=10, min_noise=0.3, max_noise=0.5):
    phrase2relation = dict()
    
    for rel in relations:
        # Number of phrases to generate for current relation
        nphrases = random.randint(min_phrases, max_phrases)
        # Choose probability of getting bad phrases for current relation
        noise_prob = random.uniform(min_noise, max_noise) # Noise is a feature of each relation
        for _ in range(nphrases):
            # Random string of char with replacement
            phrase = ''.join(random.choices(string.ascii_letters + ' ', k=10))
            # Random probability of good phrase
            phrase_score = 0 if random.random() < noise_prob else 1
            # Add the phrase -> (bad/good, relation)
            phrase2relation[phrase] = (phrase_score, rel)

    return phrase2relation


# Generate a list of facts (triples) of same relation, 
# sampling from list of instances, then add some corrupted facts.
# relation is a string, instance is a list, true_amount and corrupted_amount are integers
def generate_facts(relation, instances, true_amount, corrupted_amount):
    # List with all correct facts as triples
    facts = [(h, relation, t) for h, t in random.sample(instances, true_amount)]
    # Generate corrupted facts
    heads, tails = zip(*instances)
    heads = random.sample(heads, corrupted_amount)
    tails = random.sample(tails, corrupted_amount)
    corrupted_facts = [(h, relation, t) for h, t in zip(heads, tails)]    
    # Merge all facts
    facts += corrupted_facts
    random.shuffle(facts) 
    
    return facts


# Generate phrase -> list of fact mapping simulating the output of Lector in phrase tracking mode
# phrase2relation is a map where: phrase -> (0 if bad phrase / 1 if good phrase, relation)
# relation2instances is a map used as a ground truth of real facts 
def generate_phrase2facts(phrase2relation, relation2instances):

    phrase2facts = dict()
    
    # Randomly distribute facts (true positive) along all matching phrases with duplicates
    # Inject some random false facts generated by shuffling heads and tails of same relation
    for phrase, (phrase_score, relation) in phrase2relation.items():        
        # All the instances of current relation
        instances = relation2instances[relation]
        # Current relation cardinality
        rel_card = len(instances)
        # Total number of facts to generate
        alpha = random.uniform(0.04, 0.06)
        facts_card = rel_card/(1+alpha*np.sqrt(rel_card))
        # True facts ratio. Testing beta distribution 
        # Assumption: this ratio is always 1 when a phrase is correct
        true_facts_ratio = 1 if phrase_score == 1 else random.betavariate(3, 4)
        # Number of true facts (e.g. Barack spouse Michelle) to generate
        true_amount = int(facts_card * true_facts_ratio)
        # Number of corrupted facts (e.g. Barack spouse Trump) to generate
        corrupted_amount = int(facts_card - true_amount)
        # Generate a list of facts associated to the current phrase
        phrase2facts[phrase] = generate_facts(relation, instances, true_amount, corrupted_amount)
    
    return phrase2facts


## UTILITY SECTION ##

# Print some info about the relations inside a fact
# Total amount of facts about a relation and percentage over all facts.
def facts_file_info(facts_file, top=3, target_relations=None):

    relation2info = dict()
    facts_counter = 0    
    with open(facts_file, 'r') as f:
        for line in f:            
            _, r, _ = line.strip().split()
            facts_counter += 1
            try:
                # relation2info: relation -> (relation counter, percent over all facts)
                rcount = relation2info[r][0] + 1
                relation2info[r] = (rcount, (rcount / facts_counter) * 100)
            except:
                # Add key
                relation2info[r] = (1, (1 / facts_counter) * 100)
    
    # Sort the relation2info dict as an associative list by percentage
    relations_ranking = sorted((list)(relation2info.items()), key=lambda x: x[1], reverse=True)

    # Pretty print the top X ranking
    for r, (n, p) in relations_ranking[:top]:
        print(f'"{r}" Has: {n} facts, and is: {round(p, 1)}% of total.')
    print(f'Facts amount: {facts_counter}')

    # Print info about target relations if any
    if target_relations:
        for rel in target_relations:
            print(f'{rel} -> {relation2info[rel]}')


# Return the real name of and entity by FreeBase mID
def mID_resolve(mID):
    r = requests.get('https://cofactor.io' + mID)
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup.title.get_text()


# Require a kg to encode
def facts2samples_encoder(facts, kg):
    samples = []
    for h, r, t in facts:
        samples.append((kg.entity_name_2_id[h], kg.relation_name_2_id[r], kg.entity_name_2_id[t]))
    return samples



## TEST AREA ##
if __name__ == '__main__':
    # Pretty printer for dict
    import pprint
    pp = pprint.PrettyPrinter()

    # Show info of highest cardinality relations
    facts_file_info('data/FB15k-237/test.txt', top=10, target_relations=EXAMPLE_TARGET_RELATIONS)
    print()        

    # Load test facts for ground truth about some target relations (None=All relations)
    GT_facts = load_relation2instances('data/FB15k-237/test.txt', EXAMPLE_TARGET_RELATIONS)
    print('Ground truth:')
    pp.pprint(GT_facts)
    print()
        
    # Generating Lector state (aka Lector train mode)
    phrase2relation = generate_phrase2relation(GT_facts.keys())
    print('Phrase -> Relation')
    pp.pprint(phrase2relation)
    print()

    # Generating Lector phrase to facts mapping (aka Lector phrase tracking mode)
    phrase2facts = generate_phrase2facts(phrase2relation, GT_facts)
    print('Phrase -> List of Facts')
    pp.pprint(phrase2facts)
    print()
    
    # Load dataset
    print('Loading dataset...', end='', flush=True)
    kg = Dataset(name='FB15k-237')
    print('Done.')

    # Encoding facts using Dataset known IDs
    print('Encoding facts to samples...', end='', flush=True)
    phrase2samples = {phrase: facts2samples_encoder(facts, kg) for phrase, facts in phrase2facts.items()}
    print('Done.')    
    print('Phrase -> List of Facts')
    pp.pprint(phrase2samples)
    