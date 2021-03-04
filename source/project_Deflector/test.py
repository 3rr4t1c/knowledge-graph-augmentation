from selector import SELector
from deflector import Deflector
import random
import csv

# Legge un TSV come lista di tuple
def read_tsv(file_path):
    out = list()
    with open(file_path, 'r', encoding='utf8') as tsv_file:
        rd = csv.reader(tsv_file, delimiter='\t')
        for line in rd:                
            out.append(tuple(line))
    return out


def deflector_test(patterns, knowledge_graph, link_predictor='base', perfect_lp_knowledge=None):

    # Relation extraction model (Lector)
    print('Training relation extraction model...', flush=True)
    slc = SELector(rseed=42, unlabeled_sub=0.5)
    slc.train(patterns, knowledge_graph)

    print(f'\nTotal labeled: {len(slc.labeled_triples)}, examples:')
    [print(x) for x in slc.labeled_triples[:10]]

    print(f'\nTotal unlabeled: {len(slc.unlabeled_triples)}, examples:')
    [print(x) for x in slc.unlabeled_triples[:10]]

    print(f'\nModel triples: {len(slc.model_triples)}, examples:')
    [print(x) for x in slc.model_triples[:10]]

    # Relation extraction model wrapper
    def selector_re_wrapper(patternFile):
        input_ = read_tsv(patternFile)
        output = slc.harvest(input_, keep_unknown=True)
        # Costruisce l'output adatto per deflector
        deflist = list()
        for i, pattern_triple in enumerate(input_):
            phr, s, st, o, ot = pattern_triple
            predicted_relation = output[i][1]
            pattern = (phr, st, ot)
            pair = (s, o)
            deflist.append((pattern, predicted_relation, pair))

        return deflist

    # Load link prediction model
    if link_predictor == 'base':
        from link_prediction_perfect import LinkIndexer
        lp_model = LinkIndexer()
        print('\nLoading simple fact indexer as link prediction...', end='', flush=True)
        lp_model.train(read_tsv(perfect_lp_knowledge))
        print('Done.', flush=True)
        # Define wrapper
        def lp_wrapper(pairList):            
            return lp_model.predict(pairList)
    elif link_predictor == 'complex':
        print('\nLoading ComplEx as link predicion model...', flush=True)
        from link_prediction_complex import complex_wrapper
        lp_wrapper = complex_wrapper

    # Relation extraction meta-model (Deflector)
    dfl = Deflector(selector_re_wrapper, lp_wrapper)

    # Deflect patterns
    print('\nActivating Deflector system...', flush=True)
    dfl.deflect_patterns(patterns, pd_enabled=True)

    # Load and shuffle structures for examples
    discovered = list(dfl.pattern_discovery.items())
    random.shuffle(discovered)
    blacklisted = list(dfl.pattern_black_list.items())
    random.shuffle(blacklisted)
    
    # Show pattern discovery example
    print('\n\n\nPattern discovery size:', len(discovered))
    print('\nDiscovered patterns:')
    maxprint = 100
    for pattern, relScore in discovered:
        #if relScore[0] == 'spouse':        
        print(pattern, '-->', relScore)
        maxprint -= 1
        if maxprint == 0:
            break
    
    # Show pattern blacklist example
    print('\n\n\nPattern blacklist size:', len(blacklisted))
    print('\nBlacklisted "spouse" patterns:')
    maxprint = 100
    for pattern, relScore in blacklisted:
        if relScore[0] == 'spouse':        
            print(pattern, '-->', relScore)
            maxprint -= 1
        if maxprint == 0:
            break

    return discovered, blacklisted



# Utility for manual evaluation
def manual_eval(patterns):
    # Show how many patterns
    print(f'Total patterns: {len(patterns)}')
    print()
    # Show three times a random sample of 100 patterns
    for _ in range(3):
        samp = random.sample(patterns, k=100)
        [print(x) for x in samp]
        print()

def manual_eval_rel(patterns, relation):
    # Show how many patterns
    print(f'Total patterns: {len(patterns)}')
    print()
    relp = [x for x in patterns if x[1][0]==relation]
    print(f'Relation: "{relation}" total patterns: {len(relp)}')
    # Show three times a random sample of 100 patterns
    # for _ in range(3):
    #     random.shuffle(relp)
    #     [print(relp[n]) for n in range(100)]
    #     print()
    [print(x) for x in relp]
    print()


## TEST AREA ##
if __name__ == '__main__':

    import pickle

    # patterns = 'data/all_patterns_6.2M.tsv'

    # knowledge1 = 'data/kg_no_test_facts_degree_build.tsv'
    # knowledge2 = 'data/kg_degree_build.tsv'
    # knowledge3 = 'data/full_kg_subsampled_1.5M.tsv'

    # results = dict()

    # results['complexNoTestFacts'] = deflector_test(patterns, knowledge1, link_predictor='complex')
    # print()
    # results['baseNoTestFacts'] = deflector_test(patterns, knowledge1, link_predictor='base', perfect_lp_knowledge=knowledge1)
    # print()
    # results['baseWithTestFacts'] = deflector_test(patterns, knowledge1, link_predictor='base', perfect_lp_knowledge=knowledge2)    
    # print()
    # results['baseRandomSubsampling'] = deflector_test(patterns, knowledge3, link_predictor='base', perfect_lp_knowledge=knowledge3)
    # print()
    
    # # Save on disk
    # with open('results.pydict', 'wb') as result_file:
    #         pickle.dump(results, result_file)

    # Load from disk
    with open('results.pydict', 'rb') as result_file:
            results = pickle.load(result_file)

    #manual_eval_rel(results['baseRandomSubsampling'][1], 'spouse')
    manual_eval(results['complexNoTestFacts'][1])