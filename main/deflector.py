from tqdm import tqdm
import numpy as np

# Deflector è un meta-sistema per la relation extraction
# che può sfruttare un qualsiasi sistema di link prediction
# per migliorare un qualunque sistema di relaction extraction 
class Deflector:

    # Inizializzazione: 
    # re_wrapper_fun: è una funzione che può essere definita arbitrariamente
    # al suo interno, riceve come input lo stesso input del modello di RE
    # ma deve restituire l'input formattato come lista di triple: [(e1, pattern0, e2)] 
    # e l'output come lista di relazioni: [relation0] entrambe della stessa dimensione
    # lp_wrapper_fun: è una funzione che può essere definita arbitrariamente
    # al suo interno ma deve prendere come input una lista di istanze [(e1, e2)]
    # e deve restituire [{top relations for e1 e2}]
    # PRE: le dimensioni di input ed output devono essere identiche ed associare alle
    # le relazioni che non possono essere estratte o predette la stringa 'unknown'  
    def __init__(self, re_wrapper_fun, lp_wrapper_fun):
        self.relation_extraction_model = re_wrapper_fun
        self.link_prediction_model = lp_wrapper_fun
        self.pattern_black_list = dict() # {pattern: (old_relation, score)}
        self.pattern_discovery = dict()  # {pattern: (new_relation, score)} # solo per ex relation unknown


    # Applica i filtri sull'output invece che sull'input
    def extract_relations(self, raw_re_input, keep_unknown=False, keep_patterns=False, keep_pairs=False):
        
        # Input ed Output formattati del modello relation extraction
        re_output = self.relation_extraction_model(raw_re_input)

        # Filtra l'output di relation extraction 
        output = list()
        for pattern, p_relation, pair in re_output:
            sbj, obj = pair
            if p_relation != 'unknown':
                try:
                    self.pattern_black_list[pattern]
                    output.append((pattern, (sbj, 'unknown', obj)))
                except:
                    output.append((pattern, (sbj, p_relation, obj)))
            else:
                try:
                    new_rel = self.pattern_discovery[pattern][0]
                    output.append((pattern, (sbj, new_rel, obj)))
                except:
                    output.append((pattern, (sbj, p_relation, obj)))

        # Se si vogliono escludere fatti con relazione 'unknown'
        if not keep_unknown:
            output = [x for x in output if x[1][1] != 'unknown']

        # Se non si vogliono tenere le coppie anche nell'output
        if not keep_pairs:
            output = [(x[0], x[1][1]) for x in output]

        # Non tiene traccia del pattern che ha predetto il fatto
        if not keep_patterns:
            output = [x[1] for x in output]            

        return output


    # Da dizionario: {pattern: (relation, [(subject, object), ...]), ...}
    # A dizionario: {(pair): {set predizioni}}
    def batch_predict(self, pattern2relpairs, link_predictor, batch_size):

        # Init output structure
        all_pairs = set()
        print('\nFinding all unique pairs to be predicted...')
        for _, pairs in tqdm(pattern2relpairs.values()):
            for pair in pairs:
                all_pairs.add(pair)

        # Batch predict
        all_pairs = list(all_pairs)
        result = list()
        print('\nBatch predicting relations...')
        for i in tqdm(range(0, len(all_pairs), batch_size)):
            result += zip(all_pairs[i:i+batch_size], link_predictor(all_pairs[i:i+batch_size]))

        return dict(result)


    # Identifica pattern deboli e li aggiunge alla black list
    def deflect_patterns(self, raw_re_input, bs=5000, bl_min_len=5, bl_thresh=0.7, pd_min_len=10, pd_thresh=0.7, pd_enabled=False):

        ex_trace = self.extract_relations(raw_re_input, keep_unknown=True, keep_patterns=True, keep_pairs=True)

        # Costruisce un dizionario {pattern: (relation, [(subject, object), ...]), ...}
        pattern2instances = dict()
        for pattern, opinion in ex_trace:
            try:
                sbj, _, obj = opinion
                pattern2instances[pattern][1].append((sbj, obj))
            except:
                sbj, relation, obj = opinion 
                pattern2instances[pattern] = (relation, [(sbj, obj)])

        # Build dictionary {pair: predicted relations as a set}
        pair2pred = self.batch_predict(pattern2instances, self.link_prediction_model, batch_size=bs)

        print('\nDeflecting patterns...')
        for pattern, track in tqdm(pattern2instances.items()):
            relation, pairs = track            
            predictions = [pair2pred[pair] for pair in pairs if 'unknown' not in pair2pred[pair]]
            preds_len = len(predictions)
            if relation != 'unknown' and preds_len >= bl_min_len:
                matches = [relation in pred for pred in predictions]                
                pattern_score = sum(matches)/len(matches)
                # Pattern blacklist
                if pattern_score < bl_thresh:                    
                    self.pattern_black_list[pattern] = (relation, pattern_score)            
            elif pd_enabled and preds_len >= pd_min_len:
                all_pred_size = [len(pred) for pred in predictions]
                rel2weight = dict()
                for n, preds in enumerate(predictions):
                    for rel in preds:
                        try:
                            rel2weight[rel] += 1/all_pred_size[n]
                        except:
                            rel2weight[rel] = 1/all_pred_size[n]
                
                max_rel1 = max(rel2weight, key=rel2weight.get)
                max_wgh1 = rel2weight[max_rel1]
                del(rel2weight[max_rel1])
                try:                        
                    max_rel2 = max(rel2weight, key=rel2weight.get)
                    max_wgh2 = rel2weight[max_rel2]
                except:
                    max_wgh2 = 0

                max_rel_score = (max_wgh1 - max_wgh2)/len(pairs)
                # Pattern discovery
                if max_rel_score >= pd_thresh:
                    self.pattern_discovery[pattern] = (max_rel1, max_rel_score)
