# Selection Enhanced Lector, SELector (by Enrico Verdolotti)
# Questo modulo definisce l'oggetto SELector, un modello per l'estrazione di relazioni.
# I metodi esposti che si consiglia di usare sono:
# train(...) addestra il modello
# predict(...) predice un singolo fatto
# harvest(...) estrazione di tutti i fatti
# evaluate(...) valutazione prestazioni, precision e recall
# save_to_tsv(...) salva lo stato del modello in un tsv (non ancora implementato) 
# show_list(...) mostra il contenuto di una struttura dati interna o di una lista 

from os.path import sep
from text_normalization import parallel_phrases_normalizer, nltk_phrase_normalizer
import random
import csv

class SELector:
    
    def __init__(self, rseed=None, unlabeled_sub=0, no_types=False, text_norm=False, type_remapping=None):

        # Stato modello
        self.model_state = 'NOT READY'

        # Iperparametri        
        self.rseed = rseed  # random seed controlla la riproducibilità dei risultati      
        self.unlabeled_sub = unlabeled_sub  # unlabeled subsampling rateo [0, 1]
        self.no_types = no_types  # usa anche i tipi delle entità per addestrare il modello  
        self.text_norm = text_norm  # normalizza le frasi prima di addestrare il modello ed estrarre
        if type_remapping:
            self.no_types = False  # Specificare un riassegnamento forza l'utilizzo dei tipi
        self.type_remapping = type_remapping  # Riassegna i tipi in base ad un mapping (generalizza tipi)    
        
        # Strutture dati 
        self.text_triples = []
        self.knowledge_graph = []
        self.labeled_triples = []
        self.unlabeled_triples = []
        self.model_triples = []


    # Gestisce il caricamento da tsv, in una lista
    def load_tsv(self, file_path, dest):

        with open(file_path, 'r') as tsv_file:
            rd = csv.reader(tsv_file, delimiter='\t')
            for line in rd:                
                dest.append(tuple(line))
                

    # Genera le tabelle Labeled e Unlabeled Triples con Distant Supervision
    def distant_supervision(self):
        # Carica kg in struttura ottimizzata per ricerca
        kg_dict = dict() # NOTA: il KG potrebbe essere caricato direttamente in questa struttura        
        for h, r, t in self.knowledge_graph:
            try:
                kg_dict[(h,t)].append(r)
            except:
                kg_dict[(h,t)] = [r]
        # Iterazione a singolo ciclo for e accesso diretto
        for phr, e1, t1, e2, t2 in self.text_triples:
            # Riassegnazione tipi
            if self.type_remapping:
                try:
                    t1 = self.type_remapping[t1]
                    t2 = self.type_remapping[t2]
                except:
                    pass
            # Smistamento in labeled o unlabeled            
            try:
                for rel in kg_dict[(e1,e2)]:
                    triple = (phr, e1, t1, e2, t2, rel)
                    if self.no_types:
                        triple = (phr, e1, '', e2, '', rel)
                    self.labeled_triples.append(triple)
            except:
                triple = (phr, e1, t1, e2, t2, 'unknown')
                if self.no_types:
                    triple = (phr, e1, '', e2, '', 'unknown')
                self.unlabeled_triples.append(triple)
        # In effetti mantenere gli id delle entità non servirebbe per la fase successiva ma...
        # ...solo se la pipeline fosse quella originale, volendo usare Link Prediction gli id sono necessari. 


    # Costruisce la tabella necessaria per estrarre fatti
    def build_model_triples(self):

        # Dizionario da labeled e unlabeled {(phr, t1, t2) -> {rel1: count, rel1: count}}
        all_triples = self.labeled_triples + self.unlabeled_triples
        pattern2relc = dict()
        for phr, _, t1, _, t2, rel in all_triples:
            try:
                counts = pattern2relc[(phr, t1, t2)]
                try:
                    counts[rel] += 1
                except:
                    counts[rel] = 1
            except:
                pattern2relc[(phr, t1, t2)] = {rel: 1}

        # Max val key su ciascuna chiave per assegnare relazione
        for pattern, counts in pattern2relc.items():
            phr, t1, t2 = pattern
            rel = max(counts, key=counts.get) # Qui viene scelta la relazione con "max count" da associare al pattern
            if rel != 'unknown':
                triple = (phr, t1, t2, rel, counts[rel])
                self.model_triples.append(triple)

        # Ordina pattern per occorrenza decrescente
        self.model_triples.sort(key=lambda t: t[4], reverse=True)

    
    # Addestramento modello
    def train(self, input_text_triples, input_knowledge_graph):

        # Se viene passato un path carica da file altrimenti assegna
        if type(input_text_triples) is str:
            print('Caricamento text triples in corso...', end='', flush=True)
            self.load_tsv(input_text_triples, self.text_triples,)
            print('Fatto.', flush=True)
        else:
            self.text_triples = input_text_triples

        # Se viene passato un path carica da file altrimenti assegna
        if type(input_knowledge_graph) is str:
            print('Caricamento knowledge graph in corso...', end='', flush=True)
            self.load_tsv(input_knowledge_graph, self.knowledge_graph)
            print('Fatto.', flush=True)
        else:
            self.knowledge_graph = input_knowledge_graph

        # Distant Supervision (INSERIRE LINK PREDICTION QUI oppure...)
        print('Generazione training set con distant supervision...', end='', flush=True)
        self.distant_supervision()
        print('Fatto.', flush=True)

        # Sottocampionamento unlabeled triples (...INSERIRE LINK PREDICTION QUI)
        print('Sottocampionamento casuale delle triple non etichettate...', end='', flush=True)
        num_sample = int(len(self.unlabeled_triples) * self.unlabeled_sub)
        random.seed(self.rseed) # Deterministico, usa rseed=None per seed casuali
        self.unlabeled_triples = random.sample(self.unlabeled_triples, k=num_sample)
        print('Fatto.', flush=True)

        # Normalizzazione phrases
        if self.text_norm:
            print('Normalizzazione phrases (multiprocessing)...', end='', flush=True)
            self.labeled_triples = parallel_phrases_normalizer(self.labeled_triples, nltk_phrase_normalizer)
            self.unlabeled_triples = parallel_phrases_normalizer(self.unlabeled_triples, nltk_phrase_normalizer)
            print('Fatto.', flush=True)

        # Costruisce la tabella che serve per le predizioni
        print('Generazione delle triple del modello...', end='', flush=True)
        self.build_model_triples()
        print('Fatto.', flush=True)
        
        # Elimina le strutture ausiliarie
        #del(self.text_triples)
        #del(self.knowledge_graph)
        # TODO: cancellare tutto tranne model_triples?
        
        # Modello pronto per estrarre fatti
        self.model_state = 'READY'


    # Costruisce un indice hash su model triples
    def build_mt_map(self):
        
        mt_map = dict()
        for phr, t1, t2, r, c in self.model_triples:
            mt_map[(phr, t1, t2)] = (r, c)
        
        return mt_map
  

    # Metodo interno per l'estrazione di una relazione da testo
    def _predict(self, text_triple, mt_map):

        # Controllo stato del modello
        assert self.model_state == 'READY', 'Model not ready, please train the model.'

        # Spacchetta tripla
        phr, e1, t1, e2, t2 = text_triple
        
        # Controllo se modalità senza tipi è attiva
        if self.no_types:
            t1, t2 = '', ''

        # Riassegnazione tipi
        if self.type_remapping:
            try:
                t1 = self.type_remapping[t1]
                t2 = self.type_remapping[t2]
            except:
                pass            

        # Match ad accesso diretto (corrispondenza esatta del pattern)
        try:            
            rel, _ = mt_map[(phr, t1, t2)]
        except:
            rel = 'unknown'

        return (e1, rel, e2)


    # Estrazione di fatti da text_triples, keep_unknown = True 
    # genererà anche fatti con relazione 'unknown'
    def harvest(self, input_text_triples, keep_unknown=False):

        # Se passi un path carica da tsv altrimenti usa il riferimento
        if type(input_text_triples) is str:
            text_triples = list()
            print('Caricamento text triples...', end='', flush=True)
            self.load_tsv(input_text_triples, text_triples)
            print('Fatto.', flush=True)
        else:
            text_triples = input_text_triples

        # Controllo se modalità normalizzazione testo è attiva
        if self.text_norm:
            print('Normalizzazione phrases (multiprocessing)...', end='', flush=True)
            text_triples = parallel_phrases_normalizer(text_triples, nltk_phrase_normalizer)
            print('Fatto.', flush=True)

        # Iterazione su ogni tripla estratta dal testo
        mt_map = self.build_mt_map() # model triples map (indice hash su pattern)
        result = list()
        for triple in text_triples:
            fact = self._predict(triple, mt_map)
            if fact[1] != 'unknown' or keep_unknown:
                result.append(fact)

        return result


    # Metodo esposto per l'estrazione di singole relazioni
    def predict(self, text_triple):
        output = self.harvest([text_triple], keep_unknown=True)
        return output[0]


    # Valuta le prestazioni del modello, necessita di una ground truth
    def evaluate(self, input_text_triples, input_ground_truth, gt_map=None):
        print('*** Valutazione modello ***', flush=True)
        # Carica da disco se passi un path (text triples)
        if type(input_text_triples) == str:
            text_triples = list()
            print('Caricamento text triples in corso...', end='', flush=True)
            self.load_tsv(input_text_triples, text_triples)
            print('Fatto.', flush=True)
        else:
            text_triples = input_text_triples

        # Carica da disco se passi un path (ground truth)
        if type(input_ground_truth) == str:
            ground_truth = list()
            print('Caricamento ground truth in corso...', end='', flush=True)
            self.load_tsv(input_ground_truth, ground_truth)
            print('Fatto.', flush=True)
        else:
            ground_truth = input_ground_truth

        # Effettua le predizioni con il modello e valuta le prestazioni
        harvested = self.harvest(text_triples, keep_unknown=True)       
        pred_card, true_card, relevant_card = 0, 0, 0
        for i, gt in enumerate(ground_truth):
            # Valutazione con remapping di relazioni o senza
            if gt_map:
                try:
                    true_relations = gt_map[gt[0]]
                except:
                    true_relations = ['unknown']
            else:
                true_relations = [gt[0]]
            # Relazione estratta dal modello
            _, relation, _ = harvested[i]            
            # Se il fatto predetto non è unknown
            if relation != 'unknown':
                # Incrementa il conteggio delle relazioni estratte
                pred_card += 1
                # Se la relazione estratta è anche corretta
                if relation in true_relations:
                    # Incrementa il conteggio delle estrazioni corrette
                    true_card += 1
            # Conta le relazioni rilevanti nella ground truth
            if true_relations != ['unknown']:
                relevant_card += 1

        # Calcolo precision, recall ed fScore
        precision, recall, fscore = 0, 0, 0        
        try:
            precision = true_card / pred_card
            recall = true_card / relevant_card
            fscore =(precision*recall/(precision+recall))*2
        except:
            print('\nDivisione per 0, qualcosa non va bene:')
            print(f'Fatti estratti totali: {pred_card}')
            print(f'Fatti estratti corretti: {true_card}')
            print(f'Fatti rilevanti totali: {relevant_card}')

        return precision, recall, fscore


    # Salva il modello (model_triples)
    def save_to_tsv(self, save_path):

        # TODO: Da implementare salvataggio su file tsv di model_triples
        print('Not implemented yet.')


    # Utility, mostra il contenuto di una struttura dati
    def show_list(self, struct_name, outer=None, limit=100):

        print(f'\nShowing {struct_name}:')

        # Verifica se struttura interna al modello o esterna
        if outer != None:
            struct = outer                        
        else:
            struct = getattr(self, struct_name)

        # Stampa struttura in modo ordinato  
        for line in struct[:limit]:
            print(line)
    


## TEST AREA ## Esegui da shell per avviare questa demo
if __name__ == '__main__':

    # File esempio giocattolo
    toy_tt_path = 'data/toy_example/train/text_triples.tsv'
    toy_kg_path = 'data/toy_example/train/knowledge_graph.tsv'
    toy_gt_path = 'data/toy_example/test/text_triples_gt.tsv'  

    # Inizializza e addestra modello
    sel = SELector(rseed=42, unlabeled_sub=0.8)
    sel.train(toy_tt_path, toy_kg_path)

    # Mostra strutture dati dopo l'addestramento
    sel.show_list('text_triples')
    
    sel.show_list('knowledge_graph')

    sel.show_list('labeled_triples')

    sel.show_list('unlabeled_triples')
    
    sel.show_list('model_triples')

    # Test estrazione fatti
    harvested = sel.harvest(toy_tt_path)
    sel.show_list('harvested_triples', harvested)

    # Valutazione modello con ground truth
    precision, recall, fscore = sel.evaluate(toy_tt_path, toy_gt_path)

    print('\nValutazione sulle text triples usate per il training:')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'fScore: {fscore}')
