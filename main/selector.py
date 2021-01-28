# Selection Enhanced Lector, SELector (by Enrico Verdolotti)
# Questo progetto mira a migliorare le prestazioni di un progetto preesistente: Lector
# Tutta la pipeline è stata reimplementata, i metodi esposti che si consiglia di usare sono:
# train(...) addestra il modello
# predict(...) predice un singolo fatto
# harvest(...) estrazione di tutti i fatti
# evaluate(...) valutazione prestazioni, precision e recall
# save_to_tsv(...) salva lo stato del modello in un tsv (non ancora implementato) 
# show_list(...) mostra il contenuto di una struttura dati interna o di una lista 

from os.path import sep
import random

class SELector:
    
    def __init__(self, notypes=False):
        self.model_state = 'NOT READY'
        self.notypes = notypes # Booleano: True per non considerare i tipi delle entità
        self.text_triples = []
        self.knowledge_graph = []
        self.labeled_triples = []
        self.unlabeled_triples = []
        self.model_triples = []


    # Gestisce il caricamento da tsv, in una lista
    def load_tsv(self, file_path, dest):
        with open(file_path, 'r') as tsv_file:
            for line in tsv_file:                               
                field = tuple(field for field in line.strip().split(sep='\t'))
                dest.append(field)
                

    # Effettua il distant supervision
    def distant_supervision(self):
        # Versione improponibile con costo O(tt.size x kg.size)
        # for txt, e1, t1, e2, t2 in self.text_triples:
        #     triple = None
        #     for h, r, t in self.knowledge_graph:                
        #         if h == e1 and t == e2:
        #             triple = (txt, e1, t1, e2, t2, r)
        #             self.labeled_triples.append(triple)
        #     if triple == None:
        #         triple = (txt, e1, t1, e2, t2, 'unknown')
        #         self.unlabeled_triples.append(triple)
        
        # Versione ottimizzata con costo O(tt.size + kg.size) :)
        kg_dict = dict() # TODO: il kg può essere caricato direttamente in questa struttura
        # Carica kg in struttura ottimizzata per ricerca
        for h, r, t in self.knowledge_graph:
            try:
                kg_dict[(h,t)].append(r)
            except:
                kg_dict[(h,t)] = [r]
        # Iterazione a singolo ciclo for e accesso diretto
        for phr, e1, t1, e2, t2 in self.text_triples:
            try:
                for rel in kg_dict[(e1,e2)]:
                    triple = (phr, e1, t1, e2, t2, rel)
                    if self.notypes:
                        triple = (phr, e1, '', e2, '', rel)
                    self.labeled_triples.append(triple)
            except:
                triple = (phr, e1, t1, e2, t2, 'unknown')
                if self.notypes:
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
    def train(self, input_text_triples, input_knowledge_graph, subsamp=1, rseed=42):        
        # Se viene passato un path carica da file altrimenti assegna
        if type(input_text_triples) is str:
            self.load_tsv(input_text_triples, self.text_triples,)
        else:
            self.text_triples = input_text_triples
        # Se viene passato un path carica da file altrimenti assegna
        if type(input_knowledge_graph) is str:
            self.load_tsv(input_knowledge_graph, self.knowledge_graph)
        else:
            self.knowledge_graph = input_knowledge_graph
        
        # Distant Supervision
        print('Generando training set con distant supervision...', end='', flush=True)
        self.distant_supervision()
        print('Fatto.', flush=True)        
        
        # Sottocampionamento unknown o LP
        print('Sottocampionamento casuale delle triple non etichettate...', end='', flush=True)
        num_sample = int(len(self.unlabeled_triples) * subsamp)
        random.seed(rseed) # Deterministico, usa rseed=None per seed casuali
        self.unlabeled_triples = random.sample(self.unlabeled_triples, k=num_sample)
        print('Fatto.', flush=True)
        
        # Costruisce la tabella che serve per le predizioni
        print('Generando le triple del modello per le estrazioni...', end='', flush=True)
        self.build_model_triples()
        print('Fatto.', flush=True)
        
        # Elimina le strutture ausiliarie
        #del(self.text_triples)
        #del(self.knowledge_graph)
        # TODO: cancellare tutto tranne model_triples?
        
        # Modello pronto per estrarre fatti
        self.model_state = 'READY'


    # Costruisce un dizionario pattern -> relazione per match rapidi
    def build_mt_map(self):
        mt_map = dict()
        for phr, t1, t2, r, c in self.model_triples:
            mt_map[(phr, t1, t2)] = (r, c)
        
        return mt_map
  

    # Singola predizione di un fatto a partire da una tripla estratta dal testo
    def predict(self, text_triple, use_map=None):
        # Controllo stato del modello
        assert self.model_state == 'READY'
        # Spacchetta tripla
        phr, e1, t1, e2, t2 = text_triple
        # Controllo se modalità senza tipi è attiva
        if self.notypes:
            t1, t2 = '', ''
        # Controllo se struttura dati per match passata
        if use_map:
            model_triples_map = use_map
        else:
            model_triples_map = self.build_mt_map()
        # Match ad accesso diretto (corrispondenza esatta del pattern)
        try:            
            rel, _ = model_triples_map[(phr, t1, t2)]
        except:
            rel = 'unknown'

        return (e1, rel, e2)


    # Estrazione fatti, N.B. i fatti nella forma (e1, unknown, e2) non vengono prodotti
    def harvest(self, input_text_triples):
        # Controllo stato del modello
        assert self.model_state == 'READY'
        # Se passi un path carica da tsv altrimenti usa il riferimento
        if type(input_text_triples) is str:
            text_triples = list()
            self.load_tsv(input_text_triples, text_triples)
        else:
            text_triples = input_text_triples
        # Iterazione su ogni tripla estratta dal testo
        mt_map = self.build_mt_map()
        result = list()
        for triple in text_triples:
            fact = self.predict(triple, mt_map)
            if fact[1] != 'unknown':
                result.append(fact)

        return result


    # Valuta le prestazioni del modello, necessita di una ground truth
    def evaluate(self, input_text_triples, input_ground_truth):
        # Controllo stato del modello
        assert self.model_state == 'READY'
        # Carica da disco se passi un path (text triples)
        if type(input_text_triples) == str:
            text_triples = list()
            self.load_tsv(input_text_triples, text_triples)
        else:
            text_triples = input_text_triples
        # Carica da disco se passi un path (knowledge graph)
        if type(input_ground_truth) == str:
            ground_truth = list()
            self.load_tsv(input_ground_truth, ground_truth)
        else:
            ground_truth = input_ground_truth
        # Effettua le predizioni con il modello e valuta prestazioni
        mt_map = self.build_mt_map()
        pred_card, true_card, relevant_card = 0, 0, 0        
        for i, txt_triple in enumerate(text_triples):
            _, relation, _ = self.predict(txt_triple, mt_map)
            gt_relation = ground_truth[i][0]
            # Se il fatto predetto non è unknown
            if relation != 'unknown':
                # Incrementa il numero di predizioni totali
                pred_card += 1
                # Se inoltre la predizione è corretta
                if relation == gt_relation:
                    # Incrementa il numero di predizioni corrette
                    true_card += 1
            # Conta le relazioni rilevanti nella ground truth
            if gt_relation != 'unknown':
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
    toy_tt = f'input_data/toy_example/text_triples.tsv'
    toy_kg = f'input_data/toy_example/knowledge_graph.tsv'

    # Files esempio con dati reali estratti da Lector
    mini_lector_kg = 'input_data/lector_example/knowledge_graph.tsv'
    mini_lector_tt = 'input_data/lector_example/text_triples.tsv'

    sel = SELector(notypes=True) # Modello inizializzato in modalità senza controllo dei tipi
    sel.train(toy_tt, toy_kg, subsamp=0.8) # Sottocampionamento delle unlabeled a 0.8

    # Mostra strutture dati dopo l'addestramento
    sel.show_list('text_triples')
    
    sel.show_list('knowledge_graph')

    sel.show_list('labeled_triples')

    sel.show_list('unlabeled_triples')
    
    sel.show_list('model_triples')

    # Test estrazione fatti
    harvested = sel.harvest('input_data/toy_example/text_triples.tsv')
    sel.show_list('harvested_triples', harvested)

    # Valutazione modello con ground truth
    precision, recall, fscore = sel.evaluate('input_data/toy_example/text_triples.tsv', 
                                             'input_data/toy_example/text_triples_gt.tsv')
    print('\nValutazione sulle text triples usate per il training:')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'fScore: {fscore}')
