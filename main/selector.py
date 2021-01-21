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
    
    def __init__(self):
        self.model_state = 'NOT READY'
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
        for txt, e1, t1, e2, t2 in self.text_triples:
            triple = None
            for h, r, t in self.knowledge_graph:                
                if h == e1 and t == e2:
                    triple = (txt, e1, t1, e2, t2, r)
                    self.labeled_triples.append(triple)
            if triple == None:
                triple = (txt, e1, t1, e2, t2, 'unknown')
                self.unlabeled_triples.append(triple)
        # In effetti mantenere gli id delle entità non servirebbe per la fase successiva...
        # ...solo se la pipeline fosse quella originale, volendo usare Link Prediction gli id sono necessari. 


    # Addestra il modello, la funzione di match deve sempre restituire un bool
    def build_model_triples(self, match_function=lambda x, y: x == y):
        # Unisci tutte le triple ed assegna un id incrementale
        all_triples = self.labeled_triples + self.unlabeled_triples        
        # Group by match
        while all_triples:
            # Ottiene tutta la lista tranne l'elemento affiorante
            tail_triples = all_triples[1:]
            # Rimuove l'elemento affiorante dalla lista all_triples
            head_triple = all_triples.pop(0)
            htxt, _, ht1, _, ht2, hrel = head_triple
            # Relazione: numero di occorrenze per la phrase corrente
            rel2count = {hrel: 1}
            # Itera sul resto della lista
            for triple in tail_triples:
                txt, _, t1, _, t2, rel = triple
                # Solo se i tipi corrispondono esattamente
                if t1 == ht1 and t2 == ht2:
                    # Caso1: match esatto della Phrase (originale)
                    if txt == htxt:
                        # Incrementa il conteggio per la relazione associata
                        # e rimuove la tripla corrente da all_triples
                        try:
                            rel2count[rel] += 1
                        except:
                            rel2count[rel] = 1                                                
                        all_triples.remove(triple)                         
                    # Caso2: match inesatto della Phrase (sperimentale, disattivato di default)
                    elif match_function(txt, htxt):
                        # Incrementa il conteggio per la relazione associata 
                        # la tripla viene lasciata nella lista per successive iterazioni
                        try:
                            rel2count[rel] += 1
                        except:
                            rel2count[rel] = 1
            # Estrazione della relazione con conteggio maggiore ed aggiornamento modello
            max_rel = max(rel2count, key=rel2count.get)
            if max_rel != 'unknown':
                self.model_triples.append((htxt, ht1, ht2, max_rel, rel2count[max_rel]))
            # TODO: Gestire il caso di conteggi "pari merito" con unknown, risultato non deterministico

        self.model_triples.sort(key=lambda t: t[4], reverse=True) # Sort descending by count

    
    # Addestra il modello con Distant supervision
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
        self.distant_supervision()        
        
        # Sottocampionamento unknown o LP
        num_sample = int(len(self.unlabeled_triples) * subsamp)
        random.seed(rseed) # Deterministico, usa rseed=None per seed casuali
        self.unlabeled_triples = random.sample(self.unlabeled_triples, k=num_sample)
        
        # Costruisce la tabella che serve per le predizioni
        self.build_model_triples()
        
        # Elimina le strutture ausiliarie
        #del(self.text_triples)
        #del(self.knowledge_graph)
        # TODO: cancellare tutto tranne model_triples?
        
        # Modello pronto per estrarre fatti
        self.model_state = 'READY'


    # Singola predizione di un fatto a partire da una tripla estratta dal testo
    def predict(self, text_triple, match_function=lambda x, y: x == y):
        assert self.model_state == 'READY'
        txt, e1, t1, e2, t2 = text_triple
        inexact_matches = dict()
        for mtxt, mt1, mt2, mr, _ in self.model_triples:                
            if mt1 == t1 and mt2 == t2:
                # Caso 1: Match esatto (originale)
                if mtxt == txt:
                    return (e1, mr, e2)       
                # Caso 2: Match inesatto (sperimentale, disattivato di default)
                elif match_function(mtxt, txt):
                    try:
                        inexact_matches[mr] += 1
                    except:
                        inexact_matches[mr] = 1
        
        if inexact_matches:
            max_rel = max(inexact_matches, key=inexact_matches.get)            
            return (e1, max_rel, e2)

        # Restituisce unknown se falliscono tutti i controlli precedenti
        return (e1, 'unknown', e2)   # NOTA: questo tipo di fatto è utile in certi casi


    # Estrazione fatti, i fatti nella forma (e1, unknown, e2) vengono scartati
    def harvest(self, input_text_triples, match_function=lambda x, y: x == y):
        assert self.model_state == 'READY'

        # Se passi un path carica da tsv altrimenti usa il riferimento
        if type(input_text_triples) is str:
            text_triples = list()
            self.load_tsv(input_text_triples, text_triples)
        else:
            text_triples = input_text_triples

        # Iterazione su ogni tripla estratta dal testo
        result = list()
        for triple in text_triples:
            fact = self.predict(triple, match_function)
            if fact[1] != 'unknown':
                result.append(fact)

        return result


    # Valuta le prestazioni del modello, necessita di una ground truth
    def evaluate(self, input_text_triples, input_ground_truth):
        assert self.model_state == 'READY'

        # Carica da disco se passi un path
        if type(input_text_triples) == str:
            text_triples = list()
            self.load_tsv(input_text_triples, text_triples)
        else:
            text_triples = input_text_triples

        # Carica da disco se passi un path
        if type(input_ground_truth) == str:
            ground_truth = list()
            self.load_tsv(input_ground_truth, ground_truth)
        else:
            ground_truth = input_ground_truth

        # Effettua le predizioni con il modello
        pred_card, true_card, relevant_card = 0, 0, 0        
        for i, txt_triple in enumerate(text_triples):
            _, relation, _ = self.predict(txt_triple)
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

        precision = true_card / pred_card
        recall = true_card / relevant_card

        return precision, recall


    # Salva il modello (model_triples)
    def save_to_tsv(self, save_path):
        # TODO: Da implementare salvataggio su file tsv di model_triples
        print('Not implemented yet.')


    # Utility, mostra il contenuto di una struttura dati
    def show_list(self, struct_name, outer=None, limit=100):
        print(f'\nShowing {struct_name}:')
        if outer:
            struct = outer
        else:
            struct = getattr(self, struct_name)
        for line in struct[:limit]:
            print(line)
    


## TEST AREA ## Esegui da shell per avviare questa demo
if __name__ == '__main__':

    sel = SELector()
    sel.train(f'input_data{sep}text_triples.tsv', 
              f'input_data{sep}knowledge_graph.tsv', 
              subsamp=0.8)

    sel.show_list('text_triples')
    
    sel.show_list('knowledge_graph')

    sel.show_list('labeled_triples')

    sel.show_list('unlabeled_triples')
    
    sel.show_list('model_triples')

    sel.show_list('harvested_triples', sel.harvest(sel.text_triples))

    precision, recall = sel.evaluate(sel.text_triples, f'input_data{sep}text_triples_gt.tsv')
    print('\ntest sui dati di training')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'fScore: {(precision * recall / (precision + recall)) * 2} ')
