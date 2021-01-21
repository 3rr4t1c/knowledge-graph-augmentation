# Selection Enhanced Lector, SELector 
##### by Enrico Verdolotti
***
Questo progetto mira a migliorare le prestazioni di un progetto preesistente: Lector  
Tutta la pipeline è stata reimplementata, i metodi principali sono:

* `train(...)` addestra il modello
* `predict(...)` predice un singolo fatto
* `harvest(...)` estrazione di tutti i fatti
* `evaluate(...)` valutazione prestazioni, precision e recall
* `save_to_tsv(...)` salva lo stato del modello in un tsv (non ancora implementato) 
* `show_list(...)` mostra il contenuto di una struttura dati interna o di una lista
***
#### Demo
Eseguendo direttamente lo script `selector.py` si avvia una demo su piccoli file di esempio con sottocampionamento = 0.8
***
#### Requisiti
Eseguito su Python 3.7.7, non testato su versioni precedenti.
***
#### Utilizzo
Nella cartella contenente il file `selector.py` avviare il terminale Python:  
`>>> import selector`  
`>>> slc = selector.SELector()`  
Addestra il modello di default senza sottocampionamento di unlabeled triples.  
`>>> slc.train('input_data/text_triples.tsv', 'input_data/knowledge_graph.tsv')`    
Visualizza il contenuto delle strutture interne.  
`>>> slc.show_list('text_triples')`  
`>>> slc.show_list('knowledge_graph')`  
`>>> slc.show_list('labeled_triples')`  
`>>> slc.show_list('unlabeled_triples')`  
`>>> slc.show_list('model_triples')`  
Estrai nuovi fatti e mostra risultato:  
`>>> facts = slc.harvest('input_data/text_triples.tsv')`  
`>>> slc.show_list('harvested_triples', facts)`  
Valuta prestazioni (precision e recall):  
`>>> sample_text_triples = 'input_data/text_triples.tsv'     # utilizzo lo stesso file di training`  
`>>> groud_truth = 'input_data/text_triples_gt.tsv'`  
`>>> precs, recall = slc.evaluate(sample_text_triples, groud_truth)`  
`>>> print(f'Precision: {precs}\nRecall: {recall}')`  
`Precision: 1.0`  
`Recall: 0.5555555555555556`
