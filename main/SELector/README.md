# SELector (Second Edition Lector)
###### by Enrico Verdolotti
*** 
Tutta la pipeline è stata reimplementata, i metodi principali sono:

* `train(...)` addestra il modello
* `predict(...)` predice un singolo fatto
* `harvest(...)` estrazione di tutti i fatti
* `evaluate(...)` valutazione prestazioni, precision e recall
* `save_to_tsv(...)` salva lo stato del modello in un tsv (non ancora implementato) 
* `show_list(...)` mostra il contenuto di una struttura dati interna o di una lista
***
#### Demo
Eseguendo da terminale `python demo.py` (o `python3 demo.py` se entrambe le versioni di python sono installate) si avvia una demo su dati reali. 
Eseguendo da terminale `python selector.py` si avvia un test su piccoli file di esempio con sottocampionamento = 0.8
***
#### Requisiti
Eseguito su Python 3.7.7, non testato su versioni precedenti.
Package `pandas>=1.1.5`  (`pip install pandas` o `pip3 install pandas` o `conda install pandas`) 
***
#### Esempio utilizzo
Nella cartella contenente il file `selector.py` avviare il terminale Python:  
`>>> import selector`  
`>>> slc = selector.SELector(unlabeled_sub=0.6)`  
Addestra il modello di default utilizzando il 60% delle unlabeled triples.  
`>>> slc.train('data/toy_example/train/text_triples.tsv', 'data/toy_example/train/knowledge_graph.tsv')`    
Visualizza il contenuto delle strutture interne.  
`>>> slc.show_list('text_triples')`  
`>>> slc.show_list('knowledge_graph')`  
`>>> slc.show_list('labeled_triples')`  
`>>> slc.show_list('unlabeled_triples')`  
`>>> slc.show_list('model_triples')`  
Estrai nuovi fatti e mostra risultato:  
`>>> facts = slc.harvest('data/toy_example/train/text_triples.tsv')`  
`>>> slc.show_list('harvested_triples', facts)`  
Valuta prestazioni (precision e recall):  
`>>> sample_text_triples = 'data/toy_example/train/text_triples.tsv'     # utilizzo lo stesso file di training`  
`>>> groud_truth = 'input_data/text_triples_gt.tsv'`  
`>>> precision, recall, fscore = slc.evaluate(sample_text_triples, groud_truth)`  

