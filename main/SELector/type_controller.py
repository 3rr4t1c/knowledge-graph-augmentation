# Modulo per la gestione dei tipi
import csv


# Caricamento file tsv da disco
def load_tsv(file_path):

    output = list()    
    with open(file_path, 'r') as tsv_file:
        rd = csv.reader(tsv_file, delimiter='\t')
        for line in rd:                
            output.append(tuple(line))

    return output


# A partire da una gerarchia di tipi [(tipo child, tipo parent)]
# e.g. una lista di coppie [(SportAthlete, Person), ...]
# Costruisce un dizionario: {tipo: [path fino al tipo più generico]}
# e.g. {Poet: ['Poet', 'Writer', 'Person', 'Agent']}
def build_hierarchy_map(hierarchy):

    hmap = dict()
    for child, parent in hierarchy:
        # aggiungi arco
        try:
            hmap[child] = [child] + hmap[parent]
        except:
            hmap[child] = [child, parent]
            hmap[parent] = [parent]

        for path in hmap.values():
            last = path[-1]
            try:
                path += hmap[last][1:]
            except:
                pass

    return hmap


# Restituisce una riassegnazione dei tipi in base a certi parametri
# h_level regola la specificità del tipo che si vuole ottenere, in modo generico:
# e.g. Poet: ['Poet', 'Writer', 'Person', 'Agent'] se h_level=-1, Poet diventa Agent
# starred permette di indicare dei tipi preferiti, se il tipo corrente ne è un sottotipo
# e.g. Poet: ['Poet', 'Writer', 'Person', 'Agent'] se starred=['Person'] Poet diventa Person a prescindere
def types_remap(hierarchy_filepath, h_level=-1, starred_types=None):

    hierarchy = load_tsv(hierarchy_filepath)
    hmap = build_hierarchy_map(hierarchy)

    remap = dict()
    for typ, branch in hmap.items():
        if not starred_types:
            try:
                remap[typ] = branch[h_level]
            except:
                remap[typ] = typ
        else:
            starred = False            
            for desc in reversed(branch):
                if desc in starred_types:
                    remap[typ] = desc
                    starred = True
                    break
            if not starred:
                try:
                    remap[typ] = branch[h_level]
                except:
                    remap[typ] = typ

    return remap



## TEST AREA ##
if __name__ == '__main__':
    
    hierarchy = load_tsv('data/dbpedia_stuff/types_hierarchy.tsv')

    hmap = build_hierarchy_map(hierarchy)

    for k, v in hmap.items():
        print(f'{k}: {v}')


    remap1 = types_remap('data/dbpedia_stuff/types_hierarchy.tsv', -1, ['Person'])

    for k, v in remap1.items():
        print(f'{k}: {v}')
