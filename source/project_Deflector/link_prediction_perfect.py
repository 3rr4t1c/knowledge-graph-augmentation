# Se si avessero abbastanza fatti un sistema di LP potrebbe
# essere implementato con la sola ricerca in un KG
import random

class LinkIndexer:

    def train(self, kg):
        self.kg = dict()
        for h, r, t in kg:
            try:
                self.kg[(h, t)].add(r)
            except:
                self.kg[(h, t)] = set([r])

    def predict(self, pairList):
        out = list()
        for h, t in pairList:
            try:
                out.append(self.kg[(h, t)])
            except:
                out.append(set(['unknown']))

        return out



