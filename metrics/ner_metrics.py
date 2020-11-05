import torch
from collections import Counter

def get_entities(seq,id2label,markup="bios"):
    assert markup in ["bios","bio"]
    if markup == "bios":
        return get_entity_bios(seq,id2label)
    
    else:
        return get_entity_bio(seq,id2label)


def get_entity_bios(seq,id2label):

    seq = [id2label[x] for x in seq]
    chunks = []
    chunk = [-1,-1,-1]

    for index, word in enumerate(seq):
        if word.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1,-1,-1]
            chunk[0] = word.split("-")[1]
            chunk[1] = index
            chunk[2] = index
            chunks.append(chunk)
            chunk = [-1,-1,-1]
        elif word.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk =[-1,-1,-1]
            chunk[0] = word.split("-")[1]
            chunk[1] = index
            chunk[2] = index
            if index == len(seq) -1:
                chunks.append(chunk)
        elif word.startswith("I-") and chunk[-1] != -1:
            _type = word.split("-")[1]
            if _type == chunk[0]:
                chunk[2] = index
            if index == len(seq) -1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk =[ -1,-1,-1]
        
    return chunks


def get_entity_bios(seq,id2label):

    seq = [id2label[x] for x in seq]
    chunks = []
    chunk = [-1,-1,-1]

    for index, word in enumerate(seq):
        if word.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk =[-1,-1,-1]
            chunk[0] = word.split("-")[1]
            chunk[1] = index
            chunk[2] = index
            if index == len(seq) -1:
                chunks.append(chunk)
        elif word.startswith("I-") and chunk[-1] != -1:
            _type = word.split("-")[1]
            if _type == chunk[0]:
                chunk[2] = index
            if index == len(seq) -1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk =[ -1,-1,-1]
        
    return chunks
 


class SeqEntityScore(object):
    def __init__(self,id2label,markup="bios"):
        self.id2label = id2label
        self.markup = markup
        self.origins = []
        self.founds =  []
        self.rights = []
    
    def update(self,label_paths,pred_paths):
        for l_path,p_path in zip(label_paths,pred_paths):
            label_entities = get_entities(l_path,self.id2label,self.markup)
            pred_entities = get_entities(p_path,self.id2label,self.markup)
            self.origins.extend(label_entities)
            self.founds.extend(pred_entities)
            self.rights.extend([pred_en for pred_en in pred_entities if pred_en in label_entities])
    
    def compute(self,origin,found,right):
        recall = right / origin if origin != 0 else 0.
        precision = right / found if found != 0 else 0.
        f1 = (2 * precision * recall) /(precision + recall) if precision + recall != 0 else 0.

        return recall,precision,f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter(x[0] for x in self.rights)

        for _type,origin in origin_counter.items():
            found = found_counter.get(_type,0)
            right = right_counter.get(_type,0)
            recall,precision,f1 = self.compute(origin,found,right)
            class_info[_type] = {"acc":round(precision,4),"recall":round(recall,4),"f1":round(f1,4)}

        origins = len(self.origins)
        founds = len(self.founds)
        rights = len(self.rights)

        recall,precision,f1 = self.compute(origins,founds,rights)

        return {"acc":precision,"recall":recall,"f1",f1},class_info


