
import csv
import json
import os
import torch
import copy
from transformers import BertTokenizer
import logging
logger = logging.getLogger(__name__)


class DataProcess:
    """base class for data converters for sequence classification datas sets."""
    def get_train_examples(self,data_dir):
        raise NotImplementedError()

    def get_dev_examples(self,data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()


    @classmethod
    def _read_tsv(cls,input_file,quotechar=None):
        with open(input_file,"r",encoding="utf-8-sig") as f:
            reader = csv.reader(f,delimeter="\t",quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_txt(cls,input_file):
        lines = []
        with open(input_file,"r") as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words":words,"labels":labels})
                        words = []
                        labels = []

                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) >1:
                        labels.append(splits[-1].replace("\n",""))
                    else:
                        labels.append("O")
            if words:
                lines.append({"words":words,"labels":labels})

        return lines
                



class InputExample(object):
    def __init__(self,guid,text_a,labels):
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)

        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(),indent=2,sort_keys=True) + "\n"

class InputFeatures(object):
    def __init__(self,input_ids,input_mask,input_len,segment_id,label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_len = input_len
        self.segment_id = segment_id
        self.label_ids = label_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class CluenerProcessors(DataProcess):
    pass


class CnerProcessor(DataProcess):
    """processor for the chinese ner data set"""
    def get_train_examples(self,data_dir):
        return self._create_examples(self._read_txt(os.path.join(data_dir,"train.data")),"train")

    def get_dev_examples(self,data_dir):
        return self._create_examples(self._read_txt(os.path.join(data_dir,"dev.data")),"dev")
    def get_test_examples(self,data_dir):
        return self._create_examples(self._read_txt(os.path.join(data_dir,"train.data")),"test")

    def get_labels(self):

        return ["X","B-EMAIL","B-SCENE","B-NAME","I-MOBILE","I-POSITION","B-BOOK","B-ORGANIZATION","I-MOVIE",
                "B-MOBILE","I-COMPANY", "I-SCENE","I-EMAIL", "B-POSITION", "B-QQ" ,"B-GOVERNMENT" ,"B-COMPANY","S-MOVIE" ,"I-BOOK", 
                "I-ORGANIZATION" ,"S-NAME" ,"B-ADDRESS" ,"I-GOVERNMENT", "I-GAME", "S-COMPANY" ,"B-VX","I-NAME","B-GAME", 
                "I-ADDRESS" ,"B-MOVIE" ,"S-ADDRESS","I-QQ" ,"I-VX","[START]","[END]","O"]




#         return ["X",'B-CONT','B-EDU','B-LOC','B-NAME','B-ORG','B-PRO','B-RACE','B-TITLE',
#                 'I-CONT','I-EDU','I-LOC','I-NAME','I-ORG','I-PRO','I-RACE','I-TITLE',
#                 'O','S-NAME','S-ORG','S-RACE',"[START]", "[END]"]

    def _create_examples(self,lines,set_type):
        "create  examples for training and dev data set"
        examples =[]
        for (i,line) in enumerate(lines):
            if i ==0:
                continue
            guid = "%s-%s" %(set_type,i)
            text_a = line["words"]
            labels = []
            for l in line["labels"]:
                if "M-" in l:
                    labels.append(l.replace("M-","I-"))
                elif "E-" in l:
                    labels.append(l.replace("E-","I-"))
                else:
                    labels.append(l)
            examples.append(InputExample(guid=guid,text_a= text_a,labels=labels))
        return examples



def convert_examples_to_features(examples,label_list,max_seq_length,tokenizer,
                                cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=1,
                                sep_token="[SEP]",pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                sequence_a_segment_id=0,mask_padding_with_zero=True
    ):
    """
    load a data file into a list of inputBtch's 
    `cls_token_at_end` define the location of CLS token:
        - False （Default, BERT/XLM pattern）: [CLS] + A +[SEP] +B +[SEP]
        - True (XLnet/GPT pattern) : A + [SEP] + B + [SEP] + [CLS] 
    `cls_token_segment_id ` define the segment id associated to the CLS token(0 for BERT,2 for XLNET)
    """
    label_map = {label:i for i,label in enumerate(label_list)}
    
    features =[]

    for (ex_index,example) in enumerate(examples):
        if ex_index % 10000 ==0:
            logger.info("Writing example %d of %d",ex_index,len(examples))
        tokens = tokenizer.tokenize(example.text_a)
        label_ids = [label_map[x] for x in example.labels] 
        #account for [CLS] and [SEP] with -2
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:max_seq_length-special_tokens_count]
            label_ids = label_ids[:max_seq_length-special_tokens_count]

        tokens += [sep_token]
        label_ids += [label_map["O"]]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [label_map["O"]]
            segment_ids += [sequence_a_segment_id]

        else:
            tokens = [cls_token] + tokens
            label_ids = [label_map["O"]] + label_ids
            segment_ids = [sequence_a_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(input_ids)

        #zero-pad up to the sequsence length
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0]* padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token] * padding_length) + label_ids

        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token] * padding_length
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid:%s",example.guid)
            logger.info("tokens:%s","".join([str(x) for x in tokens]))
            logger.info("input ids:%s","".join([str(x) for x in input_ids]))
            logger.info("segment ids:%s","".join([str(x) for x in segment_ids]))
            logger.info("labels ids:%s","".join([str(x) for x in label_ids]))

        features.append(InputFeatures(input_ids=input_ids,input_mask=input_mask,
                                input_len=input_len,segment_id=segment_ids,label_ids=label_ids))

    return features





        








ner_processors = {
    "mininglamp": CnerProcessor,
    "cluener":CluenerProcessors
}
