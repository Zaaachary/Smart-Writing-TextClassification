import json

from torch.utils.data import TensorDataset
from utils.feature import Feature

class SWTCExample:

    def __init__(self, example_id, label, text):
        self.example_id = example_id
        self.text = text
        self.label = label

    def tokenize(self, tokenizer, max_seq_len):

        tokens = tokenizer.tokenize(self.text)   # 分词 
        # 转换到 feature: (idx, input_ids, input_mask, segment_ids)
        feature = Feature.make_single(self.example_id, tokens, tokenizer, max_seq_len)
        
        return feature, self.label
    
    @classmethod
    def load_from_dict(cls, dict_obj, label2id, max_len):
        '''
        通过 dict 构造一个 SWTC Example
        '''
        title = dict_obj['title']
        content = dict_obj['content'][:max_len]
        example_id = dict_obj['id']
        topics = label2id[dict_obj['topics']]
        
        text = f" {title} [SEP] {content} "

        return cls(example_id, topics, text)