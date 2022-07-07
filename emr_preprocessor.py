from typing import List

from finding_extractor import FindingExtractor

class EMRPreprocessor(object):
    def __init__(self, finding_extractor: FindingExtractor, batch_size: int):
        self.finding_extractor = finding_extractor
    
    def preprocess(self, text_l: List[str]) -> List[str]:

        return 
    
    def convert_to_state():

        return
    
    def add_label_before_every_term(self, terms_l: List[List[str]], labels_l: List[List[int]]):
        assert len(terms_l) == len(labels_l)

        text_l = list()
        for terms, labels in zip(terms_l, labels_l):
            assert len(terms) == len(labels)
            text = list()
            for term, label in zip(terms, labels):
                token = self.finding_extractor.recognizer.label2token[label]
                text += [token, term]
            text = ' '.join(text)
            text_l.append(text)
        
        return text_l
        