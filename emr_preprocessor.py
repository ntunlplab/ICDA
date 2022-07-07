from typing import List, Union, Dict, Tuple

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

    @staticmethod
    def build_patient_states(terms_l: List[List[str]], pols_l: List[List[int]], return_type: bool = "dict") -> Union[List[Dict[str, int]], List[List[Tuple[str, int]]]]:
        assert len(terms_l) == len(pols_l)
        if return_type not in ["dict", "tuple"]:
            raise ValueError("return_type should be 'dict' or 'tuple'")

        states = list()
        for terms, pols in zip(terms_l, pols_l):
            assert len(terms) == len(pols)
            state = dict() if return_type == "dict" else list()
            for term, pol in zip(terms, pols):
                t = (term, pol)
                if return_type == "dict":
                    state[t] = state.get(t, 0) + 1
                elif return_type == "tuple":
                    state.append(t)
            states.append(state)
        
        return states