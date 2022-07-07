from typing import List

from model import BertDxModel

class DiagnosisClassifier(object):
    def __init__(self, model: BertDxModel):
        self.model = model
    
    def predict(emrs: List[str]):
        return