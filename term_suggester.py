from pyexpat import model
from typing import Dict, Any, List, Tuple

import pandas as pd
import torch

from utilities.utils import load_json
from diagnosis_classifier import DiagnosisClassifier

class TermSuggester(object):
    def __init__(
            self, 
            score_matrix: pd.DataFrame, 
            id2term: Dict[int, str],
            inequality: str, 
            threshold: float,
            diagnosis_classifier: DiagnosisClassifier
        ):
        self.score_matrix = score_matrix
        self.id2term = id2term
        if inequality not in ["greater", "lesser"]:
            raise ValueError("inequality must be 'lesser' or 'greater'")
        self.inequality = inequality
        self.threshold = threshold
        self.diagnosis_classifier = diagnosis_classifier
        self.term_lists: Dict[Any, List[str]] = dict()

    def build_term_lists(self):
        # build term_id lists
        term_ids_lists = dict()
        for label in self.score_matrix.columns:
            term_id2score = self.score_matrix.loc[:, label]
            if self.inequality == "lesser":
                term_ids = term_id2score[term_id2score < self.threshold].sort_values(ascending=True).index.tolist()
            else:
                term_ids = term_id2score[term_id2score > self.threshold].sort_values(ascending=False).index.tolist()
            term_ids_lists[label] = term_ids
        
        # convert term_ids to terms
        for label, term_ids in term_ids_lists.items():
            terms = list(map(lambda term_id: self.id2term[term_id], term_ids))
            self.term_lists[label] = terms
            
    def load_term_lists(self, term_lists_path: str):
        self.term_lists = load_json(term_lists_path)
    
    def suggest_terms_l(self, text_l: List[str], all_logits: torch.FloatTensor):
        raise NotImplementedError
    
    def rank_terms(self, text: str, candidates: List[str]) -> Tuple[List[str], List[float]]:
        # NOTE: current version is based on "positive"
        text_w_term_l = list()
        for cand in candidates:
            # append each term one by one
            text_w_term = text + f" positive {cand}"
            text_w_term_l.append(text_w_term)

        # calculate entropy for each text_w_term
        all_logits = self.diagnosis_classifier.predict(text_w_term_l)
        entropies = self.diagnosis_classifier.calc_entropy(all_logits).tolist()

        assert len(candidates) == len(entropies)
        term_entropy_ts = list(zip(candidates, entropies))
        ranked_terms, scores = zip(*sorted(term_entropy_ts, key=lambda t: t[1]))
        return ranked_terms, scores
