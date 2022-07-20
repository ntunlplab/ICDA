from typing import Dict, Any, List, Tuple

import pandas as pd
import torch
from torch.nn.functional import softmax

from utilities.utils import load_json
from diagnosis_classifier import DiagnosisClassifier

class UMLSClassifier(object):
    def __init__(
            self, 
            cui2name: Dict[str, str],
            cui2typeinfo: Dict[str, List[Dict[str, str]]],
            cat2typenames: Dict[str, List[str]]
        ):
        self.name2cui = {name: cui for cui, name in cui2name.items()}
        self.cui2typeinfo = cui2typeinfo
        self.typename2cat = dict()
        for cat, typenames in cat2typenames.items():
            for typename in typenames:
                self.typename2cat[typename] = cat
    
    def __call__(self, term_lists: Dict[Any, List[str]]) -> Dict[str, Dict[str, List[str]]]:
        for label, term_list in term_lists.items():
            classified_list = dict()
            for term in term_list:
                cui = self.name2cui[term]
                typeinfos = self.cui2typeinfo[cui]
                cat = self.typeinfos_to_cat(typeinfos)
                if cat not in classified_list:
                    classified_list[cat] = [term]
                else:
                    classified_list[cat].append(term)
                
            term_lists[label] = classified_list
        
        return term_lists
    
    def typeinfos_to_cat(self, typeinfos: List[Dict[str, str]]):
        cat_num = {cat: 0 for cat in set(self.typename2cat.values())}
        cat_num["others"] = 0
        for typeinfo in typeinfos:
            typename = typeinfo["desc"]
            cat = self.typename2cat.get(typename, "others")
            cat_num[cat] += 1
        
        sorted_cat_num = sorted(cat_num.items(), key=lambda t: t[1], reverse=True)
        return sorted_cat_num[0][0]

class TermSuggester(object):
    def __init__(
            self, 
            score_matrix: pd.DataFrame, 
            id2term: Dict[int, str],
            inequality: str, 
            threshold: float,
            diagnosis_classifier: DiagnosisClassifier,
            top_k_dxs: int,
            term_lists_path: str = None
        ):
        self.score_matrix = score_matrix
        self.id2term = id2term
        if inequality not in ["greater", "lesser"]:
            raise ValueError("inequality must be 'lesser' or 'greater'")
        self.inequality = inequality
        self.threshold = threshold
        self.diagnosis_classifier = diagnosis_classifier
        self.top_k_dxs = top_k_dxs
        self.term_lists: Dict[Any, List[str]] = dict()
        if term_lists_path:
            self.load_term_lists(term_lists_path)
        else:
            self.build_term_lists()

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
    
    def select_term_candidates(self, dxs: List[Any]) -> List[str]:
        candidates = set()
        for dx in dxs:
            term_list = self.term_lists[str(dx)]
            term_set = set(term_list)
            candidates = candidates | term_set
        
        return list(candidates)
    
    def suggest_terms_l(self, text_l: List[str], obs_terms_l: List[List[str]], dxs_l: List[List[Any]], probs_l: List[List[float]]) -> Dict[Any, List[str]]:
        assert len(text_l) == len(obs_terms_l) == len(dxs_l)
        
        sug_terms_d_l = list()
        for text, obs_terms, dxs, probs in zip(text_l, obs_terms_l, dxs_l, probs_l):
            sug_terms_d = dict()
            obs_terms_s = set(obs_terms)
            top_dxs = dxs[:self.top_k_dxs]
            top_probs = probs[:self.top_k_dxs]
            for dx, prob in zip(top_dxs, top_probs):
                term_candidates = self.select_term_candidates([dx])
                ranked_terms, scores = self.rank_terms(text, target_dx=dx, original_prob=prob, candidates=term_candidates)
                sug_terms = list()
                for ranked_term in ranked_terms:
                    if ranked_term not in obs_terms_s:
                        sug_terms.append(ranked_term)
                sug_terms_d[dx] = sug_terms
            sug_terms_d_l.append(sug_terms_d)
        
        return sug_terms_d_l
    
    def rank_terms(self, text: str, target_dx: Any, original_prob: float, candidates: List[str]) -> Tuple[List[str], List[float]]:
        # NOTE: current version is based on "positive"
        text_w_term_l = list()
        for cand in candidates:
            # append each term one by one
            text_w_term = text + f" positive {cand}"
            text_w_term_l.append(text_w_term)

        # calculate entropy for each text_w_term
        all_logits = self.diagnosis_classifier.predict(text_w_term_l)
        entropies = self.diagnosis_classifier.calc_entropy(all_logits).tolist()
        
        # calculate new probabilities
        new_probs_l = torch.softmax(all_logits, dim=-1)
        target_id = self.diagnosis_classifier.dx2id[target_dx]
        new_probs = new_probs_l[:, target_id].tolist()

        # filter candidates of which new probabilities are smaller than original probability
        assert len(candidates) == len(entropies) == len(new_probs)
        dx_candidates = list()
        dx_entropies = list()
        for cand, entropy, new_prob in zip(candidates, entropies, new_probs):
            if new_prob > original_prob:
                dx_candidates.append(cand)
                dx_entropies.append(entropy)

        term_entropy_ts = list(zip(dx_candidates, dx_entropies))
        ranked_terms, scores = zip(*sorted(term_entropy_ts, key=lambda t: t[1]))
        return ranked_terms, scores
