from typing import Dict, List, Set, Any
from tqdm import tqdm

import pandas as pd
from scipy.stats import fisher_exact

def build_cooccurrence_matrix(entities_l: List[Set[str]], labels: List[Any], vocab: dict) -> pd.DataFrame:
    assert len(entities_l) == len(labels)
    ndoc_dict = {label: [0 for _ in range(len(vocab))] for label in labels}
    for entities, label in zip(entities_l, labels):
        for ent in entities:
            ent_idx = vocab[ent]
            ndoc_dict[label][ent_idx] += 1

    return pd.DataFrame(ndoc_dict)

def build_fisher_matrix(co_matrix: pd.DataFrame, label2ndoc: dict, total_ndoc: int, min_ndoc_k: int, min_ndoc_p: float) -> pd.DataFrame:
    f_matrix = co_matrix.copy()
    for label in tqdm(f_matrix.columns):
        label_ndoc = label2ndoc[label]
        min_ndoc = max(min_ndoc_k, label_ndoc * min_ndoc_p)
        for term in f_matrix.index:
            term_ndoc = co_matrix.loc[term, :].sum()
            term_label_ndoc = co_matrix.at[term, label]
            if term_label_ndoc >= min_ndoc:
                contingency_table = [
                    [term_label_ndoc, term_ndoc - term_label_ndoc],
                    [label_ndoc - term_label_ndoc, total_ndoc - term_ndoc - label_ndoc + term_label_ndoc]
                ]
                p_value = fisher_exact(table=contingency_table, alternative="greater")[1]
            else:
                p_value = 1.0
            f_matrix.at[term, label] = p_value

    return f_matrix

def build_term_ids_lists(score_matrix: pd.DataFrame, mode: str, threshold: float) -> Dict[Any, List[int]]:
    if mode not in ["greater", "lesser"]:
        raise ValueError("mode msut be 'lesser' or 'greater'")

    term_ids_lists = dict()
    for label in score_matrix.columns:
        term_id2score = score_matrix.loc[:, label]
        if mode == "lesser":
            term_ids = term_id2score[term_id2score < threshold].sort_values(ascending=True).index.tolist()
        else:
            term_ids = term_id2score[term_id2score > threshold].sort_values(ascending=False).index.tolist()
        term_ids_lists[label] = term_ids
    
    return term_ids_lists