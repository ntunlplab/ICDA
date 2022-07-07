import math
from typing import List, Any, Set, Tuple, Dict

def select_labels_subset(inputs: List[Any], labels: List[Any], target_labels: Set[Any]) -> Tuple[list]:
    assert len(inputs) == len(labels)

    new_inputs = list()
    new_labels = list()
    for input_, label in zip(inputs, labels):
        if label in target_labels:
            new_inputs.append(input_)
            new_labels.append(label)

    return new_inputs, new_labels

def build_label2id_mapping(labels: list) -> Dict[Any, int]:
    label2id = dict()
    for label in labels:
        if label not in label2id:
            label2id[label] = len(label2id)

    return label2id

def augment_full_emrs_with_partials(emrs: List[str], spans_l: List[List[List[int]]], dxs: List[Any], n_partials: int) -> Tuple[List[str], List[Any]]:
    
    def make_partial(emr: str, spans: List[List[int]], prop: float) -> str:
        if len(spans) == 0:
            return emr
        
        end_span_idx = math.floor(len(spans) * prop)
        end_offset = spans[end_span_idx][1]
        return emr[:end_offset]
    
    assert len(emrs) ==  len(spans_l) == len(dxs)
    augmented_emrs = emrs.copy()
    augmented_dxs = dxs.copy()

    for partial_idx in range(1, n_partials):
        prop = partial_idx / n_partials
        for emr, spans, dx in zip(emrs, spans_l, dxs):
            partial_emr = make_partial(emr, spans, prop)
            augmented_emrs.append(partial_emr)
            augmented_dxs.append(dx)
    
    assert len(augmented_emrs) == len(augmented_dxs) == len(dxs) * n_partials
    return augmented_emrs, augmented_dxs

def augment_extracted_emrs_with_partials(docs: List[Dict[str, Any]], dxs: List[Any], n_partials: int, intype: str) -> Tuple[List[List[str]], List[List[int]], List[Any]]:
    assert len(docs) == len(dxs)
    entities_l = [doc[intype] for doc in docs]
    pols_l = [doc["pols"] for doc in docs] # 0 == positive; 1 == negative
    augmented_dxs = dxs.copy()

    for partial_idx in range(1, n_partials):
        prop = partial_idx / n_partials
        for entities, pols, dx in zip(entities_l, pols_l, dxs):
            assert len(entities) == len(pols)
            # build partial input
            end_entity_idx = math.floor(len(pols) * prop)
            partial_entities = entities[:end_entity_idx + 1]
            partial_pols = pols[:end_entity_idx + 1]
            # add to training samples
            entities_l.append(partial_entities)
            pols_l.append(partial_pols)
            augmented_dxs.append(dx)

    assert len(entities_l) == len(pols_l) == len(augmented_dxs)
    return entities_l, pols_l, augmented_dxs

def preprocess_extracted_emrs(entities_l: List[List[str]], pols_l: List[List[int]], scheme: str = "every") -> List[str]:
    assert len(entities_l) == len(pols_l)
    X = list()
    # add "positive" ro "negative" before every entity
    if scheme == "every":
        for entities, pols in zip(entities_l, pols_l):
            pol_ent_l = list()
            for ent, pol in zip(entities, pols):
                if pol == 0:
                    pol_name = "positive"
                elif pol == 1:
                    pol_name = "negative"
                else:
                    raise ValueError("pol should be either 0 or 1")
                pol_ent_l.append(f"{pol_name} {ent}")
            pol_ent_text = ' '.join(pol_ent_l)
            X.append(pol_ent_text)

    return X

def preprocess_patient_state_tuples(state_tuples_l: List[List[Tuple[str, int]]], label2token: Dict[int, str]):
    text_l = list()
    for state_tuples in state_tuples_l:
        text = list()
        for term, label in state_tuples:
            labeltoken = label2token[label]
            text += [labeltoken, term]
        text = ' '.join(text)
        text_l.append(text)
    return text_l