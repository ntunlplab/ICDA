import math
from typing import List, Any, Set, Tuple, Dict

from sklearn.model_selection import train_test_split

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

def preprocess_patient_state_tuples(state_tuples_l: List[List[Tuple[str, int]]], label2token: Dict[int, str]) -> List[str]:
    text_l = list()
    for state_tuples in state_tuples_l:
        text = list()
        for term, label in state_tuples:
            labeltoken = label2token[label]
            text += [labeltoken, term]
        text = ' '.join(text) if text else ''
        text_l.append(text)
    return text_l

def augment_patient_states_with_partials(patient_states: List[List[Tuple[str, int]]], n_partials: int) -> List[List[Tuple[str, int]]]:
    pa_patient_states = patient_states.copy() # patient states with partial augmentation (PA)
    for partial_idx in range(1, n_partials):
        prop = partial_idx / n_partials
        for patient_state in patient_states:
            end_entity_idx = math.floor(len(patient_state) * prop)
            partial_state = patient_state[:end_entity_idx + 1]
            pa_patient_states.append(partial_state)
    
    assert len(pa_patient_states) == len(patient_states) * n_partials
    return pa_patient_states

def train_valid_test_split(inputs: list, labels: list, train_size: float, valid_size: float, test_size: float, seed: int):
    assert len(inputs) == len(labels)
    
    train_inputs, eval_inputs, train_labels, eval_labels = train_test_split(
        inputs,
        labels,
        train_size=train_size,
        test_size=valid_size + test_size,
        random_state=seed,
        stratify=labels # NOTE: the values passed to 'stratify' will affect splitting results
    )
    valid_inputs, test_inputs, valid_labels, test_labels = train_test_split(
        eval_inputs,
        eval_labels,
        train_size=valid_size / (valid_size + test_size),
        test_size=test_size / (valid_size + test_size),
        random_state=seed,
        stratify=eval_labels
    )
    return train_inputs, valid_inputs, test_inputs, train_labels, valid_labels, test_labels

def pad_int_icd(icd: int):
    return str(icd) if len(str(icd)) == 3 else '0' * (3 - len(str(icd))) + str(icd)