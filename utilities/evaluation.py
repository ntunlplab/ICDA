import torch
import pandas as pd
import sklearn.metrics
from colorama import Fore, Style
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Any
from argparse import Namespace

import seqeval.metrics
from seqeval.scheme import IOB2

from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast, AutoTokenizer

from .model import BertDxModel, BiEncoder
from .utils import move_bert_input_to_device

def predict_whole_set_ner(model: BertModel, data_loader: DataLoader, device: str) -> list:
    y_pred_raw = list()
    y_true_raw = list()
    model.to(device)
    model.eval()
    print(f"Predicting whole dataset...")
    for batch in tqdm(data_loader):
        x = move_bert_input_to_device(batch[0], device)
        with torch.no_grad():
            if len(batch) == 2:
                scores = model(x)
            if len(batch) > 2:
                scores = model(x)[1]
            pred = scores.argmax(dim=-1).detach().cpu().tolist()
            y_pred_raw.append(pred)
        true = batch[-1].detach().cpu().tolist()
        y_true_raw.append(true)

    return y_pred_raw, y_true_raw

def ids_to_iobs(y_pred_raw: List[List[List[int]]], y_true_raw: List[List[List[int]]], ner_dataset: Dataset) -> Tuple[List[List[str]], List[List[str]]]:
    y_pred = list()
    y_true = list()
    for batch_pred, batch_true in zip(y_pred_raw, y_true_raw):
        for emr_pred, emr_true in zip(batch_pred, batch_true):
            assert len(emr_pred) == len(emr_true)
            iob_preds = list()
            iob_trues = list()
            for idx_pred, idx_true in zip(emr_pred, emr_true):
                if idx_true != ner_dataset.ignore_index:
                    iob_pred = ner_dataset.idx2iob[idx_pred]
                    iob_true = ner_dataset.idx2iob[idx_true]
                    iob_preds.append(iob_pred)
                    iob_trues.append(iob_true)
            y_pred.append(iob_preds)
            y_true.append(iob_trues)
                
    return y_pred, y_true

def calc_seqeval_metrics(y_true, y_pred) -> Dict[str, float]:
    token_acc = seqeval.metrics.accuracy_score(y_true, y_pred)
    p = seqeval.metrics.precision_score(y_true, y_pred, average="micro", mode="strict", scheme=IOB2)
    r = seqeval.metrics.recall_score(y_true, y_pred, average="micro", mode="strict", scheme=IOB2)
    f1 = seqeval.metrics.f1_score(y_true, y_pred, average="micro", mode="strict", scheme=IOB2)

    return {"token_acc": token_acc, "p": p, "r": r, "f1": f1}

def visualize_ner_labels(tokenizer: BertTokenizerFast, input_ids: List[int], ner_labels: List[int]):
    for i, token_id in enumerate(input_ids[0]):
        token = tokenizer.decode(token_id)
        if token[:2] == "##":
            token = token[2:]
            print('\b', end='')
        if ner_labels[i] == 0:
            print(Style.RESET_ALL + token, end=' ')
        else:
            print(Fore.RED + token, end=' ')

def visualize_iobpol_labels(tokenizer: AutoTokenizer, input_ids: List[int], ner_labels: List[int], label_color_mappings: Dict[int, str]) -> None:
    for i, token_id in enumerate(input_ids):
        token = tokenizer.decode(token_id)
        
        # post-processing
        if token[:2] == "##":
            token = token[2:]
            print('\b', end='')
        
        # print with different color
        color = label_color_mappings[ner_labels[i]]
        print(color + token, end=' ')

def get_top_k_accuracies(y_true, y_scores, k, labels):
    accs = list()
    for i in range(1, k + 1):
        acc = sklearn.metrics.top_k_accuracy_score(y_true, y_scores, k=i, labels=labels)
        accs.append((i, acc))
    return pd.DataFrame(accs, columns=['k', "acc"]).set_index('k')

def get_evaluations(y_true, y_pred, label_size, model_outputs, model_name: str) -> pd.DataFrame: # macro/micro-f1; Cohen's kappa; Matthew’s correlation coefficient; h@1,3,6,9
    macro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    micro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    cohen_kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    mcc = sklearn.metrics.matthews_corrcoef(y_true, y_pred)
    hak = get_top_k_accuracies(y_true, model_outputs, k=10, labels=range(label_size))
    return pd.DataFrame(
        {
            "macro_f1": [macro_f1],
            "micro_f1": [micro_f1],
            "cohen_kappa": [cohen_kappa],
            "mcc": [mcc],
            "h@3": hak.loc[3].acc,
            "h@6": hak.loc[6].acc,
            "h@9": hak.loc[9].acc
        },
        index=[model_name]
    )

def predict_whole_set_dx(model: BertDxModel, data_loader: DataLoader, device: str) -> list:
    preds = list()
    model.to(device)
    model.eval()
    for batch in tqdm(data_loader):
        x = move_bert_input_to_device(batch[0], device)
        with torch.no_grad():
            pred = model(x)
            if len(batch) > 2:
                pred = pred[0]
            preds.append(pred)
    return torch.cat(preds, dim=0)

def evaluate_dx_model(model: BertDxModel, eval_loader: DataLoader, device: str, verbose: bool = False) -> Dict[str, Any]:
    # collect output logits and labels
    logits = list()
    y_true = list()
    model = model.to(device)
    model.eval()
    for X, y in tqdm(eval_loader) if verbose else eval_loader:
        X = move_bert_input_to_device(X, device)
        with torch.no_grad():
            logit = model(X)
            logits.append(logit)
        y_true += y.tolist()
    
    logits = torch.cat(logits, dim=0).cpu()
    y_pred = logits.argmax(dim=-1).tolist()

    # calculate metrics
    return {
        "macro_f1": sklearn.metrics.f1_score(y_true, y_pred, average="macro"),
        "micro_f1": sklearn.metrics.f1_score(y_true, y_pred, average="micro"),
        "cohen_kappa": sklearn.metrics.cohen_kappa_score(y_true, y_pred),
        "mcc": sklearn.metrics.matthews_corrcoef(y_true, y_pred),
        "hat3": sklearn.metrics.top_k_accuracy_score(y_true, logits, k=3, labels=range(eval_loader.dataset.num_dx_labels)),
        "hat5": sklearn.metrics.top_k_accuracy_score(y_true, logits, k=5, labels=range(eval_loader.dataset.num_dx_labels)),
        "hat8": sklearn.metrics.top_k_accuracy_score(y_true, logits, k=8, labels=range(eval_loader.dataset.num_dx_labels))
    }

def evaluate_nen(data_loader: DataLoader, model: BiEncoder, args: Namespace, entity_embeddings: torch.FloatTensor = None, entities_loader: DataLoader = None) -> float:
    assert entities_loader
    # Load or construct entity embeddings
    if entity_embeddings != None:
        all_y_ents = entity_embeddings.to(args.device)
    else:
        all_y_ents = model.encode_all_entities(entities_loader, args).to(args.device)
    
    # Evaluation
    all_cuis = entities_loader.dataset._ids
    total_correct = 0
    total_predict = 0

    model.eval()
    for emr_be, mention_indices_l, target_cuis, _ in data_loader: # No need of negative_cuis_l
        emr_be = move_bert_input_to_device(emr_be, args.device)

        with torch.no_grad():
            # Encode mentions
            y_ments = model.encode_mentions(emr_be, mention_indices_l)
            assert len(y_ments) == len(mention_indices_l) == len(target_cuis)

            # Calculate scores
            scores = model.calc_scores(y_ments, all_y_ents)
            
            # Check correctness
            preds = scores.argmax(dim=-1).cpu().tolist()
            # Calculate num correct
            for pred, target_cui in zip(preds, target_cuis):
                pred_cui = all_cuis[pred]
                if pred_cui == target_cui:
                    total_correct += 1
            
            total_predict += len(preds)
    
    fullset_acc = total_correct / total_predict
    return fullset_acc

pollabel_color_mappings = {
    0: Style.RESET_ALL,
    1: Fore.GREEN,
    2: Fore.CYAN,
    3: Fore.RED,
    4: Fore.YELLOW,
    -100: Style.RESET_ALL
}