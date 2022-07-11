import sys
import json
import pickle
import logging
from typing import Any, Set, List, Dict
from pathlib import Path
from argparse import Namespace

import torch
import random
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(name: str = __name__):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(stream=sys.stdout)
    logger = logging.getLogger(name)

    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    return logger

def load_config(config_path: str = "./config.json") -> dict:
    with open(config_path) as f:
        config = json.load(f)
    return config

def load_args(config_path: str) -> Namespace:
    return Namespace(**json.loads(Path(config_path).read_bytes()))

def load_json(path: str) -> Any:
    return json.loads(Path(path).read_bytes())

def save_json(obj: Any, f: str) -> None:
    Path(f).write_text(json.dumps(obj, indent=4))

def load_jsonl(path: str) -> List[dict]:
    dict_l = list()
    with open(path) as f:
        for line in f:
            d = json.loads(line.rstrip())
            dict_l.append(d)
    return dict_l

def load_pickle(path: str) -> Any:
    return pickle.loads(Path(path).read_bytes())

def render_exp_name(args: Namespace, hparams: List[str], sep: str) -> str:
    exp_name_l = list()
    for hparam in hparams:
        value = getattr(args, hparam)
        if hparam == "lw":
            value = value[1]
        exp_name_l.append(f"{hparam}-{value}")
    return sep.join(exp_name_l)

def visualize_ner_indices(emr: str, ner_indices: Set[int]):
    for i, char in enumerate(emr):
        if i in ner_indices:
            print(Fore.RED + char, end='')
        else:
            print(Style.RESET_ALL + char, end='')

def visualize_iob_labels(tokenizer, input_ids: List[int], label_ids: List[int], idx2label: Dict[int, str]) -> None:
    assert len(input_ids) == len(label_ids)
    for token_id, label_idx in zip(input_ids, label_ids):
        token = tokenizer.decode(token_id)
        label = idx2label[label_idx]
        if token[:2] == "##":
            token = token[2:]
            print('\b', end='')

        if label[0] == 'O':
            print(Style.RESET_ALL + token, end=' ')
        elif label[0] == 'B':
            print(Fore.RED + token, end=' ')
        elif label[0] == 'I':
            print(Fore.YELLOW + token, end=' ')

def move_bert_input_to_device(x, device):
    for k in x:
        x[k] = x[k].to(device)
    return x

"""
    Old Utilities
"""
def trainer(train_loader, val_loader, model, criterion, args):
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

    # some variables
    record = {
        "acc": list(),
        "loss": list()
    }
    best_val_acc = 0
    step = 0

    for epoch in range(1, args.nepochs + 1):
        for x, y in train_loader:
            model.train()
            # move data to device
            x = move_bert_input_to_device(x, args.device)
            y = y.to(args.device)
            # make prediction and calculate loss
            scores = model(x)
            loss = criterion(scores.transpose(1, 2), y)

            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # evaluate model at the checkpoint step
            if step % args.ckpt_steps == 0:
                print("Evaluating model at step {}...".format(step))
                best_val_acc = update_evaluation(val_loader, model, criterion, args, record, best_val_acc)            
            step += 1

        # evaluate model at the end of one epoch
        print("===== Evaluating model at epoch {} =====".format(epoch))
        best_val_acc = update_evaluation(val_loader, model, criterion, args, record, best_val_acc)

    record["best_val_acc"] = best_val_acc
    return record

def update_evaluation(data_loader, model, criterion, args, record, best_acc):
    # utility function
    def update_record(record, acc, loss):
        record["acc"].append(acc)
        record["loss"].append(loss)
        return record
    # update metrics
    acc = evaluate_model_acc(data_loader, model, args.device)
    loss = evaluate_model_loss(data_loader, model, criterion, args.device)
    record = update_record(record, acc, loss)
    print(f"Acc: {acc:.4f} / Loss: {loss:.4f}")
    if acc > best_acc:
        best_acc = acc
        if args.model_save_name:
            torch.save(model.state_dict(), "./models/{}.pth".format(args.model_save_name))
            print("Best model saved.")

    return best_acc

def evaluate_model_loss(data_loader, model, criterion, device):
    model.eval()
    total_val_loss = 0

    for x, y in data_loader:
        # move data to device
        x = move_bert_input_to_device(x, device)
        y = y.to(device)
        with torch.no_grad():
            pred = model(x).transpose(1, 2) # transpose for calculating cross entropy loss
            loss = criterion(pred, y)
        total_val_loss += loss.detach().cpu().item() * y.shape[0]
    
    mean_val_loss = total_val_loss / len(data_loader.dataset)
    return mean_val_loss

def evaluate_model_acc(data_loader, model, device):
    total_tokens = 0
    total_correct = 0
    
    model.eval()
    for x, y in data_loader:
        # inference
        x = move_bert_input_to_device(x, device)
        y = y.to(device)
        with torch.no_grad():
            pred = model(x) # no need to transpose in this case
        # calculate target metric (acc)
        total_tokens += (y != -100).sum().cpu().item()
        total_correct += (pred.argmax(dim=-1) == y).sum().cpu().item()
    
    acc = total_correct / total_tokens
    return acc

def build_reverse_dict(d: dict):
    return {v: k for k, v in d.items()}