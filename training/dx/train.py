import sys
sys.path.append("../../")

import json
import pickle
from pathlib import Path
from argparse import Namespace
from collections import Counter

import pandas as pd

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from utilities.utils import set_seeds, render_exp_name, load_args, load_pickle, load_jsonl, load_json, save_json, get_logger
from utilities.preprocess import build_label2id_mapping, augment_patient_states_with_partials, preprocess_patient_state_tuples
from utilities.data import MedicalDxDataset
from utilities.model import BertDxModel, encoder_names_mapping
from utilities.trainer import ICDATrainer
from utilities.evaluation import evaluate_dx_model

def train(args: Namespace):
    # Configuration
    # logger
    logger = get_logger(name=str(__name__))

    # set up experiment
    args.exp_name = render_exp_name(args, hparams=args.exp_hparams, sep='__')
    args.exp_path = f"{args.save_dir}/{args.exp_name}"
    Path(args.exp_path).mkdir(parents=True, exist_ok=True)

    # save args
    (Path(args.exp_path) / "config.json").write_text(json.dumps(vars(args), indent=4))
    (Path(args.exp_path) / "args.pickle").write_bytes(pickle.dumps(args))

    # Data
    # load data
    emrs = load_json(args.input_file)
    icds = load_json(args.label_file)

    # preprocess labels
    # convert ICDs to label_ids
    icd2id = build_label2id_mapping(labels=icds)
    labels = [icd2id[icd] for icd in icds]
    # save label conversion mapping
    id2icd = {id_: icd for icd, id_ in icd2id.items()}
    save_json(id2icd, f"{args.exp_path}/id2icd.json")

    # split data
    train_X, eval_X, train_y, eval_y = train_test_split(
        emrs, 
        labels, 
        train_size=args.train_size, 
        test_size=(args.valid_size + args.test_size), 
        random_state=args.seed, 
        stratify=labels
    )

    valid_X, test_X, valid_y, test_y = train_test_split(
        eval_X,
        eval_y,
        train_size=args.valid_size / (args.valid_size + args.test_size),
        test_size=args.test_size / (args.valid_size + args.test_size),
        random_state=args.seed,
        stratify=eval_y
    )

    # partial augmentation
    if args.input_type in ["unnorm", "norm"]:
        train_X = augment_patient_states_with_partials(patient_states=train_X, n_partials=args.n_partials)
    train_y = train_y * args.n_partials

    # preprocess input EMRs
    if args.input_type in ["unnorm", "norm"]:
        train_X = preprocess_patient_state_tuples(state_tuples_l=train_X, label2token={0: "positive", 1: "negative"})
        valid_X = preprocess_patient_state_tuples(state_tuples_l=valid_X, label2token={0: "positive", 1: "negative"})

    # build dataset instances
    tokenizer = AutoTokenizer.from_pretrained(encoder_names_mapping[args.tokenizer], use_fast=True)
    train_set = MedicalDxDataset(emrs=train_X, dx_labels=train_y, tokenizer=tokenizer)
    valid_set = MedicalDxDataset(emrs=valid_X, dx_labels=valid_y, tokenizer=tokenizer)

    # Model
    model = BertDxModel(
        encoder_name=encoder_names_mapping[args.encoder],
        num_dxs=len(Counter(labels))
    )
    tokenizer.model_max_length = model.bert.embeddings.position_embeddings.num_embeddings

    optimizer = AdamW(params=model.parameters(), lr=args.lr)

    total_steps = (len(train_set) // (args.train_batch_size * args.grad_accum_steps) + 1) * args.nepochs
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup_ratio * total_steps, num_training_steps=total_steps)

    # Optimization
    trainer = ICDATrainer(
        model=model,
        train_set=train_set,
        valid_set=valid_set,
        optimizer=optimizer,
        scheduler=scheduler,
        eval_func=evaluate_dx_model,
        logger=logger,
        args=args
    )
    trainer.train()

if __name__ == "__main__":
    args = load_args("./config.json")
    train(args)