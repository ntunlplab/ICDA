import sys
sys.path.append("../../")

from typing import List, Dict
from pathlib import Path
from argparse import Namespace, ArgumentParser
from collections import Counter

import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utilities.utils import set_seeds, load_pickle, load_jsonl, load_json, build_reverse_dict
from utilities.preprocess import augment_patient_states_with_partials, preprocess_patient_state_tuples
from utilities.data import MedicalDxDataset
from utilities.model import BertDxModel, encoder_names_mapping
from utilities.evaluation import evaluate_dx_model

import warnings
warnings.filterwarnings("ignore")

def main(args: Namespace):
    # Configuration
    set_seeds(args.seed)

    # Data
    # load data
    emrs = load_json(args.input_file)
    icds = load_json(args.label_file)
    print("Data loaded.")

    # preprocess labels
    icd2id = build_reverse_dict(load_json(f"{args.exp_path}/id2icd.json"))
    labels = list(map(lambda icd: int(icd2id[icd]), icds))
    print("Label preprocessed.")

    # split
    assert len(emrs) == len(labels)
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
    print(f"Data split. Train size = {len(train_y)}; Valid size = {len(valid_y)}; Test size = {len(test_y)}")

    # partial augmentation of valid and test data
    if args.input_type in ["unnorm", "norm"]:
        valid_X = augment_patient_states_with_partials(patient_states=valid_X, n_partials=args.eval_partials)
        test_X = augment_patient_states_with_partials(patient_states=test_X, n_partials=args.eval_partials)
    valid_y = valid_y * args.eval_partials
    test_y = test_y * args.eval_partials
    print(f"Partial augmentation done. Valid size = {len(valid_y)}; Test size = {len(test_y)}")

    # preprocess input EMRs
    if args.input_type in ["unnorm", "norm"]:
        # TODO: change label2token from hard-coded values to loaded dict
        label2token = {0: "positive", 1: "negative"}
        valid_X = preprocess_patient_state_tuples(state_tuples_l=valid_X, label2token=label2token)
        test_X = preprocess_patient_state_tuples(state_tuples_l=test_X, label2token=label2token)
    print(f"Data preprocessed. Example: \n {valid_X[0]}")

    # Model
    model = BertDxModel(
        encoder_name=encoder_names_mapping[args.encoder], 
        num_dxs=len(Counter(labels))
    )
    model.load_state_dict(torch.load(Path(args.exp_path) / "best_models" / f"best_{args.model_metric}.pth", map_location=args.device))
    print("Model loaded.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_names_mapping[args.tokenizer], use_fast=True)
    tokenizer.model_max_length = model.bert.embeddings.position_embeddings.num_embeddings
    print("Tokenizer loaded.")

    # calculate partial metrics
    for X, y, split in zip([valid_X, test_X], [valid_y, test_y], ["valid", "test"]):
        print(f"Evaluating {split} set...")
        partial_metrics_l = list()
        assert len(X) % args.eval_partials == 0
        seg_len = len(X) // args.eval_partials
        # evaluate each partial segment
        for seg in range(1, args.eval_partials + 1):
            print(f"Segment {seg}:")
            seg_start_idx = (seg % args.eval_partials) * seg_len
            X_seg = X[seg_start_idx:seg_start_idx + seg_len]
            y_seg = y[seg_start_idx:seg_start_idx + seg_len]
            seg_dataset = MedicalDxDataset(emrs=X_seg, dx_labels=y_seg, tokenizer=tokenizer)
            seg_loader = DataLoader(dataset=seg_dataset, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True, collate_fn=seg_dataset.collate_fn)
            metrics = evaluate_dx_model(model, seg_loader, args.device, verbose=True)
            partial_metrics_l.append(metrics)
            print(metrics)
        # merge results of the split
        incremental_metrics = merge_dicts(partial_metrics_l)

        # save evaluation results
        df = pd.DataFrame(data=incremental_metrics, index=[(i / args.eval_partials) for i in range(1, args.eval_partials + 1)]).rename_axis("prop", axis="index")
        save_path = Path(args.exp_path) / "eval_results"
        save_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path / f"incremental_{split}_{args.model_metric}.csv", index=True)        

def merge_dicts(dicts: List[dict]) -> Dict[str, list]:
    merged = {k: list() for k in dicts[0].keys()}
    for d in dicts:
        for k, v in d.items():
            merged[k].append(v)

    return merged

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--args_path",
        type=str,
        help="Path to training args"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Evaluation device"
    )

    parser.add_argument(
        "--eval_partials",
        type=int,
        default=10,
        help="How many partials are used for evaluation?"
    )

    parser.add_argument(
        "--model_metric",
        type=str,
        default="hat3",
        help="Which model (saved for different best metrics during training) to use?"
    )

    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    cmd_args = parse_args()

    args = load_pickle(cmd_args.args_path)
    args.device = cmd_args.device
    args.eval_partials = cmd_args.eval_partials
    args.model_metric = cmd_args.model_metric

    main(args)