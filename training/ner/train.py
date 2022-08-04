import warnings
import sys
warnings.filterwarnings("ignore")
sys.path.append("../../")

import time
import yaml
import json
import pickle
import random
from tqdm.auto import tqdm
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from argparse import Namespace, ArgumentParser
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from utilities.preprocess import train_valid_test_split
from utilities.data import MedicalIOBPOLDataset
from utilities.utils import set_seeds, load_yaml, load_json, save_json, load_pickle, render_exp_name, move_bert_input_to_device
from utilities.model import BertNERModelWithLoss, encoder_names_mapping
from utilities.evaluation import ids_to_iobs, calc_seqeval_metrics

def main(args: Namespace):
    # Experiment setup
    args.exp_name = render_exp_name(args, args.hparams, sep="__")
    args.exp_dir = f"{args.save_dir}/{args.exp_name}"

    Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.exp_dir) / "args.yml").write_text(data=yaml.dump(vars(args)))

    set_seeds(args.seed)
    
    # DATA
    # Load data
    emrs = load_pickle(args.emrs_path)
    ner_spans_l = load_pickle(args.ner_spans_l_path)
    icds = load_pickle(args.icds_path)
    print("Data loaded.")

    # Split
    train_emrs, valid_emrs, test_emrs, _, _, _ = train_valid_test_split(
        inputs=emrs,
        labels=icds,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed
    )
    train_spans_l, valid_spans_l, test_spans_l, _, _, _ = train_valid_test_split(
        inputs=ner_spans_l,
        labels=icds,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed
    )

    for emrs, spans_l in zip([train_emrs, valid_emrs, test_emrs], [train_spans_l, valid_spans_l, test_spans_l]):
        assert len(emrs) == len(spans_l)

    print(f"Data split. Number of train / valid / test instances = {len(train_emrs)} / {len(valid_emrs)} / {len(test_emrs)}")


    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_names_mapping[args.encoder])
    tokenizer.model_max_length = args.model_max_length
    tokenizer.save_pretrained(Path(args.exp_dir) / "tokenizer")
    print("Tokenizer loaded and saved.")

    # Dataset & Dataloader
    train_set = MedicalIOBPOLDataset(text_l=train_emrs, ner_spans_l=train_spans_l, tokenizer=tokenizer)
    valid_set = MedicalIOBPOLDataset(text_l=valid_emrs, ner_spans_l=valid_spans_l, tokenizer=tokenizer)
    print(f"Dataset built. Number of NER tags = {train_set.num_tags}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=train_set.collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=valid_set.collate_fn)
    print(f"Dataloader built. Number of train / valid steps per set = {len(train_loader)} / {len(valid_loader)}")
    
    # MODEL
    model = BertNERModelWithLoss(encoder=encoder_names_mapping[args.encoder], num_tags=train_set.num_tags).to(args.device)
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

    total_train_steps = len(train_loader) * args.nepochs
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=total_train_steps, num_warmup_steps=int(total_train_steps * args.warmup_ratio))
    print("Model, optimizer, and scheduler loaded.")
    
    # OPTIMIZATION
    train_log = {metric: list() for metric in args.metrics}
    train_log["step"] = list()

    best_target_metric = 0
    step = 0
    print("Start training...")
    for epoch in range(args.nepochs):
        print(f"Training at epoch {epoch + 1}...")
        pbar = tqdm(total=len(train_loader))
        for X, y in train_loader:
            # preparation
            model.train()
            X = move_bert_input_to_device(X, args.device)
            y = y.to(args.device)

            # forward & backward pass
            scores = model(X)
            loss = model.calc_loss(scores, y)
            loss.backward()

            # update model parameters
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            pbar.set_description(f"step loss: {loss.item():.3f}; lr: {lr:.3E}")

            # in-training evaluation
            if (step % args.ckpt_steps) == 0:
                pbar.set_description("Evaluating performance...")
                # inference
                y_pred_raw, y_true_raw, valid_loss = predict_whole_set_ner(valid_loader, model, args.device)
                y_pred, y_true = ids_to_iobs(y_pred_raw, y_true_raw, valid_set)
                token_acc, p, r, f1 = calc_seqeval_metrics(y_true, y_pred).values()
                
                # logging
                pbar.set_description(f"Token_acc = {token_acc:.3f}; Precision = {p:.3f}; Recall = {r:.3f}; F1-score = {f1:.3f}")
                time.sleep(3) # stop a while for showing metrics
                for k, v in zip(list(train_log.keys()), [token_acc, p, r, f1, valid_loss, step]):
                    train_log[k].append(v)
                save_json(obj=train_log, f=Path(args.exp_dir) / "train_log.json")

                if train_log[args.target_metric][-1] > best_target_metric:
                    pbar.set_description("Saving the best model...")
                    best_target_metric = train_log[args.target_metric][-1]
                    (Path(args.exp_dir) / "best_target_metric.txt").write_text(f"{args.target_metric} = {str(best_target_metric)}")
                    torch.save(model.state_dict(), f=Path(args.exp_dir) / f"best_{args.target_metric}.pth")

            pbar.update(n=1)
            step += 1

        pbar.close()

def predict_whole_set_ner(data_loader: DataLoader, model: BertNERModelWithLoss, device: str) -> Tuple[list, list, float]:
    y_pred_raw = list()
    y_true_raw = list()
    total_loss = 0

    model = model.to(device)
    model.eval()
    for X, y in data_loader:
        X = move_bert_input_to_device(X, device)
        y = y.to(device)
        
        with torch.no_grad():
            scores = model(X)
            loss = model.calc_loss(scores, y)

        # record loss
        total_loss += loss.detach().cpu().item() * y.shape[0]
        # record predictions
        pred = scores.argmax(dim=-1).detach().cpu().tolist()
        y_pred_raw.append(pred)
        # record ground truth
        true = y.detach().cpu().tolist()
        y_true_raw.append(true)

    mean_loss = total_loss / len(data_loader.dataset)
    return y_pred_raw, y_true_raw, mean_loss

def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="The device for training the model."
    )

    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    args = Namespace(**load_yaml("./config_ner.yml"))
    cmd_args = parse_args()

    for k, v in vars(cmd_args).items():
        setattr(args, k, v)

    main(args)