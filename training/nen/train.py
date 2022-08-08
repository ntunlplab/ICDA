import gc
import sys
sys.path.append("../../")

import time
import yaml
import json
import pickle
from tqdm import tqdm
from typing import Tuple
from pathlib import Path
from argparse import Namespace, ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from utilities.preprocess import train_valid_test_split
from utilities.data import BertNENDataset, KBEntities
from utilities.model import BiEncoder, encoder_names_mapping
from utilities.utils import load_json, save_json, load_yaml, load_pickle, set_seeds, render_exp_name, move_bert_input_to_device
from utilities.evaluation import evaluate_nen

def main(args: Namespace) -> None:
    # CONFIGURATION
    # Experiment setup
    args.exp_name = render_exp_name(args, hparams=args.hparams, sep="__")
    args.exp_dir = f"{args.save_dir}/{args.exp_name}"

    Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.exp_dir) / "config_nen.yml").write_text(data=yaml.dump(vars(args)))

    set_seeds(args.seed)

    # DATA
    # Load data
    emrs = load_pickle(args.emrs_path)
    nen_spans_l = load_pickle(args.nen_spans_l_path)
    icds = load_pickle(args.icds_path) # for splitting data
    sm2cui = load_json(args.sm2cui_path)
    smcui2name = load_json(args.smcui2name_path)

    # Split data
    train_emrs, valid_emrs, test_emrs, _, _, _ = train_valid_test_split(
        inputs=emrs,
        labels=icds,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed
    )
    train_spans_l, valid_spans_l, test_spans_l, _, _, _ = train_valid_test_split(
        inputs=nen_spans_l,
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

    # Dataset
    train_set = BertNENDataset(
        emrs=train_emrs,
        nen_spans_l=train_spans_l,
        mention2cui=sm2cui,
        cui2name=smcui2name,
        cui_batch_size=args.batch_size,
        tokenizer=tokenizer
    )
    valid_set = BertNENDataset(
        emrs=valid_emrs,
        nen_spans_l=valid_spans_l,
        mention2cui=sm2cui,
        cui2name=smcui2name,
        cui_batch_size=args.batch_size,
        tokenizer=tokenizer
    )
    entities_set = KBEntities(
        id2desc=smcui2name,
        tokenizer=tokenizer
    )
    print("Dataset built.")

    # Dataloader
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, pin_memory=True, collate_fn=lambda batch: batch[0])
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, pin_memory=True, collate_fn=lambda batch: batch[0])
    entities_loader = DataLoader(entities_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=entities_set.collate_fn)
    print("Dataloader built.")

    # MODEL
    model = BiEncoder(encoder_name=encoder_names_mapping[args.encoder]).to(args.device)
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

    total_train_steps = len(train_loader) * args.nepochs
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=total_train_steps, num_warmup_steps=int(total_train_steps * args.warmup_ratio))
    print(f"Model, optimizer, and scheduler loaded. Total training steps = {total_train_steps}; total warmup steps = {int(total_train_steps * args.warmup_ratio)}")
    
    # OPTIMIZATION
    step = 0
    best_valid_acc = 0
    train_log = {
        "valid_acc": list(),
        "lr": list(),
        "step": list()
    }

    for epoch in range(args.nepochs):
        print(f"Training at epoch {epoch + 1}...")
        pbar = tqdm(total=len(train_loader))

        for loader_idx, (emr_be, ment_indices_l, target_cuis, negative_cuis_l) in enumerate(train_loader):
            model.train()
            emr_be = move_bert_input_to_device(emr_be, args.device)

            # encode mentions
            h_ments = model.encode_mentions(emr_be, ment_indices_l)
            assert len(h_ments) == len(ment_indices_l) == len(target_cuis) == len(negative_cuis_l)

            # encode entities
            emr_loss = torch.tensor([0.0]).to(args.device)
            for h_ment, target_cui, negative_cuis in zip(h_ments, target_cuis, negative_cuis_l):
                all_cuis = [target_cui] + negative_cuis
                ents_be = train_set.make_entities_be(cuis=all_cuis).to(args.device)
                ents_labels = train_set.make_entities_labels(target_cui, negative_cuis).to(args.device)
                h_ents = model.encode_entities(ents_be)

                # calculate similarity score and loss
                scores = model.calc_scores(h_ment, h_ents)
                loss = model.calc_loss(scores.squeeze(), ents_labels)
                
                # accumulate loss
                emr_loss += (loss / len(h_ments))
            
            # evaluation
            if (step % args.ckpt_steps == 0) or (loader_idx == len(train_loader) - 1):
                valid_acc = evaluate_nen(valid_loader, model, args, entities_loader=entities_loader)
                train_log["valid_acc"].append(valid_acc)
                train_log["lr"].append(optimizer.param_groups[0]["lr"])
                train_log["step"].append(step)
                save_json(train_log, f=Path(args.exp_dir) / "train_log.json")
                pbar.set_description(f"Valid_acc = {valid_acc:.3f}")
                time.sleep(2)

                if valid_acc > best_valid_acc:
                    pbar.set_description("Saving the best model...")
                    best_valid_acc = valid_acc
                    (Path(args.exp_dir) / "best_metric.txt").write_text(f"Best valid accuracy = {best_valid_acc:.4f}")
                    torch.save(model.state_dict(), f=Path(args.exp_dir) / "best_valid_acc.pth")

            # update model parameters
            lr = optimizer.param_groups[0]["lr"]
            optimizer.zero_grad()
            if emr_loss.requires_grad:
                emr_loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_description(f"step loss = {emr_loss.item():.3f}; lr = {lr:.3E}")
            step += 1
    
            pbar.update(n=1)
        pbar.close()

    return None

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device to train on."
    )

    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    args = Namespace(**load_yaml("./config_nen.yml"))
    cmd_args = parse_args()

    for k, v in vars(cmd_args).items():
        setattr(args, k, v)
    
    main(args)