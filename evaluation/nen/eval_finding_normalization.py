import os
import sys
sys.path.append("../../")

from pathlib import Path
from argparse import Namespace, ArgumentParser

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from utilities.preprocess import train_valid_test_split
from utilities.data import BertNENDataset, KBEntities
from utilities.model import BiEncoder, encoder_names_mapping
from utilities.utils import load_json, save_json, load_yaml, load_pickle, set_seeds, render_exp_name, move_bert_input_to_device
from utilities.evaluation import evaluate_nen

def main(args: Namespace) -> None:
    # CONFIGURATION
    set_seeds(args.seed)
    exp_path = Path("../../training/nen/") / args.exp_dir

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

    print(f"Data split. Number of valid / test instances = {len(valid_emrs)} / {len(test_emrs)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(exp_path / "tokenizer")
    tokenizer.model_max_length = args.model_max_length
    print("Tokenizer loaded.")

    # Dataset
    valid_set = BertNENDataset(
        emrs=valid_emrs,
        nen_spans_l=valid_spans_l,
        mention2cui=sm2cui,
        cui2name=smcui2name,
        cui_batch_size=args.batch_size,
        tokenizer=tokenizer
    )
    test_set = BertNENDataset(
        emrs=test_emrs,
        nen_spans_l=test_spans_l,
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
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, pin_memory=True, collate_fn=lambda batch: batch[0])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True, collate_fn=lambda batch: batch[0])
    entities_loader = DataLoader(entities_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=entities_set.collate_fn)
    print("Dataloader built.")

    # MODEL
    model = BiEncoder(encoder_name=encoder_names_mapping[args.encoder]).to(args.device)
    model.load_state_dict(state_dict=torch.load(exp_path / f"best_valid_acc.pth", map_location=args.device))
    print("Model checkpoint loaded.")

    # EVALUATION
    # Load or build entity embeddings
    ent_embeds_path = exp_path / "entity_embeddings.pt"
    if os.path.exists(ent_embeds_path):
        entity_embeddings = torch.load(ent_embeds_path, map_location=args.device).to(args.device)
        print(f"Entity embeddings loaded. Shape = {entity_embeddings.shape}.")
    else:
        entity_embeddings = model.encode_all_entities(entities_loader, args).to(args.device)
        torch.save(obj=entity_embeddings, f=ent_embeds_path)
        print(f"Entity embeddings built and saved. Shape = {entity_embeddings.shape}.")
    
    # Inference
    accs = dict()
    for split, dataloader in zip(["valid", "test"], [valid_loader, test_loader]):
        acc = evaluate_nen(dataloader, model, args, entity_embeddings=entity_embeddings, entities_loader=entities_loader)
        accs[split] = dict()
        accs[split]["acc"] = acc

    # Save evaluation results
    save_json(obj=accs, f=exp_path / "eval_results.json")
    print("Evaluation results saved.")

    return None

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Configuration file of the model to be evaluated."
    )
    
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="The device for training the model."
    )

    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    cmd_args = parse_args()
    args = Namespace(**load_yaml(cmd_args.config_path))

    for k, v in vars(cmd_args).items():
        setattr(args, k, v)

    main(args)