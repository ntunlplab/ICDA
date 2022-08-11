import warnings
import sys
warnings.filterwarnings("ignore")
sys.path.append("../../")

from typing import Tuple

import torch

from pathlib import Path
from argparse import Namespace, ArgumentParser
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utilities.preprocess import train_valid_test_split
from utilities.data import MedicalIOBPOLDataset
from utilities.utils import set_seeds, load_yaml, save_json, load_pickle, move_bert_input_to_device
from utilities.model import BertNERModelWithLoss, encoder_names_mapping
from utilities.evaluation import ids_to_iobs, calc_seqeval_metrics

def main(args: Namespace):
    # Experiment setup
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

    print(f"Data split. Number of valid / test instances = {len(valid_emrs)} / {len(test_emrs)}")


    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Path("../../training/ner/") / args.exp_dir / "tokenizer")
    tokenizer.model_max_length = args.model_max_length
    print("Tokenizer loaded.")

    # Dataset & Dataloader
    valid_set = MedicalIOBPOLDataset(text_l=valid_emrs, ner_spans_l=valid_spans_l, tokenizer=tokenizer)
    test_set = MedicalIOBPOLDataset(text_l=test_emrs, ner_spans_l=test_spans_l, tokenizer=tokenizer)
    print(f"Dataset built. Number of NER tags = {valid_set.num_tags}")

    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=valid_set.collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=test_set.collate_fn)
    print(f"Dataloader built.")
    
    # MODEL
    model = BertNERModelWithLoss(encoder=encoder_names_mapping[args.encoder], num_tags=test_set.num_tags).to(args.device)
    model.load_state_dict(state_dict=torch.load(Path("../../training/ner/") / args.exp_dir / f"best_{args.target_metric}.pth", map_location=args.device))
    print("Model checkpoint loaded.")
    
    # EVALUATION
    metrics = dict()
    for split, dataloader in zip(["valid", "test"], [valid_loader, test_loader]):
        # inference
        y_pred_raw, y_true_raw, _ = predict_whole_set_ner(dataloader, model, args.device)
        y_pred, y_true = ids_to_iobs(y_pred_raw, y_true_raw, dataloader.dataset)
        token_acc, p, r, f1 = calc_seqeval_metrics(y_true, y_pred).values()
    
        # logging
        metrics[split] = {
            "token_acc": token_acc,
            "precision": p,
            "recall": r,
            "f1_score": f1
        }
    
    save_json(obj=metrics, f=Path("../../training/ner/") / args.exp_dir / "eval_results.json")
    print("Evaluation results saved.")

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
        "--config_path",
        type=str,
        required=True,
        help="Configuration file of the model to be evaluated."
    )
    
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="The device for evaluating the model."
    )

    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    cmd_args = parse_args()
    args = Namespace(**load_yaml(cmd_args.config_path))

    for k, v in vars(cmd_args).items():
        setattr(args, k, v)

    main(args)