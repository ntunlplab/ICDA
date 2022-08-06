import sys
sys.path.append("../../")

from pathlib import Path
from argparse import Namespace
from collections import Counter

from utilities.preprocess import train_valid_test_split
from utilities.utils import load_json, save_json, load_yaml, render_exp_name
from utilities.term import build_cooccurrence_matrix, build_fisher_matrix

def main(args: Namespace):
    # Make save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    states = load_json(args.states_path)
    labels = load_json(args.labels_path)
    cui2name = load_json(args.cui2name_path)
    
    # Preprocessing
    # split data
    train_states, valid_states, test_states, train_labels, valid_labels, test_labels = train_valid_test_split(
        inputs=states,
        labels=labels,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed
    )

    # preprocess entity list per note
    entities_l = list()
    for state in train_states:
        entities = set()
        for entity, _ in state: # ignore polarity
            entities.add(entity)
        entities_l.append(entities)

    # build vocab according to input_type
    if args.input_type == "norm":
        vocab = {name: i for i, (cui, name) in enumerate(cui2name.items())}
    elif args.input_type == "unnorm":
        vocab = dict()
        for entities in entities_l:
            for entity in entities:
                if entity not in vocab:
                    vocab[entity] = len(vocab)
    else:
        raise ValueError

    # save vocab
    save_json(obj=vocab, f=f"./{args.save_dir}/{args.input_type}_term2id.json")

    # Build the co-occurrence matrix
    co_matrix = build_cooccurrence_matrix(entities_l, train_labels, vocab)

    # Determine term-label relevance by Fisher's exact test
    label2ndoc = Counter(train_labels)
    total_ndoc = len(train_labels)
    f_matrix = build_fisher_matrix(co_matrix, label2ndoc, total_ndoc, min_ndoc_k=args.mink, min_ndoc_p=args.minp)

    # Save training results
    filename = render_exp_name(args, args.hparams, sep="__") + ".csv"
    f_matrix.to_csv(Path(args.save_dir) / filename, index_label="term_id")

    return None

if __name__ == "__main__":
    args = Namespace(**load_yaml("./config_term.yml"))

    main(args)