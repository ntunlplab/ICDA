import sys
sys.path.append("../")

from pathlib import Path
from argparse import Namespace

import pandas as pd

import torch
from transformers import AutoTokenizer
from transformers import logging
logging.set_verbosity_error()

from utilities.utils import load_json, set_seeds, build_reverse_dict
from utilities.model import BertNERModel, BiEncoder, BertDxModel, encoder_names_mapping

from icda import ICDA
from finding_extractor import FindingExtractor, Recognizer, Normalizer
from diagnosis_classifier import DiagnosisClassifier
from term_suggester import TermSuggester, UMLSClassifier
from emr_preprocessor import EMRPreprocessor

from flask import Flask, render_template, send_from_directory, request

"""
    Configuration
"""
args = Namespace(
    full_emr_path="../datasets/full_piphs.json",
    unnorm_states_path="../datasets/unnorm_patient_states_t.json",
    norm_states_path="../datasets/norm_patient_states_t.json",
    in_icds_path="../datasets/in_icds.json",
    out_icds_path="../datasets/out_icds.json",

    ner_model_path="../models/ner",
    batch_size=16,

    nen_model_path="../models/nen",

    dx_model_path="../models/dx",
    target_metric="hat5",

    score_matrix_path="../models/term/fisher_matrix_mink-3_minp-0.05.csv",
    term2id_path="../models/term/term2id.json",
    inequality="lesser",
    threshold=0.05,
    ndx=5,

    seed=7,
    train_size=0.8,
    valid_size=0.1,
    test_size=0.1,

    system_mode="deploy",
    extract_mode="umls",
    device="cuda:0"
)

set_seeds(args.seed)

"""
    Load ICDA Modules
"""
# Models
ner_model = BertNERModel(encoder=encoder_names_mapping["BioLinkBERT"], num_tags=5)
ner_model.load_state_dict(torch.load(Path(args.ner_model_path) / "best_model.pth", map_location=args.device))
ner_tokenizer = AutoTokenizer.from_pretrained(Path(args.ner_model_path) / "tokenizer", use_fast=True)

nen_model = BiEncoder(encoder_name=encoder_names_mapping["BERT"])
nen_model.load_state_dict(torch.load(Path(args.nen_model_path) / "best_model.pth", map_location=args.device))
nen_tokenizer = AutoTokenizer.from_pretrained(Path(args.nen_model_path) / "tokenizer", use_fast=True)
entity_embeddings = torch.load(Path(args.nen_model_path) / "entity_embeddings_5454.pt")

cui2name = load_json(Path(args.nen_model_path) / "smcui2name.json")
cui2typeinfo = load_json(Path(args.nen_model_path) / "smcui2typeinfo.json")
cat2typenames = load_json(Path(args.nen_model_path) / "cat2typenames.json")

id2dx = load_json(Path(args.dx_model_path) / "id2icd.json")
dx2name = load_json(Path(args.dx_model_path) / "icdnine2name.json")
dx_model = BertDxModel(encoder_name=encoder_names_mapping["BioLinkBERT"], num_dxs=len(id2dx))
dx_model.load_state_dict(torch.load(Path(args.dx_model_path) / f"best_{args.target_metric}.pth"))
dx_tokenizer = AutoTokenizer.from_pretrained(Path(args.ner_model_path) / "tokenizer", use_fast=True)

fisher_matrix = pd.read_csv(args.score_matrix_path, index_col="term_id")
term2id = load_json(args.term2id_path)
id2term = build_reverse_dict(term2id)

print("Models and required files loaded.")

# Sub-components
recognizer = Recognizer(
    model=ner_model,
    tokenizer=ner_tokenizer,
    batch_size=args.batch_size,
    device=args.device
)

normalizer = Normalizer(
    model=nen_model,
    tokenizer=nen_tokenizer,
    entity_embeddings=entity_embeddings,
    cui2name=cui2name,
    device=args.device,
    emr_batch_size=1,
    cui_batch_size=args.batch_size
)

umls_classifier = UMLSClassifier(
    cui2name=cui2name,
    cui2typeinfo=cui2typeinfo,
    cat2typenames=cat2typenames
)

# Components
finding_extractor = FindingExtractor(
    recognizer=recognizer,
    normalizer=normalizer
)

emr_preprocessor = EMRPreprocessor(
    finding_extractor=finding_extractor
)

dx_classifier = DiagnosisClassifier(
    model=dx_model,
    tokenizer=dx_tokenizer,
    id2dx=id2dx,
    dx2name=dx2name,
    batch_size=args.batch_size,
    device=args.device
)

term_suggester = TermSuggester(
    score_matrix=fisher_matrix,
    id2term=id2term,
    inequality=args.inequality,
    threshold=args.threshold,
    diagnosis_classifier=dx_classifier,
    umls_classifier=umls_classifier,
    top_k_dxs=args.ndx
)

print("Modules loaded.")

icda = ICDA(
    system_mode=args.system_mode,
    extract_mode=args.extract_mode,
    finding_extractor=finding_extractor,
    diagnosis_classifier=dx_classifier,
    term_suggester=term_suggester,
    emr_preprocessor=emr_preprocessor
)

print("ICDA initialized.")

"""
    Back-End Implementation
"""
app = Flask(__name__, static_folder="build/static", template_folder="build")

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/manifest.json")
def manifest():
    return send_from_directory('./build', 'manifest.json')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory('./build', 'favicon.ico')

@app.route("/post", methods=['GET'])
def index():
    # Final return object
    output = dict()

    # Get front-end data
    emr = request.args.get('medicaltext')
    n_icd = int(request.args.get('n_icd'))
    page = int(request.args.get('page'))

    # Inference
    support = icda.generate_support([emr], n_dx=n_icd)[0]

    # NOTE: refactor these
    output["pos_word"] = support["emr_display"]["extracted_terms"]["positive"]
    output["neg_word"] = support["emr_display"]["extracted_terms"]["negative"]
    output["icd"] = [f"{d['icd']} - {d['name']}" for d in support["diagnoses"]]
    output["keyword"] = [
        {
            "ss": d["symptoms"],
            "dx": d["diseases"]
        }
        for d in support["suggested_terms"].values()
    ]

    return output