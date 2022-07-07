from tqdm import tqdm
from typing import List, Dict, Union, Tuple

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from utilities.data import MedicalIOBPOLDataset, BertNENDatasetForTest, KBEntities
from utilities.model import BertNERModel, BiEncoder
from utilities.utils import move_bert_input_to_device

class Recognizer(object):
    def __init__(
            self, 
            model: BertNERModel, 
            tokenizer: AutoTokenizer, 
            batch_size: int,
            device: str,
            verbose: bool = False
        ):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.label2token = {
            0: "positive",
            1: "negative"
        }
    
    def extract_labeled_spans(self, emrs: List[str]) -> List[Dict[tuple, int]]:
        text_set = MedicalIOBPOLDataset(
            text_l=emrs,
            ner_spans_l=[dict() for _ in range(len(emrs))], # empty labels
            tokenizer=self.tokenizer,
            return_offsets=True
        )
        text_loader = DataLoader(
            dataset=text_set, 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True, 
            collate_fn=text_set.collate_fn
        )

        iob_spans_l = list()

        self.model = self.model.to(self.device)
        self.model.eval()

        for X, y, batch_offsets in tqdm(text_loader) if self.verbose else text_loader:
            X = move_bert_input_to_device(X, self.device)

            with torch.no_grad():
                scores = self.model(X)
                batch_preds = scores.argmax(dim=-1).detach().cpu().tolist()
            
            for preds, offsets in zip(batch_preds, batch_offsets):
                pred_spans_d = dict()
                for pred, offset in zip(preds, offsets):
                    pred_spans_d[tuple(offset)] = pred
                iob_spans_l.append(pred_spans_d)
        
        pol_spans_l = [self.iob_spans_to_term_spans(iob_spans) for iob_spans in iob_spans_l]
        return pol_spans_l
    
    @staticmethod
    def iob_spans_to_term_spans(iob_spans: Dict[tuple, int]) -> Dict[tuple, int]:
        term_spans = dict()
        term_span = None
        pol_label = 0 # 0: positive; 1: negative
        for iob_span, label in iob_spans.items():
            start, end = iob_span
            if label == 0: # O: out
                if term_span:
                    term_spans[tuple(term_span)] = pol_label
                    term_span = None
            elif label % 2 == 1: # B: beginning
                if term_span:
                    term_spans[tuple(term_span)] = pol_label
                term_span = [start, end]
                pol_label = (label - 1) // 2
            else: # I: in
                if term_span:
                    term_span[1] = end
                else:
                    term_span = [start, end]
                pol_label = (label - 1) // 2
                
        if (0, 0) in term_spans:
            term_spans.pop((0, 0))
        return term_spans

    @staticmethod
    def extract_spans_and_pols(pol_spans_l: List[Dict[tuple, int]]):
        spans_l = list()
        pols_l = list()
        for spans_pols in pol_spans_l:
            spans = list(spans_pols.keys())
            pols = list(spans_pols.values())
            spans_l.append(spans)
            pols_l.append(pols)
        return spans_l, pols_l

class Normalizer(object):
    def __init__(
            self,
            model: BiEncoder,
            tokenizer: AutoTokenizer,
            entity_embeddings: torch.FloatTensor,
            cui2name: Dict[str, str],
            device: str,
            emr_batch_size: int = 1,
            cui_batch_size: int = 16,
            verbose: bool = False
        ):
        self.model = model
        self.tokenizer = tokenizer
        self.entity_embeddings = entity_embeddings
        self.cui2name = cui2name
        self.device = device
        self.emr_batch_size = emr_batch_size
        self.cui_batch_size = cui_batch_size
        self.verbose = verbose

    def normalize_term_spans(self, emrs: List[str], spans_l: List[List[tuple]], mode: str) -> List[List[str]]:
        assert len(emrs) == len(spans_l)
        if mode not in ["umls", "lower"]:
            raise ValueError("mode should be 'umls' or 'lower'")
        
        terms_l = list()
        if mode == "lower":
            for emr, spans in zip(emrs, spans_l):
                terms = list()
                for s, e in spans:
                    term = emr[s:e].lower()
                    terms.append(term)
                terms_l.append(terms)
        elif mode == "umls":
            text_set = BertNENDatasetForTest(
                emrs=emrs,
                ner_spans_l=spans_l,
                cui2name=self.cui2name,
                cui_batch_size=self.cui_batch_size,
                tokenizer=self.tokenizer
            )
            text_loader = DataLoader(
                dataset=text_set,
                batch_size=self.emr_batch_size,
                shuffle=False,
                pin_memory=True,
                collate_fn=lambda batch: batch[0]
            )

            all_y_ents = self.entity_embeddings.to(self.device)
            all_cuis = list(self.cui2name.keys())

            # start normalizing terms
            self.model = self.model.to(self.device)
            self.model.eval()
            for emr_be, mention_indices_l in tqdm(text_loader) if self.verbose else text_loader:
                entities = list()
                if mention_indices_l: # if not empty
                    emr_be = move_bert_input_to_device(emr_be, self.device)
                    with torch.no_grad():
                        # encode mentions
                        y_ments = self.model.encode_mentions(emr_be, mention_indices_l)
                        assert len(y_ments) == len(mention_indices_l)

                        # calculate scores
                        scores = self.model.calc_scores(y_ments, all_y_ents)
                        # convert to concept (entity)
                        preds = scores.argmax(dim=-1).cpu().tolist()
                        for pred in preds:
                            pred_cui = all_cuis[pred]
                            pred_ent = self.cui2name[pred_cui]
                            entities.append(pred_ent)
                
                terms_l.append(entities)            

        return terms_l

class FindingExtractor(object):
    def __init__(
            self,
            recognizer: Recognizer,
            normalizer: Normalizer  
        ):
        self.recognizer = recognizer
        self.normalizer = normalizer
        
    @staticmethod
    def build_patient_states(terms_l: List[List[str]], pols_l: List[List[int]], return_type: bool = "dict") -> Union[List[Dict[str, int]], List[List[Tuple[str, int]]]]:
        assert len(terms_l) == len(pols_l)
        if return_type not in ["dict", "tuple"]:
            raise ValueError("return_type should be 'dict' or 'tuple'")

        states = list()
        for terms, pols in zip(terms_l, pols_l):
            assert len(terms) == len(pols)
            state = dict() if return_type == "dict" else list()
            for term, pol in zip(terms, pols):
                t = (term, pol)
                if return_type == "dict":
                    state[t] = state.get(t, 0) + 1
                elif return_type == "tuple":
                    state.append(t)
            states.append(state)
        
        return states