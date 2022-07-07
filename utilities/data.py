import random
from typing import Iterable, List, Set, Tuple, Dict, Any
from collections import Counter

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, AutoTokenizer, BatchEncoding

class MedicalDxDataset(Dataset):
    def __init__(self, emrs: List[str], dx_labels: List[int], tokenizer: BertTokenizerFast):
        self.x = emrs
        self.y = dx_labels
        assert len(self.x) == len(self.y)
        self.tokenizer=tokenizer
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        # find the indexed data
        emr = self.x[idx]
        dx = self.y[idx]
        # transform
        x = self.tokenizer(emr, truncation=True).convert_to_tensors(tensor_type="pt", prepend_batch_axis=False)
        # target transform
        y = torch.LongTensor([dx])        
        return x, y

    def collate_fn(self, batch):
        batch_x = list()
        batch_y = torch.LongTensor()

        for x, y in batch:
            batch_x.append(x)
            batch_y = torch.cat(tensors=(batch_y, y), dim=0)
        batch_x = self.tokenizer.pad(batch_x)

        return batch_x, batch_y

    @property
    def num_dx_labels(self):
        return len(Counter(self.y))

class MedicalIOBPOLDataset(Dataset):
    def __init__(
            self, 
            text_l: List[str], 
            ner_spans_l: List[Dict[Tuple, int]], 
            tokenizer: AutoTokenizer, 
            ignore_index: int = -100,
            return_offsets: bool = False
        ):
        assert len(text_l) == len(ner_spans_l)

        self.text_l = text_l
        self.ner_spans_l = ner_spans_l
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.return_offsets = return_offsets
        # for label / index conversion
        self.span_labels = {0: None, 1: "POS", 2: "NEG"}
        self.iob2idx = {
            "O": 0,
            "B-POS": 1,
            "I-POS": 2,
            "B-NEG": 3,
            "I-NEG": 4,
            "[PAD]": -100
        }
        self.idx2iob = {idx: label for label, idx in self.iob2idx.items()}
    
    def __len__(self):
        return len(self.text_l)
    
    def __getitem__(self, idx):
        # get x & y
        text = self.text_l[idx]
        span_pol_dict = self.ner_spans_l[idx]
        # convert span labels to token labels
        text_be = self.tokenizer(text, truncation=True, return_offsets_mapping=True).convert_to_tensors(tensor_type="pt", prepend_batch_axis=False)
        if self.return_offsets:
            offsets = text_be["offset_mapping"].tolist()
        else:    
            offsets = text_be.pop("offset_mapping").tolist()
        iob_labels = self.bert_offsets_to_iob_labels(offsets, span_pol_dict)

        return text_be, iob_labels
    
    @property
    def num_tags(self):
        return len(self.iob2idx) - 1
    
    def bert_offsets_to_iob_labels(self, offsets: List[List[int]], span_pol_dict: Dict[tuple, int]):
        seen_spans = set()
        iob_labels = list()
        span_pol_dict[(int(1e8), int(1e8))] = 0 # add dummy span for convenience
        span_pols = span_pol_dict.items()
        
        offsets_it = iter(offsets)
        span_pols_it = iter(span_pols)
        offset = next(offsets_it)
        span, pol = next(span_pols_it)
        while True:
            start, end = offset
            if (start == 0) and (end == 0):  # [CLS] or [SEP] token
                label = '[PAD]'
            elif start < span[0]: # offset is to the left of span
                label = 'O'
            elif end <= span[1]: # in: i.e. start >= span[0] & end <= span[1]
                if span not in seen_spans: # 'B'eginning of span
                    seen_spans.add(span)
                    label = f"B-{self.span_labels[pol]}"
                else: # 'I' span
                    label = f"I-{self.span_labels[pol]}"
            else: # i.e. end > span[1]
                span, pol = next(span_pols_it) # move to the next span
                continue
            # if the span is the same then execute the following codes
            iob_labels.append(label)
            try:
                offset = next(offsets_it)
            except StopIteration:
                break
        
        iob_labels = [self.iob2idx[label] for label in iob_labels]
        return iob_labels
    
    def collate_fn(self, batch):
        batch_x, batch_y = list(), list()
        for x, y in batch:
            batch_x.append(x)
            batch_y.append(y)

        if self.return_offsets:
            batch_offsets = [x.pop("offset_mapping").tolist() for x in batch_x]
        batch_x = self.tokenizer.pad(batch_x)

        dim_after_padding = batch_x["input_ids"].shape[1]
        for i in range(len(batch_y)):
            to_fill_length = dim_after_padding - len(batch_y[i])
            padding = torch.ones(to_fill_length, dtype=torch.long) * self.ignore_index
            batch_y[i] = torch.cat((torch.LongTensor(batch_y[i]), padding), dim=0)
        batch_y = torch.stack(batch_y)

        return (batch_x, batch_y, batch_offsets) if self.return_offsets else (batch_x, batch_y)

class BertNENDataset(Dataset):
    def __init__(
            self, 
            emrs: List[str], 
            ner_spans_l: List[List[Tuple[int]]], 
            mention2cui: Dict[str, str], 
            cui2name: Dict[str, str], 
            cui_batch_size: int,
            tokenizer: BertTokenizerFast
        ):

        assert len(emrs) == len(ner_spans_l)
        # attributes
        self.emrs = emrs
        self.ner_spans_l = ner_spans_l
        self.mention2cui = mention2cui
        self.cui2name = cui2name
        self.cuis = list(cui2name.keys())
        self.cui_batch_size = cui_batch_size
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.emrs)
    
    def __getitem__(self, idx): # -> BE (EMR), token_indices_l (NER), CUIs (NEN), negative samples (NEN)
        emr = self.emrs[idx]
        ner_spans = self.ner_spans_l[idx]
        # Filter ner_spans according to mention2cui & construct entity labels
        mapped_ner_spans = list()
        cuis = list()
        for ner_span in ner_spans:
            start, end = ner_span
            emr_span = emr[start:end].lower().strip()
            if emr_span in self.mention2cui:
                mapped_ner_spans.append(ner_span)
                cuis.append(self.mention2cui[emr_span])

        be = self.tokenizer(emr, truncation=True, return_offsets_mapping=True)
        offsets = be.pop("offset_mapping")
        token_indices_l = self.spans_to_token_indices_l(mapped_ner_spans, offsets)
        
        be = be.convert_to_tensors("pt", prepend_batch_axis=True)
        # Align token_indices_l & CUIs (remove empty list)
        assert len(token_indices_l) == len(cuis)
        token_indices_l_clean, cuis_clean = list(), list()
        for token_indices, cui in zip(token_indices_l, cuis):
            if token_indices != []:
                token_indices_l_clean.append(token_indices)
                cuis_clean.append(cui)
        # Prepare negative samples
        negative_cuis_l = list()
        for cui in cuis_clean:
            negative_cuis = self.random_negative_sampling(target_cui=cui, batch_size=self.cui_batch_size)
            negative_cuis_l.append(negative_cuis)

        assert len(token_indices_l_clean) == len(cuis_clean) == len(negative_cuis_l)
        return be, token_indices_l_clean, cuis_clean, negative_cuis_l
    
    def random_negative_sampling(self, target_cui: str, batch_size: int):
        random_cuis = random.sample(self.cuis, k=batch_size)
        negatives = list()
        for random_cui in random_cuis:
            if random_cui != target_cui:
                negatives.append(random_cui)
        return negatives[:batch_size - 1] # return 'batch_size - 1' (e.g. 15) negative samples
    
    def make_entities_be(self, cuis: List[str]) -> BatchEncoding:
        entities = [self.cui2name[cui] for cui in cuis]
        be = self.tokenizer(entities, padding=True, return_tensors="pt")
        return be

    @staticmethod
    def spans_to_token_indices_l(spans: List[Tuple[int]], offsets: List[Tuple[int]]) -> List[List[int]]:
        token_idx = 1 # start with the first token (skip [CLS])
        token_indices_l = list()
        for span in spans:
            token_indices = list()
            while True:
                offset = offsets[token_idx]
                start, end = offset
                if (start == 0) and (end == 0): # [CLS] or [SEP] token
                    break
                elif (start < span[0]):
                    token_idx += 1
                elif (end <= span[1]): # start >= span[0]
                    token_indices.append(token_idx)
                    token_idx += 1
                else: # end > span[1]
                    break
            token_indices_l.append(token_indices)
        
        return token_indices_l
    
    @staticmethod
    def make_entities_labels(target_cui: str, negative_cuis: List[str]) -> torch.FloatTensor:
        all_cuis = [target_cui] + negative_cuis
        labels = list()
        for cui in all_cuis:
            label = cui == target_cui
            labels.append(label)
        
        return torch.FloatTensor(labels)

class BertNENDatasetForTest(Dataset):
    def __init__(
            self, 
            emrs: List[str], 
            ner_spans_l: List[List[Tuple[int]]], 
            cui2name: Dict[str, str], 
            cui_batch_size: int,
            tokenizer: BertTokenizerFast
        ):

        assert len(emrs) == len(ner_spans_l)
        # attributes
        self.emrs = emrs
        self.ner_spans_l = ner_spans_l
        self.cui2name = cui2name
        self.cuis = list(cui2name.keys())
        self.cui_batch_size = cui_batch_size
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.emrs)
    
    def __getitem__(self, idx): # -> BE (EMR), token_indices_l (NER), CUIs (NEN), negative samples (NEN)
        emr = self.emrs[idx]
        ner_spans = self.ner_spans_l[idx]

        be = self.tokenizer(emr, truncation=True, return_offsets_mapping=True)
        offsets = be.pop("offset_mapping")
        token_indices_l = self.spans_to_token_indices_l(ner_spans, offsets)
        
        be = be.convert_to_tensors("pt", prepend_batch_axis=True)
        return be, token_indices_l
    
    def make_entities_be(self, cuis: List[str]) -> BatchEncoding:
        entities = [self.cui2name[cui] for cui in cuis]
        be = self.tokenizer(entities, padding=True, return_tensors="pt")
        return be

    @staticmethod
    def spans_to_token_indices_l(spans: List[Tuple[int]], offsets: List[Tuple[int]]) -> List[List[int]]:
        token_idx = 1 # start with the first token (skip [CLS])
        token_indices_l = list()
        for span in spans:
            token_indices = list()
            while True:
                offset = offsets[token_idx]
                start, end = offset
                if (start == 0) and (end == 0): # [CLS] or [SEP] token
                    break
                elif (start < span[0]):
                    token_idx += 1
                elif (end <= span[1]): # start >= span[0]
                    token_indices.append(token_idx)
                    token_idx += 1
                else: # end > span[1]
                    break
            token_indices_l.append(token_indices)
        
        return token_indices_l

class KBEntities(Dataset):
    def __init__(self, id2desc: Dict[Any, str], tokenizer: BertTokenizerFast):
        self._id2desc = id2desc
        self._tokenizer = tokenizer
        # Make useful properties
        self._ids = list()
        self._descs = list()
        for id_, desc in self._id2desc.items():
            self._ids.append(id_)
            self._descs.append(desc)
        
        assert len(self._ids) == len(self._descs)
    
    def __len__(self):
        return len(self._ids)
    
    def __getitem__(self, idx: int):
        return self._ids[idx], self._descs[idx]
    
    def collate_fn(self, batch):
        ids = list()
        descs = list()
        for id_, desc in batch:
            ids.append(id_)
            descs.append(desc)

        ents_be = self._tokenizer(descs, padding=True, return_tensors="pt")
        return ids, ents_be

def split_by_div(data: Iterable, fold: int, remainder: int, mode: str) -> list:
    data_l = list()
    for i, item in enumerate(data):
        if mode == "train":
            if i % fold != remainder:
                data_l.append(item)
        elif mode == "valid":
            if i % fold == remainder:
                data_l.append(item)
        else:
            raise ValueError("mode should be either train or valid")
    return data_l