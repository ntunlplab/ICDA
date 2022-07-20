from shutil import move
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from torch.distributions.categorical import Categorical
from transformers import AutoTokenizer

from utilities.data import MedicalDxDataset
from utilities.model import BertDxModel
from utilities.utils import move_bert_input_to_device

class DiagnosisClassifier(object):
    def __init__(self, 
            model: BertDxModel, 
            tokenizer: AutoTokenizer,
            id2dx: Dict[int, Any],
            dx2name: Dict[str, str],
            batch_size: int,
            device: str,
            verbose: bool = False
        ):
        self.model = model
        self.tokenizer = tokenizer
        self.id2dx = {int(k): str(v) if len(str(v)) == 3 else '0' * (3 - len(str(v))) + str(v) for k, v in id2dx.items()} # make sure that ids are integers
        self.dx2id = {v: k for k, v in self.id2dx.items()}
        self.dx2name = dx2name
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
    
    def predict(self, emrs: List[str]) -> torch.FloatTensor:
        # make dataset and dataloader
        dataset = MedicalDxDataset(emrs=emrs, dx_labels=[-100] * len(emrs), tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, self.batch_size, shuffle=False, pin_memory=True, collate_fn=dataset.collate_fn)

        all_logits = list()
        self.model = self.model.to(self.device)
        self.model.eval()
        for X, _ in tqdm(dataloader) if self.verbose else dataloader:
            X = move_bert_input_to_device(X, device=self.device)
            with torch.no_grad():
                logits = self.model(X)
                all_logits.append(logits)
        
        all_logits = torch.cat(all_logits, dim=0).cpu()
        return all_logits
    
    def get_top_dxs_with_probs(self, logits: torch.FloatTensor, top_k: int) -> Tuple:
        logits, preds = logits.sort(dim=-1, descending=True)
        dxs_l = [list(map(lambda id_: self.id2dx[id_], pred)) for pred in preds[:, :top_k].tolist()]
        probs_l = softmax(logits, dim=-1)[:, :top_k].tolist()
        
        return dxs_l, probs_l
    
    @staticmethod
    def calc_entropy(logits: torch.FloatTensor) -> torch.FloatTensor:
        dist = Categorical(logits=logits)
        entropies = dist.entropy()

        return entropies