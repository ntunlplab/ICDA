import json
import logging
from tqdm.auto import tqdm
from typing import Callable
from pathlib import Path
from argparse import Namespace

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import wandb

class ICDATrainer(object):
    def __init__(
            self, 
            model: nn.Module,
            train_set: Dataset,
            valid_set: Dataset,
            # train_collate_fn: Callable,
            # valid_collate_fn: Callable,
            optimizer: torch.optim.Optimizer,
            eval_func: Callable,
            logger: logging.Logger,
            args: Namespace
        ):
        # attributes
        self.model = model
        self.train_set = train_set
        self.valid_set = valid_set
        # self.train_collate_fn = train_collate_fn
        # self.valid_collate_fn = valid_collate_fn
        self.optimizer = optimizer
        self.eval_func = eval_func
        self.logger = logger
        self.args = args

        # configuration
        # path config
        self.exp_path = Path(args.exp_path)
        self.exp_path.mkdir(parents=True, exist_ok=True)
        self.train_losses_path = self.exp_path / "train_losses.csv"
        self.train_losses_path.write_text("step,train_loss\n")
        self.train_log_path = self.exp_path / "train_log.csv"
        self.train_log_path.write_text(','.join(["step"] + self.args.log_metrics) + '\n')
        self.best_metric_path = self.exp_path / "best_metrics.json"
        self.best_models_path = self.exp_path / "best_models"
        self.best_models_path.mkdir(parents=True, exist_ok=True)
        # wandb config
        self.wandb = wandb
        self.wandb.init(
            project=args.project_name,
            name=args.exp_name,
            config={hparam: getattr(args, hparam) for hparam in args.exp_hparams}
        )

    def train(self):
        # prepare dataloader
        self.logger.info("Building dataloaders...")
        train_loader = DataLoader(self.train_set, batch_size=self.args.train_batch_size, shuffle=True, pin_memory=True, collate_fn=self.train_set.collate_fn)
        valid_loader = DataLoader(self.valid_set, batch_size=self.args.valid_batch_size, shuffle=False, pin_memory=True, collate_fn=self.valid_set.collate_fn)

        # train stat trackers
        step = 0
        step_loss = 0
        train_losses = list()
        best_metrics = {metric: -float("inf") for metric in self.args.log_metrics}

        # optimization
        self.logger.info("Start training...")
        self.model = self.model.to(self.args.device)
        for epoch in range(1, self.args.nepochs + 1):
            self.logger.info(f"===== Training at epoch {epoch} =====")
            pbar = tqdm(total=len(train_loader))
            for loader_idx, batch in enumerate(train_loader):
                self.model.train()
                # move to device
                X, y = batch
                X = move_bert_input_to_device(X, self.args.device)
                y = y.to(self.args.device)

                # forward
                logits = self.model(X)
                loss = self.model.calc_loss(logits, y) / self.args.grad_accum_steps

                # backward
                loss.backward()
                step_loss += loss.detach().cpu().item()

                pbar.update(n=1)
                # update step
                if (loader_idx % self.args.grad_accum_steps == self.args.grad_accum_steps - 1) or (loader_idx == len(train_loader) - 1):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # log training loss
                    train_losses.append(step_loss)
                    pbar.set_description(f"Step loss: {step_loss:.3f}") # terminal
                    with self.train_losses_path.open(mode="at") as f: # local file
                        f.write(f"{step},{step_loss}\n")
                    self.wandb.log({"train_loss": step_loss}, step=step) # wandb

                    # evaluate during training
                    if (step % self.args.eval_every_k_steps == 0) or (loader_idx == len(train_loader) - 1):
                        pbar.set_description(f"Evaluating...")
                        metrics = self.eval_func(model=self.model, eval_loader=valid_loader, device=self.args.device)
                        # log eval metrics
                        with self.train_log_path.open(mode="at") as f: # local file
                            f.write(','.join([str(step)] + [str(metrics[k]) for k in self.args.log_metrics]) + '\n')
                        self.wandb.log(metrics, step=step)

                        # save the best models (for every metric)
                        for metric_name in self.args.log_metrics:
                            metric = metrics[metric_name]
                            if metric > best_metrics[metric_name]:
                                best_metrics[metric_name] = metric
                                self.best_metric_path.write_text(json.dumps(best_metrics, indent=4))
                                pbar.set_description(f"Saving the best model ({metric_name})...")
                                torch.save(obj=self.model.state_dict(), f=self.best_models_path / f"best_{metric_name}.pth")

                    # update train stat trackers
                    step += 1
                    step_loss = 0

            pbar.close()

        for metric_name in self.args.log_metrics:
            self.wandb.run.summary[f"best_{metric_name}"] = best_metrics[metric_name]
        self.wandb.finish(exit_code=0)

def move_bert_input_to_device(x, device):
    for k in x:
        x[k] = x[k].to(device)
    return x