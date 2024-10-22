"""
Core function definitions
"""
import json
import os
import sys
import time
import copy
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings
import torch
import numpy as np
import matplotlib
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from torch_geometric.loader import DataLoader

from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction
from gears.data_utils import get_DE_genes
sys.path.insert(0, "../")

import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
    masked_relative_error,
)
from scgpt.tokenizer import tokenize_batch, pad_batch, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id, compute_perturbation_metrics, get_pert_flags, get_gene_names

import random
import collections
import pickle
import argparse
import pandas as pd
import scanpy as sc
import scipy
import psutil
from torch_geometric.data import Data
from sklearn.decomposition import PCA
import umap

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def train(model: nn.Module, train_loader: torch.utils.data.DataLoader, n_genes: int, gene_ids: np.array, criterion: "criterion function", scaler: torch.cuda.amp.GradScaler, optimizer: torch.optim.Adam, scheduler: torch.optim.lr_scheduler.StepLR, logger: "logger", epoch: int, gene_idx_map: dict, random_shuffle: bool, always_keep_pert_gene: bool, loss_type: str, var: dict) -> dict:
    """
    Train the model for one epoch.
    """
    model.train()
    epoch_loss = 0.0
    epoch_mse_loss = 0.0
    interval_loss, interval_mse = 0.0, 0.0
    start_time = time.time()
    num_batches = len(train_loader)
    if loss_type == "mse+triplet":
        t_loss = torch.nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)) ##can use cosine distance instead of Euclidean...
        condition_map = get_condition_map_from_loader(train_loader) ##tensors will be on CPU to save GPU memory 
        epoch_triplet_loss = 0.0
    if loss_type == "mse+pearson":
        epoch_pearson_loss = 0.0
    for batch, batch_data in enumerate(train_loader):
        input_gene_ids, mapped_input_gene_ids, input_values, input_pert_flags, src_key_padding_mask, target_values = batch_data_to_tensors(batch_data, var, n_genes, gene_ids, gene_idx_map, random_shuffle, always_keep_pert_gene, subsample=True)
        with torch.cuda.amp.autocast(enabled=var["amp"]):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=var["CLS"],
                CCE=var["CCE"],
                MVC=var["MVC"],
                ECS=var["ECS"],
            )
            output_values = output_dict["mlm_output"]
            masked_positions = torch.ones_like(input_values, dtype=torch.bool)  # Use all
            loss = loss_mse = criterion(output_values, target_values, masked_positions)
            epoch_mse_loss += loss.item()
            if loss_type == "mse+triplet":
                tl = get_triplet_loss(src=mapped_input_gene_ids, input_values=input_values, input_pert_flags=input_pert_flags, target_values=target_values, perts=batch_data.pert, condition_map=condition_map, input_gene_ids=input_gene_ids, t_loss=t_loss, model=model, device=var["device"], amp=var["amp"], not_perturbed_id=var["not_perturbed_id"], sample_ctrl_loader=True) ##sample loader for training
                loss = loss + tl ##scale of tl seems to be one order higher? can multiply by .10 to put in same order of magnitude as MSE to have about equal influence
                epoch_triplet_loss += tl.item()
            if loss_type == "mse+pearson":
                pearson_loss = pearson_corr_loss(output_values - input_values, target_values - input_values)
                loss = loss + pearson_loss
                epoch_pearson_loss += pearson_loss.item()
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()
        interval_loss += loss.item()
        epoch_loss += loss.item()
        interval_mse += loss_mse.item()
        if batch % var["log_interval"] == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / var["log_interval"]
            cur_loss = interval_loss / var["log_interval"]
            cur_mse = interval_mse / var["log_interval"]
            print(f"cur loss: {cur_loss}, interval_loss: {interval_loss}")
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"cur_loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
            )
            interval_loss = 0
            interval_mse = 0
            start_time = time.time()
    print(f"total epoch loss: {epoch_loss}")
    print(f"loss / sample size: {epoch_loss / float(len(train_loader.dataset))}")
    print(f"    total epoch mse loss: {epoch_mse_loss}")
    if loss_type == "mse+triplet":
        print(f"    total epoch triplet loss: {epoch_triplet_loss}")
        return {"epoch_loss": epoch_loss / float(len(train_loader.dataset)), "epoch_triplet_loss": epoch_triplet_loss / float(len(train_loader.dataset))}
    if loss_type == "mse+pearson":
        print(f"    total epoch pearson loss: {epoch_pearson_loss}")
        return {"epoch_loss": epoch_loss / float(len(train_loader.dataset)), "epoch_pearson_loss": epoch_pearson_loss / float(len(train_loader.dataset))}
    return {"epoch_loss": epoch_loss / float(len(train_loader.dataset))}

class MeanPredictor():
    """
    Simple model class for predicting mean expression, method definitions mirror scGPT for sake of consistency for downstream use
    """
    def __init__(self, pert_data, data_name, mean_type):
        self.pert_data = pert_data
        if not os.path.isfile(f"pickles/training_mean_{mean_type}_expression_{data_name}.pkl"): ##if saved average training expression does not exist write it 
            self.mean_expression = self.get_mean_expression(self.pert_data.dataloader["train_loader"], mean_type=mean_type) ##for sake of fairness with other models that don't have access to the val/test set upon training, the average should only be computed from the training set 
            pickle.dump(self.mean_expression, open(f"pickles/training_mean_{mean_type}_expression_{data_name}.pkl", "wb"))
        else: 
            ##from one run to the next of the training loader, mean will be slightly different because drop_last == True for pertdata and training loader is shuffled; for sake of consistency, return the pickled mean
            loaded = pickle.load(open(f"pickles/training_mean_{mean_type}_expression_{data_name}.pkl", "rb"))
            self.mean_expression = loaded
        self.tiled_mean_expression = torch.from_numpy(np.tile(self.mean_expression, (256, 1))) ##convenience / speed up: compute once and extract it in pred_perturb instead of tiling over and over for each batch
    
    def get_mean_expression(self, loader, mean_type="control+perturbed"):
        for batch, batch_data in enumerate(loader):
            if batch == 0:
                matrix = torch.empty((0, batch_data.y.shape[1]))
            target = batch_data.y ##perturbed cell expression if pert != ctrl, control cell expression if pert == ctrl
            if mean_type == "control+perturbed":
                matrix = torch.cat((matrix, target), 0)
            elif mean_type == "control":
                control_indices = [i for i in range(0, len(batch_data.pert)) if batch_data.pert[i] == "ctrl"]
                matrix = torch.cat((matrix, target[control_indices, :]), 0)
            elif mean_type == "perturbed":
                perturbed_indices = [i for i in range(0, len(batch_data.pert)) if batch_data.pert[i] != "ctrl"]
                matrix = torch.cat((matrix, target[perturbed_indices, :]), 0)
            else:
                raise Exception("mean_type must be one of control+perturbed, control, perturbed")
        col_avg = torch.mean(matrix, 0)
        return col_avg      
    
    def pred_perturb(self, batch, gene_ids=None, gene_idx_map=None, var=None):
        """
        Given batch, will just yield mean expression (assumes batch.y column order is in same column order as pert_data.adata.var)
        """
        if len(batch) > 256:
            return torch.from_numpy(np.tile(self.mean_expression, (len(batch), 1))) ##repeat mean expression prediction for each cell
        else:
            return self.tiled_mean_expression[0:len(batch)]

    def test_ordering(self, loader):
        """
        make sure each sc expression in batch.t is present in original pert_data.adata.X expression, would indicate that column ordering of batch is in same ordering as pert_data.adata.X
        """
        for itr, batch in enumerate(loader):
            batch.to("cpu")
            t = batch.y
            for sc_expr in t: 
                sc_expr = sc_expr.numpy()
                found = False
                for array in self.pert_data.adata.X.A:
                    if np.array_equal(array, sc_expr):
                        found = True
                        break
                assert(found == True)
    def eval(self):
        return
    def to(self, device):
        return

class SmartMeanPredictor(MeanPredictor):
    """
    If perturbed gene predict 0, else predict mean
    """
    def __init__(self, pert_data, data_name, mean_type, crispr_type):
        super(SmartMeanPredictor, self).__init__(pert_data, data_name, mean_type)
        self.genes = pert_data.adata.var["gene_name"].tolist()
        if crispr_type == "crispri": ##for crispri: for cells perturbed by gene A, we expect gene A expression to have 0x the expression of gene A expression for cells NOT perturbed by gene A (for a good crispri)
            self.multiplier = 0.0 
        else: ##for crispra, we don't know how much fold change to expect for upregulation - not as clear cut as crispri: keep mean expression 
            self.multiplier = 2.0 

    def pred_perturb(self, batch, gene_ids=None, gene_idx_map=None, var=None):
        """
        Given batch, will just yield mean expression (assumes batch.y column order is in same column order as pert_data.adata.var)
        but for the perturbed gene, wiil yield a prediction of 0
        """
        perturbations = [extract_genes(p) for p in batch.pert]
        indices = [[self.genes.index(pert_gene) for pert_gene in perturbations[i]] for i in range(0, len(perturbations))] ##list of indices list for each perturbation (some perturbations might be compound)
        preds = np.tile(self.mean_expression, (len(batch), 1))
        for i in range(0, len(indices)):
            sub_indices= indices[i]
            for index in sub_indices:
                preds[i][index] = self.multiplier * preds[i][index]
        return torch.from_numpy(preds)

def eval_perturb(
    loader: DataLoader, model: TransformerGenerator, gene_ids: list, gene_idx_map: dict, var: dict, loss_type: str
) -> Dict:
    """
    Run model in inference mode using a given data loader
    """
    model.eval()
    model.to(var["device"])
    total_loss = 0.0
    total_mse_loss = 0.0
    if loss_type == "mse+triplet":
        total_triplet_loss = 0.0
        condition_map = get_condition_map_from_loader(loader)
        t_loss = torch.nn.TripletMarginLoss()
    if loss_type == "mse+pearson":
        total_pearson_loss = 0.0
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {} ##will be dictionary with keys: pred, truth, pred_de, truth_de, containing np arrays with shape (n, # of genes)
    logvar = []
    for itr, batch in enumerate(loader):
        batch.to(var["device"])
        pert_cat.extend(batch.pert)
        with torch.no_grad():
            ##p and t of shapes (batch_size, # of genes), batch.de_idx is list of length batch size, with each item being of length 20
            p = model.pred_perturb(
                batch,
                gene_ids=gene_ids,
                gene_idx_map=gene_idx_map, 
                var=var
            )
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())
            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])
            ##loss
            batch_mse_loss = masked_mse_loss(p, t, torch.ones_like(p, dtype=torch.bool)).item() ##use masked_mse_loss for sake of function call consistency, even though no actual masking
            total_mse_loss += batch_mse_loss 
            total_loss += batch_mse_loss
            if not isinstance(model, MeanPredictor): ##if baseline MeanPredictor don't compute these special losses
                if loss_type == "mse+triplet":
                    input_gene_ids, mapped_input_gene_ids, input_values, input_pert_flags, src_key_padding_mask, target_values = batch_data_to_tensors(batch_data=batch, var=var, n_genes=len(gene_ids), gene_ids=gene_ids, gene_idx_map=gene_idx_map, random_shuffle=False, always_keep_pert_gene=False, subsample=False)
                    batch_triplet_loss = get_triplet_loss(src=mapped_input_gene_ids, input_values=input_values, input_pert_flags=input_pert_flags,target_values=target_values, perts=batch.pert, condition_map=condition_map, input_gene_ids=input_gene_ids, t_loss=t_loss, model=model, device=var["device"], amp=var["amp"], sample_ctrl_loader=False).item() ##validation and test loaders do NOT have ctrl pert, so set sample_ctrl_loader=False
                    total_triplet_loss += batch_triplet_loss
                    total_loss += batch_triplet_loss
                if loss_type == "mse+pearson":
                    reshaped_x = torch.reshape(batch.x, p.shape)
                    pearson_loss = pearson_corr_loss(p - reshaped_x, t - reshaped_x).item()
                    total_pearson_loss += pearson_loss
                    total_loss += pearson_loss
    # all genes
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(np.float32)
    results["truth"] = truth.detach().cpu().numpy().astype(np.float32)
    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(np.float32)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(np.float32)
    if loss_type == "mse+triplet":
        results["avg_triplet_loss"] = total_triplet_loss / float(len(loader.dataset)) ##keep track of mse_loss too (total loss / sample size in this case)
    if loss_type == "mse+pearson":
        results["avg_pearson_loss"] = total_pearson_loss / float(len(loader.dataset))
    results["avg_mse_loss"] = total_mse_loss / float(len(loader.dataset)) ##keep track of mse_loss too (total loss / sample size in this case)
    results["avg_loss"] = total_loss / float(len(loader.dataset))
    return results

def get_variables(load_model=None, config_path=None):
    """
    Reads config file and returns dictionary of variables
    """
    with open(config_path, "r") as f:
        var = json.load(f)
    var["device"] = torch.device(var["device"])
    var["load_model"] = load_model
    print(var)
    return var

def get_model_setup(var, pert_data, logger):
    """
    Take var, pert_data (modifies in place), logger
    return model_file, vocab, n_genes, gene_ids, ntokens
    """
    if var["load_model"] is not None:
        model_dir = Path(var["load_model"])
        model_file = model_dir / "best_model.pt"
        vocab_file = "models/scgpt-pretrained/scGPT_human/vocab.json" ##for now just use the vocabulary that scGPT authors shipped out 
        vocab = GeneVocab.from_file(vocab_file)
        for s in var["special_tokens"]:
            if s not in vocab:
                vocab.append_token(s)
        pert_data.adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]]
        gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        genes = pert_data.adata.var["gene_name"].tolist()
    else:
        genes = pert_data.adata.var["gene_name"].tolist()
        vocab = Vocab(VocabPybind(genes + special_tokens, None))  # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array([vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int)
    n_genes = len(genes)
    n_tokens = len(vocab)
    ntokens = len(vocab)  # size of vocabulary
    return model_file, vocab, n_genes, gene_ids, ntokens

def get_module_names_for_lora(model):
    module_names = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.modules.linear.Linear) or isinstance(mod, torch.nn.Embedding): ##take any module that LoRa supports (so far Linear and Embedding according to LoRa authors)
            module_names.append(name)
    return module_names

def load_model(var, model, model_file, logger, attention_control, freeze_input_encoder, freeze_transformer_encoder, mode, use_lora, lora_rank, pretrain_control, transformer_encoder_control, input_encoder_control):
    """
    Take var, model, model_file, logger, attention_control
    return model 
    """
    if use_lora and mode == "test": ##if testing a LoRa model, LoRa-fy the model first, then load the saved weights (which have LoRa-fied names, requires that model_file is saved LoRa weights), can also look into set_peft_model_state_dict but this does the job of reproducing exact test metrics 
        from peft import get_peft_model, LoraConfig
        module_names = get_module_names_for_lora(model)
        logger.info(f"LoRa target modules: {module_names}")
        peft_config = LoraConfig(target_modules=module_names, inference_mode=True, r=lora_rank, lora_alpha=32, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    model_dict = model.state_dict()
    if pretrain_control:
        logger.info("WARNING: no pretrained model weights loaded")
    else: ##load weights 
        pretrained_dict = torch.load(model_file)
        if var["use_fast_transformer"] == False: ##taken from scgpt.utils.load_pretrained to account for no flash_attn case
            pretrained_dict = {k.replace("Wqkv.", "in_proj_"): v for k, v in pretrained_dict.items()}
        ##depending on different controls, will filter out the pretrained_dict of weights that we are not going to load into the model
        if attention_control: 
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "self_attn" not in k}
        if transformer_encoder_control:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "transformer_encoder" not in k}
        if input_encoder_control:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "encoder.embedding" not in k and "encoder.enc_norm" not in k and "value_encoder" not in k}
        ##check to see if any keys are in the pretrained weights but NOT in the instantiated model
        in_pretrained_but_not_in_instantiated = [(key, pretrained_dict[key].shape) for key in pretrained_dict if key not in model_dict]
        if len(in_pretrained_but_not_in_instantiated) > 0:
            print(f"WARNING: the following keys are in the pretrained weights but NOT in the instantiated model: {in_pretrained_but_not_in_instantiated}")
        if var["load_param_prefixs"] is not None and var["load_model"] is not None:
            logger.info(f"loading model:  {model_file}")
            # only load params that start with the prefix
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if any([k.startswith(prefix) for prefix in var["load_param_prefixs"]])
            }
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        elif var["load_model"] is not None:
            try:
                model.load_state_dict(pretrained_dict)
                logger.info(f"Loading all model params from {model_file}")
            except:
                # only load params that are in the model and match the size
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                for k, v in pretrained_dict.items():
                    logger.info(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

        ##tests for each weight of model to return: if it's not present in pretrained then warn the user, if it is present then make sure it's the same value as the pretrained value
        state_dict = model.state_dict()
        in_instantiated_but_not_pretrained = []
        for key in state_dict:
            if key not in pretrained_dict: 
                in_instantiated_but_not_pretrained.append(key)
                continue
            if not np.array_equal(state_dict[key].cpu().numpy(), pretrained_dict[key].cpu().numpy()):
                raise Exception(f"mismatch with model weight key: {key}")
        if len(in_instantiated_but_not_pretrained) > 0:
            logger.info(f"WARNING: the following keys are in the instantiated model but not in the pretrained weights {model_file}: {in_instantiated_but_not_pretrained}")
    if freeze_input_encoder or freeze_transformer_encoder: ##always keep pert encoder unfrozen because no pert encoder in human CP foundation 
        keywords_to_freeze = []
        if freeze_input_encoder:
            keywords_to_freeze = keywords_to_freeze + ["encoder.embedding", "encoder.enc_norm", "value_encoder"]
        if freeze_transformer_encoder:
            keywords_to_freeze.append("transformer_encoder")
        frozen = []
        for name, param in model.named_parameters():
            for ktf in keywords_to_freeze:
                if ktf in name:
                    frozen.append(name)
                    param.requires_grad = False
        logger.info(f"WARNING: Froze parameters: {frozen}")
    if use_lora and mode == "train": ##if training using lora, we want to wrap it / LoRa-fy it AFTER we load the appropriate saved weights 
        from peft import get_peft_model, LoraConfig
        module_names = get_module_names_for_lora(model)
        logger.info(f"LoRa target modules: {module_names}")
        peft_config = LoraConfig(target_modules=module_names, inference_mode=False, r=lora_rank, lora_alpha=32, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    return model

def filter_pertdata_perturbations(pert_data):
    """
    Given PertData object, will filter the rows of the PertData's AnnData object such that rows with a perturbed gene NOT part of the gene set will be removed
    Removes rows in place and returns None 
    """
    conditions = list(pert_data.adata.obs['condition'])
    gene_perturbations = [get_gene_names(condition) for condition in conditions] ##list of lists corresponding to the perturbed gene(s) for each cell, gene_perturbations[i] ==  list of genes perturbed for cell i, len(gene_perturbations) == length of adata
    gene_set = set(pert_data.adata.var['gene_name'].values) ##set of genes present in dataset 
    row_bool = []
    for i in range(0, len(gene_perturbations)):
        present = True
        for j in range(0, len(gene_perturbations[i])): ##iterate over each gene in nested list 
            if gene_perturbations[i][j] not in gene_set:
                present = False
                break 
        if present:
            row_bool.append(1)
        else:
            row_bool.append(0)
    row_bool = np.array(row_bool)
    pert_data.adata = pert_data.adata[row_bool > 0, :]

def modify_pertdata_anndata(pert_data):
    """
    Given PertData, will modify it's .adata object in-place to just retain cells with perturbations that are part of the PertData.AnnData's gene set. For combo perturbations, if a single gene perturbation of a cell is not present in the dataset's genes, will remove that cell. 
    """
    gene_set = set(pert_data.adata.var['gene_name'].values) ##set of genes present in dataset 
    perturbations = pert_data.adata.obs["condition"]
    indices_to_keep = []
    for i in range(0, len(pert_data.adata)):
        if pert_data.adata.obs["condition"][i] == "ctrl":
            indices_to_keep.append(i)
        else:
            present = True 
            gene_names = get_gene_names(pert_data.adata.obs["condition"][i])
            for gene in gene_names: 
                if gene not in gene_set:
                    present = False
                    break
            if present:
                indices_to_keep.append(i)
    print("before filter anndata: ", len(pert_data.adata))
    pert_data.adata = pert_data.adata[indices_to_keep]
    print("after: ", len(pert_data.adata))

def modify_pertdata_dataloaders(pert_data, logger=None):
    """
    Given PertData, will modify it's train, val, and test loaders in-place to just retain cells with perturbations that are part of the PertData.AnnData's gene set. For combo perturbations, if a single gene perturbation of a cell is not present in the dataset's genes, will remove that cell. 
    """
    if logger != None:
        logger.info(f"len adata: {len(pert_data.adata)}")
        for load_type in ["train", "val", "test"]:
            logger.info(f"    old {load_type} loader length: {len(pert_data.dataloader[f'{load_type}_loader'])}")
    gene_set = set(pert_data.adata.var['gene_name'].values) ##set of genes present in dataset 
    old_dataloaders = pert_data.dataloader #{'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader}
    new_dataloaders = {}
    shuffle = {"train": True, "val":True, "test":False}
    for load_type in ["train", "val", "test"]:
        new_data = []
        skipped = set()
        old_loader = old_dataloaders[f"{load_type}_loader"]
        for batch, batch_data in enumerate(old_loader): ##batch_data is of type torch_geometric.data.batch.DataBatch, batch_data[i] is of type torch_geometric.data.data.Data
            for i in range(0, len(batch_data)):
                if batch_data.pert[i] == "ctrl": ##always keep the ctrl condition
                    new_data.append(batch_data[i])
                else:
                    present = True 
                    gene_names = get_gene_names(batch_data.pert[i])
                    for gene in gene_names: 
                        if gene not in gene_set:
                            present = False
                            skipped.add(gene)
                            break
                    if present: 
                        new_data.append(batch_data[i])
        if logger != None: 
            logger.info(f"    filtered the following {len(skipped)} genes from the {load_type} loader: {skipped}")
        new_loader = DataLoader(new_data, batch_size=old_loader.batch_size, shuffle=shuffle[load_type])
        new_dataloaders[f"{load_type}_loader"] = new_loader 
    pert_data.dataloader = new_dataloaders
    if logger != None:
        for load_type in ["train", "val", "test"]:
            logger.info(f"    new {load_type} loader length: {len(pert_data.dataloader[f'{load_type}_loader'])}")

def check_args(opt):
    if opt.pretrain_control == True and opt.mode == "test":
        raise Exception("opt.pretrain_control == True and opt.mode == test")

def check_dictionary_match(d1, d2, logger):
    for key in d1:
        if key in d2 and d1[key] != d2[key]:
            logger.info(f"WARNING: mismatch key, value for key: {key}, values: {d1[key]}, {d2[key]}")

def convert_ENSG_to_gene(ensg_list):
    """
    Given a list in ENSG format, will convert to gene name and return
    """
    gene_dict = pd.read_table('genes.tsv', header=None, index_col=None)
    gene_dict = dict(gene_dict[[0, 1]].values)
    if 'ENSG' not in ensg_list[0] or 'ENSG' not in ensg_list[1]:
        raise Exception("ENSG format not detected!")
    new_list = [gene_dict.get(x, 'unknown') for x in ensg_list]
    return new_list 

def get_replogle_gwps_pert_data(split, batch_size, test_batch_size, generate_new=False):
    if generate_new: 
        adata = sc.read_h5ad("/home/wongd26/mlcs/mlhub/single_cell/scPerturb/selected/ReplogleWeissman2022_K562_gwps.h5ad") ##from scPerturb 
        adata.var["gene_name"] = list(adata.var.index) ##this particular adata is raw, already has gene_name as var.index, adata.X is already a np.ndarray
        adata.obs["condition"] = convert_pert_to_condition_format(list(adata.obs["perturbation"])) ##to conform to GEARS condition format 
        adata.obs["cell_type"] = ["K562"] * len(adata)
        adata.X = scipy.sparse.csr_matrix(adata.X) ##convert numpy to sparse
        sc.pp.log1p(adata) ##scGPT authors report log1p pre-processing
        pert_data = PertData("./data")
        init_time = time.time()
        pert_data.new_data_process(dataset_name="replogle_k562_gwps", adata=adata)
        print(f"time for new_data_process: {time.time() - init_time}")
        init_time = time.time()
        pert_data.load(data_path = './data/replogle_k562_gwps') # to load the processed data
        print(f"time for load: {time.time() - init_time}")
        init_time = time.time()
        pert_data.prepare_split(split=split, seed=1)
        print(f"time for prepare_split: {time.time() - init_time}")
        init_time = time.time()
        pert_data.get_dataloader(batch_size=batch_size, test_batch_size=test_batch_size)
        print(f"time for get_dataloader: {time.time() - init_time}")
        return pert_data
    else:
        if os.path.isfile("data/replogle_k562_gwps/perturb_processed.h5ad"):
            pert_data = PertData("./data")
            pert_data.load(data_path = './data/replogle_k562_gwps')
            pert_data.prepare_split(split=split, seed=1)
            pert_data.get_dataloader(batch_size=batch_size, test_batch_size=test_batch_size)
            return pert_data
        else:
            raise Exception("data/replogle_k562_gwps/perturb_processed.h5ad does not exist! generate new")

def convert_pert_to_condition_format(pert_list):
    """
    For consistency with GEARS:
    Given list of perturbation genes: ["A", "B", "A+B", "control"], will convert to format: ["A+ctrl", "B+ctrl", "A+B", "ctrl"]
    """
    new = []
    for i in range(0, len(pert_list)):
        if pert_list[i] == "control":
            new.append("ctrl")
        elif "+" in pert_list[i]: ##double perturbation, format is already correct
            new.append(pert_list[i])
        else:
            new.append(pert_list[i] + "+ctrl")
    return new 

def convert_pert_to_condition_name_format(pert_list, cell_type):
    """
    For consistency with GEARS:
    Given list of perturbation genes: ["A", "B", "A+B", control] with celltype K562, will convert to format: ["K562_A+ctrl_1+1", "K562_B+ctrl_1+1", "K562_A+B_1+1", "K562_ctrl_1"]
    """
    new = []
    for i in range(0, len(pert_list)):
        if pert_list[i] == "control":
            new.append(f"{cell_type}_ctrl_1")
        elif "+" in pert_list[i]:
            new.append(f"{cell_type}_{pert_list[i]}_1+1")
        else:
            new.append(f"{cell_type}_{pert_list[i]}_1+1")
    return new

def extract_genes(string):
    """
    Given a string, will return the genes as list: e.g.
    A+ctrl --> [A]
    A+B -> [A, B]
    """
    genes = string.split("+")
    return [g for g in genes if g!="ctrl"]

def merge_loaders(loader_list, batch_size=64, shuffle=True):
    """
    Given a list of dataloaders, will  merge them into one and return
    """
    new_data = []
    for loader in loader_list:
        for batch, batch_data in enumerate(loader):
            for i in range(0, len(batch_data)):
                new_data.append(batch_data[i])
    new_loader = DataLoader(new_data, batch_size=batch_size, shuffle=shuffle)
        
def get_condition_map_from_loader(loader):
    """
    Returns a map with key: perturbation, value: list of perturbed cell expressions for that perturbation
    Includes ctrl condition if part of loader
    """
    mapp = {}
    for batch, batch_data in enumerate(loader):
        for i in range(0, len(batch_data)):
            pert = batch_data.pert[i]
            if pert not in mapp:
                mapp[pert] = [batch_data.y[i]]
            else:
                mapp[pert].append(batch_data.y[i])
    return mapp 

def get_triplet_loss(src=None, input_values=None, input_pert_flags=None, target_values=None, perts=None, condition_map=None, input_gene_ids=None, t_loss=None, model=None, device=None, amp=None, not_perturbed_id=None, sample_ctrl_loader=True):
    """
    Input tensors column selected by indices in input_gene_ids: 
        src = gene tokens, input_values = control expression, input_pert_flags = pert flags, target_values = perturbed expression
    perts = sequence of perturbations corresponding to target_values
    Get the control cell expressions, the perturbed expressions, and the sampled perturbed expressions fromm condition_map,
    Compute the embeddings of each of the three, and return the triplet loss (anchor = embedding of original perturbed expression, positive = embedding of sampled perturbed expression, and negative = embedding of control expression)
    """
    #extract expression vectors 
    if sample_ctrl_loader: ##get control samples from dataloader 
        sampled_controls = get_randomly_chosen_conditions_tensor(["ctrl"] * len(input_values), condition_map, input_gene_ids).to(device)
    else: ##get control samples from same batch 
        sampled_controls = input_values[torch.randperm(input_values.size()[0])]
    column_shuffled_pert_flags = input_pert_flags[:,torch.randperm(input_pert_flags.size()[1])]
    src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool, device=device)
    with torch.cuda.amp.autocast(enabled=amp): ##provides massive speedup 
        anchor = model._get_cell_emb_from_layer(model._encode(src, input_values, input_pert_flags, src_key_padding_mask))
        positive = model._get_cell_emb_from_layer(model._encode(src, sampled_controls, input_pert_flags, src_key_padding_mask))
        negative = model._get_cell_emb_from_layer(model._encode(src, input_values, column_shuffled_pert_flags, src_key_padding_mask))
    # calculate triplet loss and return 
    loss = t_loss(anchor=anchor, positive=positive, negative=negative)
    return loss

def get_randomly_chosen_conditions_tensor(perts, condition_map, input_gene_ids):
    """
    Given perts (a list of perturbations) will return a tensor of perturbed cell expressions PE such that PE[i] corresponds to the same perturbation as perts[i], 
    there is no guarantee though that will be a different cell
    """
    l = []
    for pert in perts:
        selection = random.choice(condition_map[pert])
        l.append(selection[input_gene_ids.cpu()]) ##subselect the returned 1-D tensor by input_gene_ids
    return torch.stack(l)
    
def batch_data_to_tensors(batch_data=None, var=None, n_genes=None, gene_ids=None, gene_idx_map=None, random_shuffle=None, always_keep_pert_gene=None, subsample=None):
    """
    Converts batch_data from PertData loader to tensors on device ready for input to model
    subsample=True if want a subsample of genes <= max_seq_len, else take the whole set

    batch_data.x is tensor of control expression of shape (batch_size * n_genes, 1) but is supposed to be of shape (batch_size * n_genes, 2) according to original scGPT implementation which used old GEARS loader, where second dimensions is supposed to be the pert_flag data of shape (batch_size * n_genes)
    batch_data.y is tensor of actual perturbed expression of shape (batch_size, n_genes)
        if batch_data.pert[i]==ctrl then batch_data.y[i] == reshaped batch_data.x[i]
    batch_data.pert_idx is a list with size (batch_size) of single element lists, e.g. [[9118], [7688], [-1], [-1]]
    batch_data.pert is a list of strings with size (batch_size), e.g. ['YIPF5+ctrl', 'SRP68+ctrl', 'ctrl', 'ctrl']
    batch_data.de_idx is a list of np arrays
    original (broken) line from authors:
        pert_flags = x[:, 1].long().view(batch_size, n_genes)
    """
    batch_size = len(batch_data.y)
    batch_data.to(var["device"])  
    x: torch.Tensor = batch_data.x
    ori_gene_values = x[:, 0].view(batch_size, n_genes)
    pert_flags = get_pert_flags(batch_data, var["device"], batch_size, n_genes, gene_idx_map, random_shuffle, pert_pad_id=var["pert_pad_id"], not_perturbed_id=var["not_perturbed_id"], is_perturbed_id=var["is_perturbed_id"])
    target_gene_values = batch_data.y  # (batch_size, n_genes)
    if var["include_zero_gene"] == "all":
        input_gene_ids = torch.arange(n_genes, device=var["device"], dtype=torch.long)
    else:
        input_gene_ids = (ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0])
    # sample input_gene_id
    if subsample and len(input_gene_ids) > var["max_seq_len"]:
        if always_keep_pert_gene: ##option to always include the perturbed_gene's expression, AND the flag indicating which gene was perturbed (if not control)
            x_pert, y_pert = (pert_flags == var["is_perturbed_id"]).nonzero(as_tuple=True) ##tuple of (x coords, y coords) where we have pert_flag == 2, just care about the y_coords here in this case indicating gene
            pert_set = set(y_pert.tolist())
            selected = list(pert_set)
            remaining = [i for i in range(0, n_genes) if i not in selected]
            random.shuffle(remaining)
            while len(selected) < var["max_seq_len"]: ##if perturbed genes is not enough to fill up max_seq_len, pull remaining at random
                selected.append(remaining.pop())
            input_gene_ids = torch.tensor(selected, device=var["device"]) 
        else: ##original sampling from scGPT authors (regardless of pert_flag), seems like a flaw: on any given sample we do not always tell the model which gene was perturbed...we only give the model mostly a subset of genes that are NOT perturbed and say that it was NOT perturbed... 
            input_gene_ids = torch.randperm(len(input_gene_ids), device=var["device"])[:var["max_seq_len"]]
    input_values = ori_gene_values[:, input_gene_ids]
    input_pert_flags = pert_flags[:, input_gene_ids]
    target_values = target_gene_values[:, input_gene_ids]
    ##suppose PertData gene order is G1, G2, G3 ...
    ##input_gene_ids: randomly selected genes of length |max_seq_length| from PertData   e.g. [63, 54, 12 ... ]
    ##gene_ids: vocab ids for each gene in PertData, in the order of the genes in PertData  = [vocab[G1], vocab[G2], ...]
    ##mapped_input_gene_ids = gene_ids[input_gene_ids], vocab tokens in same order as input_values
    ##src_key_padding_mask is given to torch.nn.TransformerEncoderLayer forward function, 0 will mean no masking
    ##mapped_input_gene_ids.shape, input_values.shape, input_pert_flags.shape = (batch_size, # of genes)
    mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids) 
    mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)
    src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool, device=var["device"])
    return input_gene_ids, mapped_input_gene_ids, input_values, input_pert_flags, src_key_padding_mask, target_values

def pearson_corr_loss(outputs, targets):
    """
    Calculates and returns the negative pearson correlation loss
    """
    vx = outputs - torch.mean(outputs)
    vy = targets - torch.mean(targets)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return -1 * cost

def update_loss_map(loss_map, train_loss, val_res):
    """
    Function to update loss_map in place with train and val loss
    """
    for key in train_loss:
        if key not in loss_map["train"]:
            loss_map["train"][key] = [train_loss[key]]
        else:
            loss_map["train"][key].append(train_loss[key])
    for key in val_res:
        if "loss" in key:
            if key not in loss_map["val"]:
                loss_map["val"][key] = [val_res[key]]
            else:
                loss_map["val"][key].append(val_res[key])
    return loss_map 
    
def check_pert_split(data_name, pert_data):
    """
    test to see if perturbation split among train, val, test are constant
    """
    splits = ["train", "val", "test"]
    split_to_perts = {split: set() for split in splits}
    for load_type in splits:
        loader = pert_data.dataloader[f"{load_type}_loader"]
        for batch, batch_data in enumerate(loader):
            for i in range(0, len(batch_data)):
                pert = batch_data.pert[i]
                split_to_perts[load_type].add(pert)
    if os.path.isfile(f"pickles/{data_name}_perturbation_splits.pkl"): ##check if perturbations splits are the same if pickle already exists
        existing_map = pickle.load(open(f"pickles/{data_name}_perturbation_splits.pkl", "rb"))
        for split in existing_map:
            assert(existing_map[split] == split_to_perts[split])
    else: ##if doesn't exist, make it 
        pickle.dump(split_to_perts, open(f"pickles/{data_name}_perturbation_splits.pkl", "wb"))

def perturb_predict(model: TransformerGenerator, pert_list: List[str], pool_size: Optional[int] = None, pert_data: PertData = None, var: dict = None, gene_ids: list = None, gene_idx_map: dict = None, average: bool = True) -> Dict:
    """
    Predict the gene expression values for the given perturbations. Helper function for plot_perturbation
    Args:
        model (:class:`torch.nn.Module`): The model to use for prediction.
        pert_list (:obj:`List[str]`): The list of perturbations to predict.
        pool_size (:obj:`int`, optional): For each perturbation, use this number
            of cells in the control and predict their perturbation results. Report
            the stats of these predictions. If `None`, use all control cells.
    """
    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)
    gene_list = pert_data.gene_names.values.tolist()
    for pert in pert_list:
        for i in pert:
            if i not in gene_list:
                raise ValueError(f"Perturbation: {pert}; The gene is not in the perturbation graph. Please select from GEARS.gene_list!")
    if isinstance(model, MeanPredictor):
        device = torch.device("cpu")
    else:
        model.eval()
        device = next(model.parameters()).device
    with torch.no_grad():
        results_pred = {}
        for pert in pert_list:
            cell_graphs = create_cell_graph_dataset_for_prediction(pert, ctrl_adata, gene_list, device, num_samples=pool_size)
            loader = DataLoader(cell_graphs, batch_size=var["eval_batch_size"], shuffle=False)
            preds = []
            for batch_data in loader:  #batch_data.x.shape = (batch_size * n_genes), batch_data.pert = [[pert], [pert]]
                ##adjust batch_data.x and batch_data.pert to be in correct format
                batch_data.x = torch.reshape(batch_data.x,(batch_data.x.shape[0],1)) ##append 1 dimension 
                batch_data.pert = [x[0] if len(x) == 1 else "+".join(x) for x in batch_data.pert] ##unpack inner list of batch_data.pert, if singleton just extract it, if double perturbation format as "A+B"
                batch_data.pert = convert_pert_to_condition_format(batch_data.pert)
                pred_gene_values = model.pred_perturb(batch_data, gene_ids=gene_ids, gene_idx_map=gene_idx_map, var=var)  
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            if average: ##get average of the columns: e.g. average the cell expression predictions
                results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)
            else: ##keep unaggregated
                results_pred["_".join(pert)] = preds.detach().cpu().numpy()
    return results_pred

def set_box_color(bp, color, plt):
    """
    Helper function color boxplots
    """
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def get_complete_data_structure_for_perturbation(model: nn.Module, pert_list: list, save_directory: str = None, pert_data: PertData = None, var: dict = None, gene_ids: list = None, gene_idx_map: dict = None, data_name: str = None, model_type: str = None) -> dict:
    """
    Convenience function to get most everything we'd want from a perturbation and 
    return as dictionary with key: pert, value: dictionary with keys: all_genes, all_truth, all_ctrl_means, all_pred, de_idx, de_genes, de_truth, de_ctrl_means, de_pred, pert_string
    """
    pert_map = {}
    for query in pert_list:
        all_genes, all_truth, all_ctrl_means, all_pred, de_idx, de_genes, de_truth, de_ctrl_means, de_pred, pert_string = get_truth_and_preds(pert_data, query, model, var, gene_ids, gene_idx_map)
        pert_map[query] = {"all_genes": all_genes, "all_truth": all_truth, "all_ctrl_means": all_ctrl_means, "all_pred": all_pred, "de_idx": de_idx, "de_genes": de_genes, "de_truth": de_truth, "de_ctrl_means": de_ctrl_means, "de_pred": de_pred, "pert_string": pert_string}
    all_perturbed_means = pert_data.adata[pert_data.adata.obs["condition"] != "ctrl"].to_df().mean().values
    pert_map["actual_mean_perturbed"] = all_perturbed_means
    return pert_map

def get_truth_and_preds(pert_data, query, model, var, gene_ids, gene_idx_map):
    """
    Helper function for different plots, 
    returns all_genes, all_truth, all_ctrl_means; de_idx, de_genes, de_truth, de_ctrl_means; pred; pert_string
    """
    ##output: all_genes, all_truth, all_ctrl_means,    de_idx, de_genes, de_truth, de_ctrl_means,     pred, pert_string
    adata = pert_data.adata
    gene2idx = pert_data.node_map ##key: normal gene name, value: index 
    cond2name = dict(adata.obs[["condition", "condition_name"]].values) ##key: condition, value: condition_name
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values)) ##key: ENSG gene name, value: normal gene name
    de_idx = [gene2idx[gene_raw2id[i]] for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]] ##adata.uns["top_non_dropout_de_20"][cond2name[query]] is a list of ENSG genes
    
    all_genes = [gene_raw2id[i] for i in adata.var.index.values] ##alternatively just take adata.var.gene_name.values, seems redundant...
    all_truth = adata[adata.obs.condition == query].X.toarray()
    all_ctrl_means = adata[adata.obs["condition"] == "ctrl"].to_df().mean().values

    de_genes = [gene_raw2id[i] for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]] ##or: de_genes = np.array(all_genes)[de_idx]
    de_truth = all_truth[:, de_idx]
    de_ctrl_means = adata[adata.obs["condition"] == "ctrl"].to_df().mean()[de_idx].values
    
    if "ctrl" in query: ##singleton perturbation 
        parsed = [x for x in query.split("+") if x != "ctrl"] ##list of perturbation strings without control
        assert(len(parsed) == 1)
        wrapped_pert_list = [[x] for x in parsed] #wrapped_pert_list = [query.split("+")[0]] ##original implementatation from authors, but does not account for the case of ctrl being in index 1 instead of 0!
        pert_string = parsed[0]
    else: ##combination perturbation 
        wrapped_pert_list = [query.split("+")]
        pert_string = "_".join(query.split("+"))

    pool_size = sum([1 for i in range(0, len(adata.obs["condition"])) if adata.obs["condition"][i] == query]) ##number of cells = query condition in the dataset, use this as the pool_size, which will be the number of randomly sampled control cells (without replacement) for use in forward prediction
    ##get predictions by sampling pool size # of control cells and running them through model 
    ##key: perturbation, value: prediction array of shape (# cells perturbed by perturbation, all genes) if average = False, else shape (all genes,)
    pred = perturb_predict(model, wrapped_pert_list, pool_size=pool_size, pert_data=pert_data, var=var, gene_ids=gene_ids, gene_idx_map=gene_idx_map, average=False) 
    all_pred = pred[pert_string]
    de_pred = pred[pert_string][:, de_idx]
    
    return all_genes, all_truth, all_ctrl_means, all_pred, de_idx, de_genes, de_truth, de_ctrl_means, de_pred, pert_string

def get_rank(model: nn.Module, pert_list: list, pert_data: PertData = None, var: dict = None, gene_ids: list = None, gene_idx_map: dict = None) -> tuple:
    """
    Wrapper for a call to compute_rank, organizes input data structure for non-GEARS models
    returns dictionary with key: perturbation condition, value: rank 
    """
    ##key: pert, value: rank
    pert_map = {query: () for query in pert_list} 
    for query in pert_list:
        all_genes, all_truth, all_ctrl_means, all_pred, de_idx, de_genes, de_truth, de_ctrl_means, de_pred, pert_string = get_truth_and_preds(pert_data, query, model, var, gene_ids, gene_idx_map)
        pert_map[query] = np.mean(all_truth, axis=0), np.mean(all_pred, axis=0)
    return compute_rank(pert_map)
   
def compute_rank(pert_map: dict = None) -> tuple:
    """
    Given pert_map  with key: pert, value: (actual avg truth vector, predicted avg vector), 
    will return dictionary with key: perturbation condition, value: rank 
    """
    pert_list = list(set(pert_map.keys()))
    ##key: query1, key: query 2, value:cosine sim between query1 truth and query2 prediction 
    rank_map = {query1: {query2: () for query2 in pert_list} for query1 in pert_list} 
    for query1 in pert_map: 
        for query2 in pert_map:
            rank_map[query1][query2] = 1.0 - scipy.spatial.distance.cosine(pert_map[query1][0], pert_map[query2][1])
    ##compute the rank score for each pert
    ranks = {} ##key: perturbation condition, value: rank 
    for query in pert_map:
        sorted_items = sorted(rank_map[query].items(), key=lambda x: x[1], reverse=True) ##reverse to keep consistent with Wu et. al, want most similar at earlier indices
        length = len(sorted_items)
        perts = [x[0] for x in sorted_items]
        my_index = perts.index(query)##index of where query is in sorted_items
        rank = my_index / float(length)
        ranks[query] = rank 
    return ranks 

def plot_perturbation_boxplots(model: nn.Module, pert_list: list, save_directory: str = None, pert_data: PertData = None, var: dict = None, gene_ids: list = None, gene_idx_map: dict = None, data_name: str = None, model_type: str = None) -> tuple:
    """
    Makes boxplots of actual, predicted, (and if plot_control_differential == False: control) for top 20 DE genes 
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    pert_to_gene_iqr = {f"control_differential={boolean}": {query: "" for query in pert_list} for boolean in [True, False]} ##key: control_differential={boolean} key: pert, value: tuple(overall pearson score, dictionary of key: gene, value: boolean is in between first quartile and third quartile of actual expression)
    for query in pert_list:
        all_genes, all_truth, all_ctrl_means, all_pred, de_idx, de_genes, de_truth, de_ctrl_means, de_pred, pert_string = get_truth_and_preds(pert_data, query, model, var, gene_ids, gene_idx_map)
        for plot_control_differential in [True, False]:
            if plot_control_differential: ##subtract mean control expression for these de_genes
                de_pred_plot = de_pred - de_ctrl_means ##shape: (cells, 20)
                de_truth_plot = de_truth - de_ctrl_means
            else:
                de_pred_plot = de_pred
                de_truth_plot = de_truth
            ##get gene to boolean map to see how good the prediction is for each gene 
            sample_distances = []
            gene_iqr_map = {} ##key: gene, value: boolean for if predicted median is in actual interquartile range
            for i in range(0, len(de_genes)):
                actual_q1 = np.percentile(de_truth_plot[:, i], 25)
                actual_q3 = np.percentile(de_truth_plot[:, i], 75)
                pred_median = np.median(de_pred_plot[:, i])
                if actual_q1 <= pred_median <= actual_q3:
                    gene_iqr_map[de_genes[i]] = True
                else:
                    gene_iqr_map[de_genes[i]] = False
                ##wasserstein distance
                sample_distances.append(scipy.stats.wasserstein_distance(de_truth_plot[:, i], de_pred_plot[:, i]))
            distance_score = np.mean(sample_distances)
            pearson_score = scipy.stats.pearsonr(np.mean(de_pred_plot, axis=0), np.mean(de_truth_plot, axis=0))[0] ##will be pearson_de_delta if plot_control_differential, else pearson_de
            ##based off https://stackoverflow.com/questions/16592222/how-to-create-grouped-boxplots 
            fig, ax = plt.subplots(figsize=[16.5, 4.5])
            spacer = 0.4 if plot_control_differential else 0.3
            widths = 0.5 if plot_control_differential else 0.2

            if plot_control_differential:
                ##truth
                bp_truth = plt.boxplot(de_truth_plot, showfliers=False, positions=np.array(range(0, len(de_truth_plot[0])))*2.0-spacer, sym='', widths=widths)
                set_box_color(bp_truth, '#707DBD', plt) # colors are from http://colorbrewer2.org/
                plt.plot([], c='#707DBD', label=f'perturbed cells T={query.replace("+ctrl", "")} (ground truth)') # draw temporary red and blue lines and use them to create a legend
                ##pred
                bp_pred = plt.boxplot(de_pred_plot, showfliers=False, positions=np.array(range(0, len(de_pred_plot[0])))*2.0+spacer, sym='', widths=widths)    
                set_box_color(bp_pred, '#B24288', plt)
                plt.plot([], c='#B24288', label=f'{model_type} prediction for cells T={query.replace("+ctrl", "")}')
            else:
                #control
                ctrl_values = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"].to_df().to_numpy()[:, de_idx] ##np array of control values for de 
                bp_ctrl = plt.boxplot(ctrl_values, showfliers=False, positions=np.array(range(0, len(ctrl_values[0])))*2.0 - spacer, sym='', widths=widths)
                set_box_color(bp_ctrl, 'green', plt) 
                plt.plot([], c='green', label='control cells')
                ##plot expression distribution of all other perturbed cells that weren't perturbed by query
                non_control = pert_data.adata[pert_data.adata.obs["condition"] != "ctrl"]
                other_perturbed_values = non_control[non_control.obs["condition"] != query].to_df().to_numpy()[:, de_idx]
                bp_other_perturbed = plt.boxplot(other_perturbed_values, showfliers=False, positions=np.array(range(0, len(other_perturbed_values[0])))*2.0, sym='', widths=widths)
                set_box_color(bp_other_perturbed, 'orange', plt) 
                plt.rcParams.update({'mathtext.default': 'regular' })
                # plt.plot([], c='orange', label='perturbed $cells_{{{}}}$'.format(f"target≠{query.replace('+ctrl', '')}"))
                plt.plot([], c='orange', label=f'perturbed cells T≠{query.replace("+ctrl", "")}')
                ##truth
                bp_truth = plt.boxplot(de_truth_plot, showfliers=False, positions=np.array(range(0, len(de_truth_plot[0])))*2.0 + spacer, sym='', widths=widths)
                set_box_color(bp_truth, '#707DBD', plt)
                # plt.plot([], c='#707DBD', label='perturbed $cells_{{{}}}$'.format(f"target={query.replace('+ctrl', '')}"))
                plt.plot([], c='#707DBD', label=f'perturbed cells T={query.replace("+ctrl", "")} (ground truth)')
                ##pred                
                bp_pred = plt.boxplot(de_pred_plot, showfliers=False, positions=np.array(range(0, len(de_pred_plot[0])))*2.0 + (2.0*spacer), sym='', widths=widths)    
                set_box_color(bp_pred, '#B24288', plt)
                # plt.plot([], c='#B24288', label='{} Prediction $cells_{{{}}}$'.format(model_type, f"target={query.replace('+ctrl', '')}"))
                plt.plot([], c='#B24288', label=f'{model_type} prediction for cells T={query.replace("+ctrl", "")}')

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
            ax.legend(loc='upper right', prop={"size":8.5}, bbox_to_anchor=(1, 1.37))
            plt.gcf().subplots_adjust(top=.76)

            ticks = de_genes
            plt.xticks(range(0, len(ticks) * 2, 2), ticks, rotation=90, fontsize=12)
            plt.xlim(-2, len(ticks)*2)
            plt.tight_layout()
            if plot_control_differential: 
                plt.title(f"{data_name.replace('_', ' ').title()}: {query}, Pearson DE Delta = {round(pearson_score, 3)}", fontsize=15)
                plt.ylabel("Change in Gene Expression over Control", labelpad=10, fontsize=12)
            else:
                plt.title(f"{data_name.replace('_', ' ').title()}: {query}, Pearson DE = {round(pearson_score, 3)}", fontsize=15)
                plt.ylabel("Gene Expression", labelpad=10, fontsize=12)
            if plot_control_differential: ##plot zero line
                plt.axhline(0, linestyle="dashed", color="grey")
            fig.savefig(os.path.join(save_directory, f"{model_type}_{query}_{plot_control_differential}.png"), bbox_inches="tight", transparent=False, dpi=300)
            pert_to_gene_iqr[f"control_differential={plot_control_differential}"][query] = (pearson_score, distance_score, gene_iqr_map)
    return pert_to_gene_iqr

def plot_perturbation_scatterplots(model: nn.Module, query: str, save_directory: str = None, pert_data: PertData = None, var: dict = None, gene_ids: list = None, gene_idx_map: dict = None, model_type: str = None) -> None:
    """
    Makes many xy scatter plots for perturbation, with different xy permutations drawn from (y, pred, and mean_control)
    colors de_genes red, other genes black
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math
    all_genes, all_truth, all_ctrl_means, all_pred, de_idx, de_genes, de_truth, de_ctrl_means, de_pred, pert_string = get_truth_and_preds(pert_data, query, model, var, gene_ids, gene_idx_map)
    ##get the index/indices of the actual perturbed gene(s)
    if "_" in pert_string: ##combo perturbation for this cell 
        perts = pert_string.split("_")
    else: ##singleton perturbation 
        perts = [pert_string]
    pert_idx_all = np.array([all_genes.index(pert) for pert in perts])
    ##dictionary containing plot information 
    p_m = {
    f"{query}: Actual vs {model_type}\npearson_de={round(scipy.stats.pearsonr(np.mean(de_pred, axis=0), np.mean(de_truth, axis=0))[0], 3)}": 
        {"y": np.mean(all_pred, axis=0), "x": np.mean(all_truth, axis=0), 
        "y_de": np.mean(de_pred, axis=0), "x_de": np.mean(de_truth, axis=0), 
        "y_pert": np.mean(all_pred[:, pert_idx_all], axis=0), "x_pert": np.mean(all_truth[:, pert_idx_all], axis=0),
        "y_label": f"{model_type} Expression", "x_label": "Actual Expression"}, 
    f"{query}: Actual vs Control":
        {"y": all_ctrl_means, "x": np.mean(all_truth, axis=0), 
        "y_de": de_ctrl_means, "x_de": np.mean(de_truth, axis=0), 
        "y_pert": all_ctrl_means[pert_idx_all], "x_pert": np.mean(all_truth[:, pert_idx_all], axis=0),
        "y_label": "Control Expression", "x_label": "Actual Perturbed Expression"},
    f"{query}: Control vs {model_type}":
        {"y": np.mean(all_pred, axis=0), "x": all_ctrl_means, 
        "y_de": np.mean(de_pred, axis=0), "x_de": de_ctrl_means, 
        "y_pert": np.mean(all_pred[:, pert_idx_all], axis=0), "x_pert": all_ctrl_means[pert_idx_all],
        "y_label":  f"{model_type} Expression", "x_label": "Control Expression"},
    f"{query}: Actual - Control vs {model_type} - Control\npearson_de_delta={round(scipy.stats.pearsonr(np.mean(de_pred, axis=0) - de_ctrl_means, np.mean(de_truth, axis=0) - de_ctrl_means)[0], 3)}":
        {"y": np.mean(all_pred, axis=0) - all_ctrl_means, "x": np.mean(all_truth, axis=0) - all_ctrl_means, 
        "y_de": np.mean(de_pred, axis=0) - de_ctrl_means, "x_de": np.mean(de_truth, axis=0) - de_ctrl_means,
        "y_pert": np.mean(all_pred[:, pert_idx_all], axis=0) - all_ctrl_means[pert_idx_all], "x_pert": np.mean(all_truth[:, pert_idx_all], axis=0) - all_ctrl_means[pert_idx_all],
        "y_label": f"{model_type} - Control Expression", "x_label": "Actual - Control Expression"}, 
    }
    for title in p_m:
        ##scatter all_truth vs all_pred
        fig, ax = plt.subplots(figsize=[16.5, 4.5]) 
        ##plot scatter all genes 
        ax.scatter(x=p_m[title]["x"], y=p_m[title]["y"], color="black", label="non-DE gene")
        ##plot scatter de genes in red 
        ax.scatter(x=p_m[title]["x_de"], y=p_m[title]["y_de"], color="red", label="DE gene")
        ##plot the perturbed gene 
        ax.scatter(x=p_m[title]["x_pert"], y=p_m[title]["y_pert"], color="gold", label="perturbed gene")
        ##plot x=y line
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, linestyle="dashed", color="blue", label="x=y")
        ##draw best fit line
        plt.plot(np.unique(p_m[title]["x"]), np.poly1d(np.polyfit(p_m[title]["x"], p_m[title]["y"], 1))(np.unique(p_m[title]["x"])), linestyle="dashed", color="grey", label="best fit (all)")
        ##draw the best de fit line 
        plt.plot(np.unique(p_m[title]["x_de"]), np.poly1d(np.polyfit(p_m[title]["x_de"], p_m[title]["y_de"], 1))(np.unique(p_m[title]["x_de"])), linestyle="dashed", color="lightcoral", label="best fit (de)")
        ##set aspect ratio of axes to be the same, and ticks to be the same
        ax.set_aspect(aspect="equal")
        plt.xticks(np.arange(math.floor(lims[0]), math.ceil(lims[1]) + 1))
        plt.yticks(np.arange(math.floor(lims[0]), math.ceil(lims[1]) + 1))
        ##label the plot and add legends
        plt.title(title, fontsize=12)
        plt.xlabel(p_m[title]["x_label"], labelpad=10, fontsize=12)
        plt.ylabel(p_m[title]["y_label"], labelpad=10, fontsize=12)
        ##figure legend lower right
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
        ax.legend(loc='lower left', prop={"size":6}, bbox_to_anchor=(1.0, 0)) ##if set lower right, will put legend inside plot
        plt.gcf().subplots_adjust(right=.76)
        fig.savefig(os.path.join(save_directory, f"{title.replace(' ', '_')}.png"), bbox_inches="tight", transparent=False)

def get_optimal_pairings(a, b, optimal_pairings="optimal_transport", metric="sqeuclidean", plot=False, use_all_destination=False):
    """
    Given a matrix of source coordinates a, and another matrix of destination coordinates b (can be multi-dimensional),
    return a list of optimal distance pairs (i,j) such that i is row index of a, j is row index of b
    compute pairings either by: 
    if optimal_pairings == "distance": simply pair source to target by shortest distance
    if optimal_pairings == "optimal_transport": pair by solving the OT problem 
    uses squared euclidean distance by default 
    based off: https://pythonot.github.io/auto_examples/plot_OT_2D_samples.html 
    """
    import ot  
    ##a is shape a x d, b is shape b x d
    ##weights for each sample: make uniform 
    a_weight = np.ones((len(a),)) / len(a) ##a x 1
    b_weight = np.ones((len(b),)) / len(b) ##b x 1
    C = ot.dist(a, b, metric=metric) ##default for ot.dist is sqeuclidean distance, a x b
    if optimal_pairings == "distance": ##if want to make pairings based on shortest distance distance
        col_indices = np.argmin(C, axis=1)
    elif optimal_pairings == "optimal_transport": ##if want to make pairings based on solving the optimal transport problem
        ot_emd = ot.emd(a_weight, b_weight, C) ##earth movers distance, will be of shape (a x b)
        col_indices = np.argmax(ot_emd, axis=1)
    else:
        raise Exception("optimal_pairings must be one of distance or optimal_transport")
    pairs = [(i, col_indices[i]) for i in range(0, len(C))]

    if len(set(col_indices)) != len(b):
        print(f"WARNING: pairings are not 1-1. Have {len(a)} source points, missing {len(b) - len(set(col_indices))} destination points out of {len(b)}")
        if use_all_destination:
            print("using all destination cells and adding to closest source cell")
            unused_col_indices = [i for i in range(0, len(b)) if i not in col_indices]
            for unused_index in unused_col_indices:
                ##find closest perturbed cell (row with minimum value)
                distances = C[:, unused_index] ##distance of target to each of the sources 
                min_row = np.argmin(distances, axis=0) ##corresponds to which source has minimal distance to target 
                pairs.append((min_row, unused_index)) ##add to pairs 
    if plot: 
        import ot.plot
        import matplotlib.pyplot as plt  
        fig, ax = plt.subplots()
        ##subset n random a, get corresponding b
        a_subset = np.array(random.sample(list(range(0, len(a))), 100))
        b_subset = np.array(col_indices[a_subset])
        a = a[a_subset]
        b = b[b_subset]
        if optimal_pairings == "distance":
            C = C[a_subset, :]
            C = C[:, b_subset]
        if optimal_pairings == "optimal_transport":
            ot_emd = ot_emd[a_subset, :]
            ot_emd = ot_emd[:, b_subset]
        if len(a[0]) > 2 and len(b[0]) > 2:
            try: 
                reducer = PCA(n_components=2)
                reducer.fit(np.vstack((a, b)))
                a = reducer.transform(a)
                b = reducer.transform(b)
                red_type = "PCA"
            except:
                reducer = umap.UMAP()
                reducer.fit(np.vstack((a, b)))
                a = reducer.transform(a)
                b = reducer.transform(b)
                red_type = "UMAP"
        else:
            red_type = "" ##if already in 2D space, then was reduced prior by either UMAP or PCA 
        if optimal_pairings == "distance":
            ot.plot.plot2D_samples_mat(a, b, C, c=[.5, .5, 1])
        if optimal_pairings == "optimal_transport":
            ot.plot.plot2D_samples_mat(a, b, ot_emd, c=[.5, .5, 1])
        ax.plot(a[:, 0], a[:, 1], '+b', label='Source samples')
        ax.plot(b[:, 0], b[:, 1], 'xr', label='Target samples')
        ax.legend(loc=0)
        plt.title(f'{red_type} OT matrix')
        random_seed = random.randint(0, 1000000000)
        plt.savefig(f"outputs/ot_{optimal_pairings}_{metric}_{random_seed}.png")
    return pairs 

def make_optimal_loaders(pert_data, optimal_pairings="None", load_types=["train"], metric=None, plot=False, use_all_destination=False, optimal_space="raw"):
    """
    Modifies pert_data object by changing the loaders specified by load_types to be optimally paired
    optimal_pairings in ["distance", "optimal_transport"] 
    """
    for load_type in load_types: 
        loader = pert_data.dataloader[f"{load_type}_loader"]
        old_length = len(loader)
        ##iterate over loader and make array for control cells, an array for perturbed cells, a list for keeping track of perturbation, and a map from perturbation to de gene indices 
        de_idx_map = {} ##key: perturbation, value: de_idx; keep track of perturbation differentially expressed indices, cells with same perturbation will have same de_idx by definition because dependent on perturbation 
        control_count = 0 ##number of control perturbations in this loader #for reference 24,263 in entire Adamson dataset (train/val/test)
        for batch, batch_data in enumerate(loader):
            control_indices = [i for i in range(0, len(batch_data.pert)) if batch_data.pert[i] == "ctrl"]
            pert_indices = [i for i in range(0, len(batch_data.pert)) if batch_data.pert[i] != "ctrl"]
            control_count += len(control_indices)
            if load_type in ["val", "test"]: ##in test and val loader, there are not pert=="ctrl", so need to make control array by taking the inputs in .x
                inputs = batch_data.x[:, 0].view(batch_data.y.shape[0], len(batch_data.y[0]))
            if batch == 0: 
                if load_type in ["val", "test"]:
                    control_array = np.copy(inputs)
                else:
                    control_array = np.copy(batch_data.y[control_indices, :]) ##for pert==ctrl, y is the control expression 
                perturbed_array = np.copy(batch_data.y[pert_indices, :])
                perturbation_label = [batch_data.pert[i] for i in pert_indices] ##will keep track of the perturbation label for perturbed_array 
                batch_size = batch_data.y.shape[0]
            else:
                if load_type in ["val", "test"]:
                    control_array = np.concatenate((control_array, inputs), axis=0)
                else:
                    control_array = np.concatenate((control_array, batch_data.y[control_indices, :]), axis=0)
                perturbed_array = np.concatenate((perturbed_array, batch_data.y[pert_indices, :]), axis=0)
                perturbation_label += [batch_data.pert[i] for i in pert_indices]
            for i in range(0, len(batch_data)): ##getting de_idx_map is trickier because batch_data.de_idx is a list of np arrays...
                pert = batch_data.pert[i]
                if pert not in de_idx_map:
                    de_idx_map[pert] = batch_data.de_idx[i]
                else:
                    assert(np.array_equal(de_idx_map[pert], batch_data.de_idx[i]))
        if "ctrl" in de_idx_map and isinstance(de_idx_map["ctrl"], list):
            de_idx_map["ctrl"] = np.array(de_idx_map["ctrl"]) ##ctrl de_idx is list instead of numpy for some reason
        ##create OT pairings between control and perturbed, have source be perturbed and control be destination to ensure that each perturbed cell has a mapping to a (possibly redundant) control cell  
        if optimal_space == "raw":
            pairings = get_optimal_pairings(perturbed_array, control_array, optimal_pairings=optimal_pairings, metric=metric, plot=plot, use_all_destination=use_all_destination)
        elif optimal_space in ["pca", "umap"]:
            reducer = PCA(n_components=50) if optimal_space == "pca" else umap.UMAP(n_components=50)
            reducer.fit(np.vstack((perturbed_array, control_array)))
            pairings = get_optimal_pairings(reducer.transform(perturbed_array), reducer.transform(control_array), optimal_pairings=optimal_pairings, metric=metric, plot=plot, use_all_destination=use_all_destination)
        else:
            raise Exception("optimal space must be one of raw, pca, or umap")
        ##put pairings into new data pairs and make new dataloader 
        new_data = []
        if load_type == "train":
            for i in range(0, control_count):  ##make control datapoints to add to new loader (same # as original loader)
                ##reshaping x to be (g x 1) is necessary, as well as reshaping y to be (1 x g), de_idx is a list of tensors
                entry = Data(x=torch.Tensor(control_array[i].reshape(len(control_array[i]), 1)), y=torch.Tensor(control_array[i].reshape(1, len(control_array[i]))), de_idx=torch.Tensor(de_idx_map["ctrl"].reshape(1, len(de_idx_map["ctrl"]))), pert="ctrl")
                new_data.append(entry)
        for source_c, dest_c in pairings: ##make control<->perturbed matched datapoints
            control_cell = control_array[dest_c]
            perturbed_cell = perturbed_array[source_c]
            pert = perturbation_label[source_c]
            entry = Data(x=torch.Tensor(control_cell.reshape(len(control_cell), 1)), y=torch.Tensor(perturbed_cell.reshape(1, len(perturbed_cell))), de_idx=torch.tensor(de_idx_map[pert].reshape(1, len(de_idx_map[pert]))), pert=pert)  
            new_data.append(entry)
        shuffle = True if load_type == "train" else False
        from torch_geometric.data import DataLoader as tgdl 
        new_loader = tgdl(new_data, batch_size=batch_size, shuffle=shuffle) # new_loader = DataLoader(new_data, batch_size=batch_size, shuffle=shuffle)
        if use_all_destination == False:
            assert(old_length == len(new_loader)) ##make sure new dataloader is same length as old one
        pert_data.dataloader[f"{load_type}_loader"] = new_loader

def convert_pert_flags_to_one_hot(pert_list, one_hot_map):
    """
    Given a list of perturbations and a one hot encoding map with perturbations as keys, will return a tensor of one hot encodings
    """
    one_hot_matrix = []
    for pert in pert_list:
        one_hot_matrix.append(one_hot_map[pert])
    one_hot_matrix = torch.stack(one_hot_matrix, dim=0)
    return one_hot_matrix

class MLP(nn.Module):
    """
    MLP definition from PerturBench paper
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
        norm: str = "layer",
        elementwise_affine: bool = False,
    ):
        """Class for defining MLP with arbitrary number of layers"""
        super(MLP, self).__init__()

        if norm not in ["layer", "batch", None]:
            raise ValueError("norm must be one of ['layer', 'batch', None]")

        layers = nn.Sequential()
        layers.append(nn.Linear(input_dim, hidden_dim))

        if norm == "layer":
            layers.append(
                nn.LayerNorm(hidden_dim, elementwise_affine=elementwise_affine)
            )
        elif norm == "batch":
            layers.append(nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001))

        layers.append(nn.ReLU())
        if dropout is not None:
            layers.append(nn.Dropout(dropout))

        for _ in range(0, n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))

            if norm == "layer":
                layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
            elif norm == "batch":
                layers.append(nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001))

            layers.append(nn.ReLU())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = layers

    def forward(self, x):
        return self.layers(x)

class LatentAdditive(nn.Module):
    """
    Baseline model from PerturBench paper 
    """
    def __init__(self, encoder_input_dim, pert_data, var):
        super().__init__()
        ##default params
        self.encoder_width = 128
        self.latent_dim = 32
        self.n_layers = 2
        self.dropout = None 
        self.softplus_output = True
        
        self.perts = [x for x in set(pert_data.adata.obs["condition"])]
        self.n_perts = len(self.perts)
        self.one_hot_map = {}
        for i in range(0, len(self.perts)):
            one_hot = torch.zeros(self.n_perts, device=var["device"])
            one_hot[i] = 1.0 
            self.one_hot_map[self.perts[i]] = one_hot
        self.gene_encoder = MLP(encoder_input_dim, self.encoder_width, self.latent_dim, self.n_layers, self.dropout)
        decoder_input_dim = self.latent_dim
        self.decoder = MLP(decoder_input_dim, self.encoder_width, encoder_input_dim,self.n_layers, self.dropout)
        self.pert_encoder = MLP(self.n_perts, self.encoder_width, self.latent_dim, self.n_layers, self.dropout)
    
    def forward(self, control_input: torch.Tensor, perturbation: torch.Tensor):
        latent_control = self.gene_encoder(control_input)
        latent_perturbation = self.pert_encoder(perturbation)
        latent_perturbed = latent_control + latent_perturbation
        predicted_perturbed_expression = self.decoder(latent_perturbed)
        if self.softplus_output:
            predicted_perturbed_expression = F.softplus(predicted_perturbed_expression)
        return predicted_perturbed_expression
    
    def train_model(self, train_loader, val_loader, var, gene_ids):
        """
        Simple training definition, keep separate rather than use the train method because much simpler 
        returns best model with lowest MSE over validation set 
        """
        scaler = torch.cuda.amp.GradScaler(enabled=var["amp"])
        optimizer = torch.optim.Adam(self.parameters(), lr=var["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, var["schedule_interval"], gamma=0.9)
        best_model = ""
        best_val_loss = 10000000000000000000
        for epoch in range(0, var["epochs"]):
            epoch_loss = 0.0
            self.train()
            for batch, batch_data in enumerate(train_loader):
                input_values = batch_data.x.to(var["device"])
                input_values = input_values[:, 0].view(len(batch_data.y), len(gene_ids))
                target_values = batch_data.y.to(var["device"])
                one_hot_matrix = convert_pert_flags_to_one_hot(batch_data.pert, self.one_hot_map)
                with torch.cuda.amp.autocast(enabled=var["amp"]):
                    predicted_perturbed_expression = self.forward(input_values, one_hot_matrix)
                    loss = F.mse_loss(predicted_perturbed_expression, target_values)
                self.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
            # print(epoch, epoch_loss)
            latent_res = eval_perturb(val_loader, self, gene_ids=gene_ids, gene_idx_map={}, var=var, loss_type="mse")  
            val_loss = latent_res["avg_mse_loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self)
        return best_model
    
    def pred_perturb(self, batch, gene_ids=None, gene_idx_map=None, var=None):
        """
        Get predicted expression from batch
        Keep method definition the same as scGPT for ease and reusability 
        """
        one_hot_matrix = convert_pert_flags_to_one_hot(batch.pert, self.one_hot_map)
        input_values = batch.x
        input_values = input_values[:, 0].view(len(batch.pert), len(gene_ids))
        predicted_perturbed_expression = self.forward(input_values, one_hot_matrix)
        return predicted_perturbed_expression

class LinearAdditive(LatentAdditive):
    """
    Baseline model from PerturBench paper 
    train_model and pred_perturb will be the exact same as Latent Additive 
    """
    def __init__(self, encoder_input_dim, pert_data, var):
        super(LinearAdditive, self).__init__(encoder_input_dim, pert_data, var)
        self.n_genes = encoder_input_dim
        self.fc_pert = nn.Linear(self.n_perts, self.n_genes)
    
    def forward(self, control_input: torch.Tensor, perturbation: torch.Tensor):
        predicted_perturbed_expression = control_input + self.fc_pert(perturbation)
        if self.softplus_output:
            predicted_perturbed_expression = F.softplus(predicted_perturbed_expression)
        return predicted_perturbed_expression
    
class DecoderOnly(LatentAdditive):
    """
    Baseline model from PerturBench paper 
    train_model and pred_perturb will be the exact same as Latent Additive 
    """
    def __init__(self, encoder_input_dim, pert_data, var):
        super(DecoderOnly, self).__init__(encoder_input_dim, pert_data, var)
        self.n_genes = encoder_input_dim
        self.decoder = MLP(self.n_perts, self.encoder_width, self.n_genes, self.n_layers, dropout=None)

    def forward(self, control_input: torch.Tensor, perturbation: torch.Tensor):
        embedding = perturbation
        predicted_perturbed_expression = self.decoder(embedding)
        if self.softplus_output:
            predicted_perturbed_expression = F.softplus(predicted_perturbed_expression)
        return predicted_perturbed_expression






        




