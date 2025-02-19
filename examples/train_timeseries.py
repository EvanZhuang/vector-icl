import sys
from slerp import slerp

# load a LLAMA3 Model First
import numpy as np
import transformers
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel
import datasets
import evaluate
import wandb

from scripts.prompts import get_prompts, get_verbalizer, get_preprompts, get_instructions
import tqdm, argparse
from modeling_projector import ProjectorConfig, LinearProjector, MLPProjector

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from chronos import ChronosPipeline
from chronos.chronos import left_pad_and_stack_1D
from typing import Union, List

from aeon.classification.deep_learning import LITETimeClassifier
from aeon.classification.shapelet_based import SASTClassifier
from aeon.datasets import load_classification
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from time import time
import tqdm

import datasets, transformers
from transformers import AutoModelForSequenceClassification, AutoConfig
from torch.utils.data import DataLoader
from run_timeseries_train import ChronosTransform

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description="Train a linear projector")
    parser.add_argument("--base_model_name_or_path", type=str, default="mistralai/Mistral-7B-v0.1", help="The model name or path to the model")
    parser.add_argument("--embed_model_name_or_path", type=str, default="amazon/chronos-t5-tiny", help="The model name or path to the Embedding model")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="The dataset to use")
    parser.add_argument("--output_dir", type=str, default="../ckpts", help="The output directory to save the model")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size")
    parser.add_argument("--max_length", type=int, default=512, help="The maximum length of the input")
    parser.add_argument("--lr", type=float, default=0.01, help="The learning rate")
    parser.add_argument("--hub_model_id", type=str, default=None, help="The HuggingFace ID to load from")
    parser.add_argument("--icl_pairs", type=int, default=0, help="The number of ICL pairs to use")
    parser.add_argument("--projector_type", type=str, default="linear", help="The type of projector to use")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="The number of training epochs")
    parser.add_argument("--task_type", choices=["timeseries_prediction"], default="classification", help="The type of task to use")
    return parser.parse_args()


def custom_collate_fn(batch):
    new_batch = {}
    for key in batch[0].keys():
        if key in ["labels", "pre_input_ids", "post_input_ids"]:
            new_batch[key] = [torch.tensor(item[key]) for item in batch]
        else:
            new_batch[key] = [item[key] for item in batch]
    return new_batch


def main():
    args = parse_args()
    # Load the model
    attn_implementation = "flash_attention_2"
    model = transformers.AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, device_map="balanced_low_0", torch_dtype=torch.bfloat16, attn_implementation=attn_implementation)
    embedding_matrix = model.get_input_embeddings().weight.detach()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    num_gpus = torch.cuda.device_count()
    linear_device = torch.device('cuda', 0)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    chronos_transform = ChronosTransform.from_pretrained(
        args.embed_model_name_or_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    config = transformers.AutoConfig.from_pretrained(args.embed_model_name_or_path)


    # Initialize the projector
    if "tiny" in args.embed_model_name_or_path:
        linear_config = ProjectorConfig(in_features=256, out_features=4096)
    elif "small" in args.embed_model_name_or_path:
        linear_config = ProjectorConfig(in_features=512, out_features=4096)
    elif "base" in args.embed_model_name_or_path:
        linear_config = ProjectorConfig(in_features=768, out_features=4096)
    elif "large" in args.embed_model_name_or_path:
        linear_config = ProjectorConfig(in_features=1024, out_features=4096)

    if args.projector_type == "linear":
        Projector = LinearProjector
    elif args.projector_type == "nonlinear":
        Projector = MLPProjector

    if args.hub_model_id is not None:
        linear_projector = Projector.from_pretrained(args.hub_model_id, config=linear_config, dtype=model.dtype).to(linear_device)
    else:
        #linear_projector = OrthoProjector(config=linear_config, dtype=model.dtype).to(model.device)
        linear_projector = Projector(config=linear_config, dtype=model.dtype).to(linear_device)
        # linear_projector.load_state_dict(torch.load('../ckpts/linear_projector_parallel-sentences_NV-Embed-v1_Meta-Llama-3-8B-Instruct_inter.pth'))
    
    # Load dataset
    if len(args.dataset_name.split()) == 1:
        # only one dataset
        raw_dataset = datasets.load_dataset(args.dataset_name)
    else:
        # a set of datasets
        raw_dataset = None
        for dataset_name in args.dataset_name.split(" "):
            # need to have the same scheme for all datasets!
            if raw_dataset is None:
                raw_dataset = datasets.load_dataset(dataset_name)
            else:
                raw_dataset['train'] = datasets.concatenate_datasets([raw_dataset['train'], datasets.load_dataset(dataset_name)['train']])
                raw_dataset['validation'] = datasets.concatenate_datasets([raw_dataset['validation'], datasets.load_dataset(dataset_name)['validation']])
        # shuffle the dataset
        raw_dataset = raw_dataset.shuffle(seed=42)

    main_text = "("
    closing_text = ")'s class (positive, negative) is"
    breaking_text = "\n"

    prefix = tokenizer.encode(main_text, truncation=False, padding=False, add_special_tokens=False, return_tensors="pt").to(model.device)
    postfix = tokenizer.encode(closing_text, truncation=False, padding=False, add_special_tokens=False, return_tensors="pt").to(model.device)
    seperator = tokenizer.encode(breaking_text, truncation=False, padding=False, add_special_tokens=False, return_tensors="pt").to(model.device)

    if len(args.dataset_name.split()) == 1:
        dataset_shorthand = args.dataset_name.split("/")[-1]
    else:
        dataset_shorthand = args.task_type

    label_dict = {0: " negative.", 1: " positive."}
    dataloader = torch.utils.data.DataLoader(raw_dataset['train'], batch_size=args.batch_size, shuffle=True, collate_fn=transformers.default_data_collator, num_workers=8, pin_memory=False, drop_last=True)
    max_step = len(dataloader)*args.num_train_epochs
    model_saving_key = "{}_ts_v2_{}_{}_{}".format(args.projector_type, dataset_shorthand, args.embed_model_name_or_path.split('/')[-1], args.base_model_name_or_path.split('/')[-1])

    # start training
    optimizer = torch.optim.AdamW(linear_projector.parameters(), lr=args.lr, weight_decay=0.1)
    learning_rate_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*len(dataloader)*args.num_train_epochs), num_training_steps=len(dataloader)*args.num_train_epochs)

    n_repeat = 1
    pbar = tqdm.tqdm(total=len(dataloader)*args.num_train_epochs)
    CELoss = nn.CrossEntropyLoss()
    for epoch in range(args.num_train_epochs):
        loss_lst = []
        for ctr, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # get the embeddings
            input_len = 0
            model_input = []
            reference_ids = []
            loss_idx = []            
            # do a max batch limit
            max_batch = 32
            last_states_lst = []
            for data_ctr in range(0, args.batch_size, max_batch):
                Xtrain_reshaped = batch['features'][data_ctr:data_ctr+max_batch]
                Xtrain_transf = chronos_transform.transform(Xtrain_reshaped, max_batch)
                last_states_lst.append(Xtrain_transf)
            last_states = torch.cat(last_states_lst, dim=0)
            last_states = last_states.unsqueeze(1)
            transformed_seq = linear_projector.forward(last_states.type(linear_projector.dtype).to(linear_device)).type(model.dtype).to(model.device)
            for data_ctr in range(last_states.shape[0]):
                # prefix
                model_input.append(model.get_input_embeddings()(prefix))
                model_input.append(transformed_seq[data_ctr,:].view(1,1,-1))
                model_input.append(model.get_input_embeddings()(postfix))
                # postfix + real text
                label_text = label_dict[int(batch['labels'][data_ctr].detach().cpu().item())]
                label_ids = tokenizer.encode(label_text, return_tensors="pt", truncation=False, padding=False, add_special_tokens=False).to(model.device)
                model_input.append(model.get_input_embeddings()(label_ids))
                model_input.append(model.get_input_embeddings()(seperator))

                reference_ids.append(label_ids)
                input_len = input_len + prefix.shape[1] + last_states.shape[1] + postfix.shape[1]
                loss_idx.extend(list(range(input_len-1, input_len+label_ids.shape[1]-1)))
                input_len = input_len + label_ids.shape[1] + seperator.shape[1]

                assert torch.cat(model_input, dim=1).shape[1] == input_len, "Length MisCalculated"

            model_input = torch.cat(model_input, dim=1)

            reference_ids = torch.cat(reference_ids, dim=1)
            output = model(inputs_embeds=model_input)
            
            loss = CELoss(output.logits[:,loss_idx,:].contiguous().view(-1, output.logits.shape[-1]), reference_ids.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            learning_rate_scheduler.step()
            loss_lst.append(loss.item())
            pbar.update(1)

    torch.save(linear_projector.state_dict(), "../ckpts/{}.pth".format(model_saving_key))
    linear_projector.push_to_hub(model_saving_key)


if __name__ == "__main__":
    main()