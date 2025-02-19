import sys
from slerp import slerp

# load a LLAMA3 Model First
import numpy as np
import transformers
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoConfig
import datasets
import evaluate
import wandb
import logging
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate import infer_auto_device_map
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download, snapshot_download

import tqdm, argparse, random
from modeling_projector import ProjectorConfig, LinearProjector, MLPProjector

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a linear projector")
    parser.add_argument("--base_model_name_or_path", type=str, default="mistralai/Mistral-7B-v0.1", help="The model name or path to the model")
    parser.add_argument("--embed_model_name_or_path", type=str, default="nvidia/NV-Embed-v1", help="The model name or path to the Embedding model")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="The dataset to use")
    parser.add_argument("--output_dir", type=str, default="../ckpts", help="The output directory to save the model")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size")
    parser.add_argument("--max_length", type=int, default=512, help="The maximum length of the input")
    parser.add_argument("--lr", type=float, default=0.01, help="The learning rate")
    parser.add_argument("--hub_model_id", type=str, default=None, help="The HuggingFace ID to load from")
    parser.add_argument("--dataset_subsample", action="store_true", help="Whether to subsample the dataset")
    parser.add_argument("--epochs", type=int, default=10, help="The number of epochs to train")
    parser.add_argument("--projector_type", type=str, default="ortho", help="The type of projector to use")
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator_log_kwargs = {}


    accelerator = Accelerator(gradient_accumulation_steps=1, **accelerator_log_kwargs)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Load the model
    attn_implementation = "flash_attention_2"
    config = AutoConfig.from_pretrained(args.base_model_name_or_path, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation=attn_implementation)
    model.gradient_checkpointing_enable()



    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # No instruction needed for retrieval passages
    passage_prefix = ""
    max_length = args.max_length
    in_features = 200
    expansion_ratio = 1

    # linear_device is the last GPU
    num_gpus = torch.cuda.device_count()
    #linear_device = torch.device('cuda', num_gpus-1)
    linear_device = model.device


    # Initialize the projector
    out_features = model.get_input_embeddings().weight.shape[1]
    linear_config = ProjectorConfig(in_features=in_features, out_features=out_features, expansion_ratio=expansion_ratio)
    if args.projector_type == "linear":
        Projector = LinearProjector
    elif args.projector_type == "nonlinear":
        Projector = MLPProjector

    if args.hub_model_id is not None:
        linear_projector = Projector.from_pretrained(args.hub_model_id, config=linear_config, dtype=model.dtype).to(linear_device)
    else:
        linear_projector = Projector(config=linear_config, dtype=model.dtype).to(linear_device)


    model_saving_key = f"{args.projector_type}_projector_{args.dataset_name.split('/')[-1]}_{args.base_model_name_or_path.split('/')[-1]}_class"
    
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
                raw_dataset['validation'] = datasets.concatenate_datasets([raw_dataset['validation'], datasets.load_dataset(dataset_name)['test']])
        # shuffle the dataset
        raw_dataset = raw_dataset.shuffle(seed=42)

    # find 10 instructions
    instructions_set = set()
    for i, instruction in enumerate(raw_dataset['train']['instruction']):
        instructions_set.add(instruction)
    instructions_set = sorted(list(instructions_set)) #[0:10] #+ ["What is the English translation of the input?"]
    #instructions_set = ["What is the English translation of the input?"]
    print(instructions_set)
    # filter our the non-reconstrcution data
    idx = [i for i, (x, y) in enumerate(zip(raw_dataset['train']['instruction'], raw_dataset['train']['text'])) if x in instructions_set and len(y) > 0]
    raw_dataset['train'] = raw_dataset['train'].select(idx)
    
    dataloader = torch.utils.data.DataLoader(raw_dataset['train'], batch_size=args.batch_size, shuffle=True)
    dataloader = accelerator.prepare(dataloader)
    
    # start training
    optimizer = torch.optim.AdamW(linear_projector.parameters(), lr=args.lr, weight_decay=0.001)
    learning_rate_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*len(dataloader)*args.epochs), num_training_steps=len(dataloader)*args.epochs)
    #learning_rate_scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=50)

    n_repeat = 1
    if accelerator.is_main_process:
        pbar = tqdm.tqdm(total=len(dataloader)*args.epochs)
    CELoss = nn.CrossEntropyLoss(ignore_index=2)
    loss_lst = []

    min_loss = 1000000
    max_patience = 100
    english_question = "What is the English translation of the input?"
    for epoch in range(args.epochs):
        batches = list(dataloader)

        # Shuffle the batches
        random.shuffle(batches)
        for ctr, batch in enumerate(batches):
            optimizer.zero_grad()
            # get the embeddings
            
            loss_idx = []
            model_input = []
            reference_ids = []
            input_len = 0
            brain_representation = torch.vstack(batch['representation']).transpose(0, 1)
            for data_ctr in range(len(batch['text'])):
                question, answer = batch['instruction'][data_ctr], batch['text'][data_ctr]
                instruction = question + " Input: "
                instruction_representation = model.get_input_embeddings()(tokenizer(instruction, return_tensors="pt", add_special_tokens=False, padding=False, truncation=False)['input_ids'].to(model.device)).detach()
                reference = tokenizer(answer, return_tensors="pt", add_special_tokens=False, padding=False, truncation=False)['input_ids']
                # prefix
                model_input.append(instruction_representation)
                # projection
                transformed_seq = linear_projector.forward(brain_representation[data_ctr,].type(linear_projector.dtype).to(linear_device)).type(model.dtype).to(model.device)
                model_input.append(transformed_seq.view(1, 1, transformed_seq.shape[-1]))
                # postfix
                postfix = ", Response: the answer to the question is "
                if question == english_question:
                    postfix = ', Response: the input in English is "'
                line_breaker = '" \n' #+ tokenizer.eos_token
                line_breaker_ids = tokenizer(line_breaker, return_tensors="pt", add_special_tokens=False, padding=False, truncation=False)['input_ids']
                postfix = model.get_input_embeddings()(tokenizer(postfix, return_tensors="pt", add_special_tokens=False, padding=False, truncation=False)['input_ids'].to(model.device)).detach()
                model_input.append(postfix)
                input_len = sum([x.shape[1] for x in model_input])

                if reference.shape[1] <= 0:
                    # empty reference, replace with a space tok
                    reference = torch.tensor([[tokenizer.pad_token_id]])
                model_input.append(model.get_input_embeddings()(reference.to(model.device)).detach())
                reference_ids.append(reference.to(model.device))

                loss_idx.extend([item + input_len - 1 for item in range(model_input[-1].shape[1])])

                input_len_breaker = sum([x.shape[1] for x in model_input])
                model_input.append(model.get_input_embeddings()(line_breaker_ids.to(model.device)).detach())
                loss_idx.extend([item + input_len_breaker - 1 for item in range(model_input[-1].shape[1])])
                reference_ids.append(line_breaker_ids.to(model.device))

                assert torch.cat(model_input, dim=1).shape[1] == input_len + model_input[-1].shape[1] + model_input[-2].shape[1], "Length MisCalculated"

            model_input = torch.cat(model_input, dim=1)
            reference_ids = torch.cat(reference_ids, dim=1)

            output = model(inputs_embeds=model_input)
            loss = CELoss(output.logits[:,loss_idx,:].contiguous().view(-1, output.logits.shape[-1]), reference_ids.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            learning_rate_scheduler.step()
            loss_lst.append(loss.detach().item())
        
            if accelerator.is_main_process:
                pbar.update(1)
    torch.save(linear_projector.state_dict(), "../ckpts/{}.pth".format(model_saving_key))
    linear_projector.push_to_hub(model_saving_key)


if __name__ == "__main__":
    main()