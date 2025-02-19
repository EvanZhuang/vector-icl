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

from transformers.models.deprecated.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator


torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description="Train a linear projector")
    parser.add_argument("--base_model_name_or_path", type=str, default="mistralai/Mistral-7B-v0.1", help="The model name or path to the model")
    parser.add_argument("--embed_model_name_or_path", type=str, default="clefourrier/graphormer-base-pcqm4mv2", help="The model name or path to the Embedding model")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="The dataset to use")
    parser.add_argument("--output_dir", type=str, default="../ckpts", help="The output directory to save the model")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size")
    parser.add_argument("--max_length", type=int, default=512, help="The maximum length of the input")
    parser.add_argument("--lr", type=float, default=0.01, help="The learning rate")
    parser.add_argument("--hub_model_id", type=str, default=None, help="The HuggingFace ID to load from")
    parser.add_argument("--icl_pairs", type=int, default=0, help="The number of ICL pairs to use")
    parser.add_argument("--projector_type", type=str, default="linear", help="The type of projector to use")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="The number of training epochs")
    parser.add_argument("--task_type", choices=["graph_prediction"], default="classification", help="The type of task to use")
    return parser.parse_args()


def process_batched(examples, tokenizer, prefix, postfix, max_length=128, few_shot_verbalizer=None):
    # format = prefix + projection + postfix + real text
    # projection needs to be added later during training, so we just need to get the real text
    batch = {'pre_input_ids': [], "post_input_ids": [], "labels": [], "loss_idx": [], "texts": []}
    input_len = 0
    eos_tensor = torch.tensor([[tokenizer.eos_token_id]])
    for example_ctr in range(len(examples['text'])):
        # first, map the label text
        example_text = examples['text'][example_ctr]
        example_label = examples['label'][example_ctr]
        if few_shot_verbalizer is not None:
            label_text = few_shot_verbalizer[int(example_label)]
        else:
            label_text = " " + example_label
        batch['texts'].append(example_text)
        batch['pre_input_ids'].append(tokenizer.encode(prefix, return_tensors="pt", truncation=False, padding=False, add_special_tokens=False))
        # there will be a projection added later between these two texts
        post_input_ids = torch.cat([tokenizer.encode(postfix + label_text, return_tensors="pt", truncation=False, padding=False, add_special_tokens=False), eos_tensor], dim=1)
        batch['post_input_ids'].append(post_input_ids)

        ## labels
        labels = torch.cat([tokenizer.encode(label_text, return_tensors="pt", truncation=False, padding=False, add_special_tokens=False), eos_tensor], dim=1)
        batch['labels'].append(labels)
        # track input length so far
        input_len = batch['pre_input_ids'][-1].shape[1] + 1 + batch['post_input_ids'][-1].shape[1]
        cur_seq_len = batch['labels'][-1].shape[1]
        batch['loss_idx'].append(list(range(input_len-cur_seq_len-1, input_len-1)))
    return batch


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
    wandb.init(project="embedding_projector")
    wandb.config.update(args)
    # Load the model
    #balanced_low_0
    attn_implementation = "flash_attention_2"
    model = transformers.AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, device_map="balanced_low_0", torch_dtype=torch.bfloat16, attn_implementation=attn_implementation)
    embedding_matrix = model.get_input_embeddings().weight.detach()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    num_gpus = torch.cuda.device_count()
    linear_device = torch.device('cuda', 0)

    config = transformers.AutoConfig.from_pretrained(args.embed_model_name_or_path)
    graph_model = transformers.GraphormerForGraphClassification.from_pretrained(args.embed_model_name_or_path, config=config, torch_dtype=torch.bfloat16).to(linear_device)

    # Initialize the projector
    out_features = model.get_input_embeddings().weight.shape[1]
    linear_config = ProjectorConfig(in_features=768, out_features=out_features)

    if args.projector_type == "nonlinear":
        Projector = MLPProjector
    elif args.projector_type == "linear":
        Projector = LinearProjector

    if args.hub_model_id is not None:
        linear_projector = Projector.from_pretrained(args.hub_model_id, config=linear_config, dtype=model.dtype).to(linear_device)
    else:
        linear_projector = Projector(config=linear_config, dtype=model.dtype).to(linear_device)
    
    # Load dataset
    graph_dataset = datasets.load_dataset(args.dataset_name)
    graph_dataset = graph_dataset.map(preprocess_item, batched=False, num_proc=48)

    main_text = "("
    closing_text = ")'s class (positive or negative) is: "
    breaking_text = "\n"

    prefix = tokenizer.encode(main_text, truncation=False, padding=False, add_special_tokens=False, return_tensors="pt").to(model.device)
    postfix = tokenizer.encode(closing_text, truncation=False, padding=False, add_special_tokens=False, return_tensors="pt").to(model.device)
    seperator = tokenizer.encode(breaking_text, truncation=False, padding=False, add_special_tokens=False, return_tensors="pt").to(model.device)

    if len(args.dataset_name.split()) == 1:
        dataset_shorthand = args.dataset_name.split("/")[-1]
    else:
        dataset_shorthand = args.task_type

    

    target = graph_dataset['train']['labels']
    target = np.array(target)
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in [0, 1]])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t[0]] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

    label_dict = {0: "negative.", 1: "positive."}
    graph_collator = GraphormerDataCollator()
    dataloader = torch.utils.data.DataLoader(graph_dataset['train'], batch_size=args.batch_size, shuffle=True, collate_fn=graph_collator, num_workers=0, pin_memory=False, drop_last=True) #sampler=sampler)
    max_step = len(dataloader)*args.num_train_epochs
    real_dataloader = torch.utils.data.DataLoader(graph_dataset['train'], batch_size=1, shuffle=False, collate_fn=graph_collator, num_workers=0, pin_memory=True, drop_last=False, sampler=sampler)
    model_saving_key = "{}_graph_icl_{}_{}_{}".format(args.projector_type, dataset_shorthand, args.embed_model_name_or_path.split('/')[-1], args.base_model_name_or_path.split('/')[-1])

    # start training
    optimizer = torch.optim.AdamW(linear_projector.parameters(), lr=args.lr, weight_decay=0.01)
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
                sub_batch = {k: v[data_ctr:data_ctr+max_batch].to(linear_device) for k, v in batch.items()}
                with torch.no_grad():
                    last_states_lst.append(graph_model.encoder(**sub_batch)["last_hidden_state"][:,0:1,:].detach())
            last_states = torch.cat(last_states_lst, dim=0)
            transformed_seq = linear_projector.forward(last_states.type(linear_projector.dtype).to(linear_device)).type(model.dtype).to(model.device)
            for data_ctr in range(last_states.shape[0]):
                # prefix
                model_input.append(model.get_input_embeddings()(prefix))
                model_input.append(transformed_seq[data_ctr,:].unsqueeze(0))
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