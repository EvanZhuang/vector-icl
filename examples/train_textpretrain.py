import sys
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
from accelerate import infer_auto_device_map
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download, snapshot_download


import tqdm, argparse
from modeling_projector import ProjectorConfig, LinearProjector, MLPProjector, ModelWrapper
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize


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
    parser.add_argument("--max_train_steps", type=int, default=None, help="The maximum number of training steps")
    parser.add_argument("--projector_type", type=str, default="ortho", help="The type of projector to use")
    return parser.parse_args()


def process_batched(examples, tokenizer, max_length=128, embed_model=None):
    # format = prefix + projection + postfix + real text
    # projection needs to be added later during training, so we just need to get the real text
    batch = {"post_input_ids": [], "labels": [], "loss_idx": [], "texts": []}
    prefix, postfix = None, None
    input_len = 0
    eos_tensor = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)
    for example in examples:
        # split the text into prefix, text and postfix
        # prefix + text token + postfix
        # first. do a truncation if needed
        tokenized_input = tokenizer.encode(example, return_tensors="pt", truncation=True, padding=False, add_special_tokens=False, max_length=max_length)
        example = tokenizer.batch_decode(tokenized_input)[0]

        # now we split it into three parts
        sentences = sent_tokenize(example)
        if len(sentences) >= 2:
            # at least 2 sentences
            rand_idx_start = np.random.randint(0, len(sentences)-1)
            prefix = " "
            main_text = " ".join(sentences[:rand_idx_start])
            postfix = " ".join(sentences[rand_idx_start:])
        else:
            # only one sentence, set prefix = None
            rand_token_idx = np.random.randint(0, tokenized_input.shape[1]-1)
            prefix = " "
            main_text = tokenizer.decode(tokenized_input[0, :rand_token_idx])
            postfix = tokenizer.decode(tokenized_input[0, rand_token_idx:])

        prefix = None
        batch['texts'].append(main_text)
        # there will be a projection added later between these two texts
        post_input_ids = torch.cat([tokenizer.encode(postfix, return_tensors="pt", truncation=False, padding=False, add_special_tokens=False), eos_tensor], dim=1)
        batch['post_input_ids'].append(post_input_ids)

        ## labels
        labels = torch.cat([tokenizer.encode(postfix, return_tensors="pt", truncation=False, padding=False, add_special_tokens=False), eos_tensor], dim=1)
        batch['labels'].append(labels)
        # track input length so far
        input_len = 1 + batch['post_input_ids'][-1].shape[1]
        cur_seq_len = batch['labels'][-1].shape[1]
        batch['loss_idx'].append(list(range(input_len-cur_seq_len-1, input_len-1)))
    return batch


def custom_collate_fn(batch):
    new_batch = {}
    for key in batch[0].keys():
        if key in ["labels", "post_input_ids"]:
            new_batch[key] = [torch.tensor(item[key]) for item in batch]
        else:
            new_batch[key] = [item[key] for item in batch]
    return new_batch


def main():
    args = parse_args()
    wandb.init(project="embedding_projector")
    wandb.config.update(args)
    # Load the model
    attn_implementation = "flash_attention_2"
    model = transformers.AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, device_map="balanced_low_0", torch_dtype=torch.bfloat16, attn_implementation=attn_implementation, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    config = AutoConfig.from_pretrained(args.base_model_name_or_path, trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Setting pad token to eos token")
        tokenizer.pad_token = tokenizer.eos_token

    num_gpus = torch.cuda.device_count()
    linear_device = torch.device('cuda', 0)

    # No instruction needed for retrieval passages
    passage_prefix = ""
    max_length = args.max_length
    # load model with tokenizer
    if args.embed_model_name_or_path == "nvidia/NV-Embed-v1":
        embed_config = AutoConfig.from_pretrained("nvidia/NV-Embed-v1", trust_remote_code=True)
        embed_config.latent_attention_config._attn_implementation_internal = None
        embed_model = AutoModel.from_pretrained(args.embed_model_name_or_path, config=embed_config, trust_remote_code=True, torch_dtype=torch.float16).to(linear_device)
        in_features = 4096
        expansion_ratio = 1
    elif args.embed_model_name_or_path == "sentence-transformers/gtr-t5-base":
        embed_model = SentenceTransformer(args.embed_model_name_or_path, device=linear_device)
        embed_model.max_seq_length = args.max_length
        in_features = 768
        expansion_ratio = 1
    elif args.embed_model_name_or_path == "dunzhang/stella_en_1.5B_v5":
        embed_model = SentenceTransformer(args.embed_model_name_or_path, trust_remote_code=True, device=linear_device)
        embed_model.max_seq_length = args.max_length
        in_features = 1024
        expansion_ratio = 1
    elif args.embed_model_name_or_path == "Salesforce/SFR-Embedding-2_R":
        embed_model = SentenceTransformer(args.embed_model_name_or_path, device=linear_device)
        embed_model.max_seq_length = args.max_length
        in_features = 4096
        expansion_ratio = 1
    

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

    
    key = 'text'
    if args.dataset_name == "wikitext":
        raw_dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        raw_dataset['train'] = raw_dataset['train'].filter(lambda x: len(x[key]) > 10)
        raw_dataset['validation'] = raw_dataset['validation'].filter(lambda x: len(x[key]) > 10)
    else:
        raw_dataset = datasets.load_dataset(args.dataset_name)
        # MAKE SURE the datasets are in the right format
        print(f"Using custum dataset {args.dataset_name}, make sure the dataset is in the right format with 'text' key as the main text")


    # rename the columns if needed
    if key != 'text':
        raw_dataset['train'] = raw_dataset['train'].rename_column(key, 'text')
        raw_dataset['train'] = raw_dataset['train'].shuffle(seed=42)
        key = 'text'

    train_dataset = raw_dataset['train'].map(process_batched, fn_kwargs={"tokenizer": tokenizer}, batched=True, batch_size=args.batch_size, num_proc=64, input_columns=['text'], remove_columns=raw_dataset['train'].column_names)
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=0, pin_memory=False, drop_last=True)
    model_saving_key = "{}_vicl_clm_{}_{}_{}".format(args.projector_type, args.dataset_name, args.embed_model_name_or_path.split('/')[-1], args.base_model_name_or_path.split('/')[-1])

    if args.max_train_steps is None:
        max_train_steps = len(dataloader)
    else:
        max_train_steps = min(args.max_train_steps, len(dataloader))

    optimizer = torch.optim.AdamW(linear_projector.parameters(), lr=args.lr, weight_decay=0.0001)
    learning_rate_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*max_train_steps), num_training_steps=max_train_steps)

    n_repeat = 1
    pbar = tqdm.tqdm(total=max_train_steps)
    #CELoss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    CELoss = nn.CrossEntropyLoss()
    loss_lst = []
    for ctr, batch in enumerate(dataloader):
        optimizer.zero_grad()
        # get the embeddings
        model_input = []
        reference_ids = []
        loss_idx = []

        with torch.no_grad():
            if args.embed_model_name_or_path == "nvidia/NV-Embed-v1":
                last_states = embed_model.encode(batch['texts'], instruction=passage_prefix, max_length=max_length).unsqueeze(1).to(model.device).type(model.dtype).detach()
            elif args.embed_model_name_or_path == "sentence-transformers/gtr-t5-base":
                last_states = embed_model.encode(batch['texts'], device=embed_model.device, convert_to_tensor=True).unsqueeze(1).to(model.device).type(model.dtype).detach()
            elif args.embed_model_name_or_path == "dunzhang/stella_en_1.5B_v5":
                last_states = embed_model.encode(batch['texts'], device=embed_model.device, convert_to_tensor=True).unsqueeze(1).to(model.device).type(model.dtype).detach()
            elif args.embed_model_name_or_path == "Salesforce/SFR-Embedding-2_R":
                last_states = embed_model.encode(batch['texts'], device=embed_model.device, convert_to_tensor=True).unsqueeze(1).to(model.device).type(model.dtype).detach()

        input_len = 0
        for data_ctr in range(len(batch['labels'])):
            # projection
            transformed_seq = linear_projector.forward(last_states[data_ctr,:].type(linear_projector.dtype).to(linear_device)).type(model.dtype).to(model.device)
            model_input.append(transformed_seq.expand(1, n_repeat, -1))
            # postfix + real text
            model_input.append(model.get_input_embeddings()(batch['post_input_ids'][data_ctr]))
            batch['loss_idx'][data_ctr] = [item + input_len for item in batch['loss_idx'][data_ctr]]

            input_len = batch['loss_idx'][data_ctr][-1] + 2
            assert torch.cat(model_input, dim=1).shape[1] == input_len, "Length MisCalculated"

        model_input = torch.cat(model_input, dim=1)
        loss_idx = batch['loss_idx']
        loss_idx = [item for sublist in loss_idx for item in sublist]
        reference_ids = torch.cat(batch['labels'], dim=1).to(model.device)

        if ctr == 0:
            print(model_input.shape)
        output = model(inputs_embeds=model_input)
        loss = CELoss(output.logits[:,loss_idx,:].contiguous().view(-1, output.logits.shape[-1]), reference_ids.contiguous().view(-1))
        
        loss.backward()
        optimizer.step()
        learning_rate_scheduler.step()
        loss_lst.append(loss.item())
        if ctr % 20 == 0:
            print("Loss:", np.mean(loss_lst))
            wandb.log({"loss": np.mean(loss_lst), "lr": learning_rate_scheduler.get_last_lr()[0], "step": ctr})
            loss_lst = []
        if (ctr+1) % 2000 == 0:
            linear_projector.push_to_hub(model_saving_key)
        pbar.update(1)
        if ctr >= max_train_steps:
            break
    linear_projector.push_to_hub(model_saving_key)


if __name__ == "__main__":
    main()