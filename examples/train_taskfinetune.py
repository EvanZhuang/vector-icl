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
from accelerate import infer_auto_device_map
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download, snapshot_download

from scripts.prompts import get_prompts, get_verbalizer, get_preprompts, get_instructions
import tqdm, argparse
from modeling_projector import ProjectorConfig, LinearProjector, OrthoProjector, MLPProjector, random_orthogonal_matrix
from inversion_eval import batch_encode
from sentence_transformers import SentenceTransformer


torch.multiprocessing.set_sharing_strategy('file_system')


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
    parser.add_argument("--projector_type", type=str, default="linear", help="The type of projector to use")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="The number of training epochs")
    parser.add_argument("--task_type", choices=["sentiment_analysis", "summarization", "molecule", "reconstruction"], default="classification", help="The type of task to use")
    parser.add_argument("--max_bs", type=int, default=8, help="The maximum batch size for the embedding model")
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
    attn_implementation = "flash_attention_2"
    config = AutoConfig.from_pretrained(args.base_model_name_or_path, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation=attn_implementation)
    model.gradient_checkpointing_enable()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model_name_or_path)

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
    linear_config = ProjectorConfig(in_features=in_features, out_features=out_features)

    if args.projector_type == "linear":
        Projector = LinearProjector
    elif args.projector_type == "nonlinear":
        Projector = MLPProjector


    if args.hub_model_id is not None:
        linear_projector = Projector.from_pretrained(args.hub_model_id, config=linear_config, dtype=model.dtype).to(linear_device)
    else:
        linear_projector = Projector(config=linear_config, dtype=model.dtype).to(linear_device)
    
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


    key = "text"
    # Get the verbalizer and prompts
    if len(args.dataset_name.split()) == 1 and args.task_type == "sentiment_analysis":
        verbalizer = get_verbalizer(args.dataset_name)
        few_shot_verbalizer = {}
        for k in verbalizer.keys():
            few_shot_verbalizer[verbalizer[k]] = k
    else:
        few_shot_verbalizer = None


    if args.task_type == "sentiment_analysis":
        main_text = "("
        closing_text = ")'s sentiment is"
    elif args.task_type == "summarization":
        main_text = "("
        closing_text = ")'s summarization is: "
    elif args.task_type == "molecule":
        main_text = "("
        closing_text = ")'s molecule caption is: "
    elif args.task_type == "reconstruction":
        main_text = "Translate the text in brackets: ("
        closing_text = "), Translation: "

    if len(args.dataset_name.split()) == 1:
        dataset_shorthand = args.dataset_name.split("/")[-1]
    else:
        dataset_shorthand = args.task_type
    raw_dataset['train'] = raw_dataset['train'].shuffle(seed=42)
    train_dataset = raw_dataset['train'].map(process_batched, fn_kwargs={"tokenizer": tokenizer, "prefix": main_text, "postfix": closing_text, "few_shot_verbalizer": few_shot_verbalizer}, batched=True, batch_size=args.batch_size, num_proc=48)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=8, pin_memory=False, drop_last=True)
    model_saving_key = "{}_vicl_task_{}_{}_{}".format(args.projector_type, dataset_shorthand, args.embed_model_name_or_path.split('/')[-1], args.base_model_name_or_path.split('/')[-1])

    # start training
    optimizer = torch.optim.AdamW(linear_projector.parameters(), lr=args.lr, weight_decay=0.01)
    learning_rate_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*len(dataloader)*args.num_train_epochs), num_training_steps=len(dataloader)*args.num_train_epochs)

    n_repeat = 1
    max_patience = 500
    patience = 0
    min_loss = 1000000
    max_bs = args.max_bs
    pbar = tqdm.tqdm(total=len(dataloader)*args.num_train_epochs)
    CELoss = nn.CrossEntropyLoss()
    for epoch in range(args.num_train_epochs):
        loss_lst = []
        for ctr, batch in enumerate(dataloader):
            optimizer.zero_grad()
            # get the embeddings
            model_input = []
            reference_ids = []
            loss_idx = []

            main_text_embeds = []
            with torch.no_grad():
                for enc_batch in range(len(batch['texts'])//max_bs + 1):
                    start_idx = enc_batch * max_bs
                    end_idx = min((enc_batch + 1) * max_bs, len(batch['texts']))
                    if start_idx == end_idx:
                        break
                    with torch.no_grad():
                        if args.embed_model_name_or_path == "nvidia/NV-Embed-v1":
                            encode_states = embed_model.encode(batch['texts'][start_idx:end_idx], instruction="", max_length=max_length).unsqueeze(1).to(model.device).type(model.dtype).detach()
                        elif args.embed_model_name_or_path == "sentence-transformers/gtr-t5-base":
                            encode_states = embed_model.encode(batch['texts'][start_idx:end_idx], device=embed_model.device, convert_to_tensor=True).unsqueeze(1).to(model.device).type(model.dtype).detach()
                        elif args.embed_model_name_or_path == "dunzhang/stella_en_1.5B_v5":
                            encode_states = embed_model.encode(batch['texts'][start_idx:end_idx], device=embed_model.device, convert_to_tensor=True).unsqueeze(1).to(model.device).type(model.dtype).detach()
                        elif args.embed_model_name_or_path == "Salesforce/SFR-Embedding-2_R":
                            encode_states = embed_model.encode(batch['texts'][start_idx:end_idx], device=embed_model.device, convert_to_tensor=True).unsqueeze(1).to(model.device).type(model.dtype).detach()
                    main_text_embeds.append(encode_states)
            last_states = torch.cat(main_text_embeds, dim=0)

            input_len = 0
            for data_ctr in range(len(batch['labels'])):
                # prefix
                model_input.append(model.get_input_embeddings()(batch['pre_input_ids'][data_ctr].to(model.device)))
                # projection
                transformed_seq = linear_projector.forward(last_states[data_ctr,:].type(linear_projector.dtype).to(linear_device)).type(model.dtype).to(model.device)
                model_input.append(transformed_seq.expand(1, n_repeat, -1))
                # postfix + real text
                model_input.append(model.get_input_embeddings()(batch['post_input_ids'][data_ctr].to(model.device)))
                batch['loss_idx'][data_ctr] = [item + input_len for item in batch['loss_idx'][data_ctr]]

                input_len = batch['loss_idx'][data_ctr][-1] + 2
                assert torch.cat(model_input, dim=1).shape[1] == input_len, "Length MisCalculated"

            model_input = torch.cat(model_input, dim=1)
            loss_idx = batch['loss_idx']
            loss_idx = [item for sublist in loss_idx for item in sublist]
            reference_ids = torch.cat(batch['labels'], dim=1).to(model.device)
            
            output = model(inputs_embeds=model_input)
            
            loss = CELoss(output.logits[:,loss_idx,:].contiguous().view(-1, output.logits.shape[-1]), reference_ids.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            learning_rate_scheduler.step()
            loss_lst.append(loss.item())

            patience += 1
            if loss_lst[-1] < min_loss:
                min_loss = loss_lst[-1]
                patience = 0

            if patience > max_patience:
                print("Early Stopping at epoch: {}, step: {}".format(epoch, ctr))
                break

            # loss = linear_projector.optimize(last_states.detach(), word_embedding.detach(), torch.tensor(tokenized_word['input_ids']).to(model.device), model, input_sentence_representation.detach(), closing_representation.detach())
            if ctr % 20 == 0:
                wandb.log({"loss": np.mean(loss_lst), "lr": learning_rate_scheduler.get_last_lr()[0], "step": ctr})

            if (ctr+1) % 2000 == 0 or (ctr+1) == len(dataloader):
                torch.save(linear_projector.state_dict(), "../ckpts/{}_inter.pth".format(model_saving_key))
                linear_projector.push_to_hub(model_saving_key, private=True, commit_message="Intermediate Checkpoint, epoch: {}, step: {}".format(epoch, ctr))

            pbar.update(1)

    torch.save(linear_projector.state_dict(), "../ckpts/{}.pth".format(model_saving_key))
    linear_projector.push_to_hub(model_saving_key, private=True)


if __name__ == "__main__":
    main()