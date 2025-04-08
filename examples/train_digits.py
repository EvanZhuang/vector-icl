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

import tqdm, argparse
from modeling_projector import LinearProjector, ProjectorConfig, MLPProjector
from sentence_transformers import SentenceTransformer
from embedding_merge import encode_digits_to_one_hot

def parse_args():
    parser = argparse.ArgumentParser(description="Train a linear projector")
    parser.add_argument("--base_model_name_or_path", type=str, default="mistralai/Mistral-7B-v0.1", help="The model name or path to the model")
    parser.add_argument("--embed_model_name_or_path", type=str, default="nvidia/NV-Embed-v1", help="The model name or path to the Embedding model")
    parser.add_argument("--dataset_name", type=str, default="yzhuang/number_clm_simple", help="The dataset to use")
    parser.add_argument("--output_dir", type=str, default="../ckpts", help="The output directory to save the model")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size")
    parser.add_argument("--max_length", type=int, default=512, help="The maximum length of the input")
    parser.add_argument("--lr", type=float, default=0.01, help="The learning rate")
    parser.add_argument("--hub_model_id", type=str, default=None, help="The HuggingFace ID to load from")
    parser.add_argument("--dataset_subsample", action="store_true", help="Whether to subsample the dataset")
    parser.add_argument("--epochs", type=int, default=10, help="The number of epochs to train")
    parser.add_argument("--projector_type", type=str, default="linear", help="The type of projector to use")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the model
    attn_implementation = "flash_attention_2"
    model = transformers.AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # No instruction needed for retrieval passages
    passage_prefix = ""
    max_length = args.max_length
    in_features = 100
    expansion_ratio = 1

    # linear_device is the last GPU
    num_gpus = torch.cuda.device_count()
    linear_device = torch.device('cuda', num_gpus-1)


    if args.projector_type == "linear":
        Projector = LinearProjector
    else:
        raise ValueError("Projector type not supported")

    # Initialize the projector
    verson = "v0"
    linear_config = ProjectorConfig(in_features=in_features, out_features=model.get_input_embeddings().weight.shape[-1], expansion_ratio=expansion_ratio)
    if args.hub_model_id is not None:
        linear_projector = Projector.from_pretrained(args.hub_model_id, config=linear_config, dtype=model.dtype).to(linear_device)
        # check if it has the version number in the model id
        if args.hub_model_id.split("_v")[-1].isdigit():
            verson = "v" + str(int(args.hub_model_id.split("_v")[-1]) + 1)
    else:
        linear_projector = Projector(config=linear_config).to(linear_device)

    model_saving_key = f"{args.projector_type}_projector_{args.dataset_name.split('/')[-1]}_{args.base_model_name_or_path.split('/')[-1]}" + "_" + verson

    
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
    
    dataloader = torch.utils.data.DataLoader(raw_dataset['train'], batch_size=args.batch_size, shuffle=True)
    

    # start training
    optimizer = torch.optim.AdamW(linear_projector.parameters(), lr=args.lr, weight_decay=0.001)
    learning_rate_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*len(dataloader)*args.epochs), num_training_steps=len(dataloader)*args.epochs)

    n_repeat = 1
    pbar = tqdm.tqdm(total=len(dataloader)*args.epochs)
    CELoss = nn.CrossEntropyLoss(ignore_index=2)
    loss_lst = []

    min_loss = 1000000
    patience = 0
    max_patience = 100
    one_hot_proj = torch.eye(in_features).to(linear_projector.device)

    for epoch in range(args.epochs):
        loss_lst = []
        for ctr, batch in enumerate(dataloader):
            optimizer.zero_grad()
            # get the embeddings
            
            loss_idx = []
            model_input = []
            reference_ids = []
            input_len = 0
            brain_representation = torch.vstack(batch['representation']).transpose(0, 1)

            rand_A = np.random.random()
            rand_B = np.random.random()
            
            for data_ctr in range(len(batch['text'])):
                question, answer = batch['instruction'][data_ctr], batch['text'][data_ctr]
                var_name = "A"
                var2_name = "B"

                if np.random.rand() > 0.95:
                    instruction = f"{var_name}="
                    postfix_text = f"{var_name} equals to (digits): "
                    instruction_representation = model.get_input_embeddings()(tokenizer(instruction, return_tensors="pt", add_special_tokens=False, padding=False, truncation=False)['input_ids']).detach().to(model.device)
                    reference = tokenizer(answer, return_tensors="pt", add_special_tokens=False, padding=False, truncation=False)['input_ids']
                    # prefix
                    model_input.append(instruction_representation)
                    # projection
                    number = batch['representation'][0][data_ctr]
                    transformed_seq = linear_projector(encode_digits_to_one_hot(digits=number).type(linear_projector.dtype).to(linear_device)).type(model.dtype).to(model.device)
                    model_input.append(transformed_seq.view(1, 1, -1))
                    # postfix
                    line_breaker = '\n' 
                    line_breaker_ids = tokenizer(line_breaker, return_tensors="pt", add_special_tokens=False, padding=False, truncation=False)['input_ids']
                    postfix = model.get_input_embeddings()(tokenizer(postfix_text, return_tensors="pt", add_special_tokens=False, padding=False, truncation=False)['input_ids']).detach().to(model.device)
                    model_input.append(postfix)
                    input_len = sum([x.shape[1] for x in model_input])

                    if reference.shape[1] <= 0:
                        # empty reference, replace with a space tok
                        reference = torch.tensor([[tokenizer.pad_token_id]])
                    model_input.append(model.get_input_embeddings()(reference).detach().to(model.device))
                    reference_ids.append(reference.to(model.device))

                    loss_idx.extend([item + input_len - 1 for item in range(model_input[-1].shape[1])])

                    input_len_breaker = sum([x.shape[1] for x in model_input])
                    model_input.append(model.get_input_embeddings()(line_breaker_ids).detach().to(model.device))
                    loss_idx.extend([item + input_len_breaker - 1 for item in range(model_input[-1].shape[1])])
                    reference_ids.append(line_breaker_ids.to(model.device))
                    assert torch.cat(model_input, dim=1).shape[1] == input_len + model_input[-1].shape[1] + model_input[-2].shape[1], "Length MisCalculated"
                else:
                    rand_data_ctr = np.random.randint(0, len(batch['text']))
                    A_number = batch['representation'][0][data_ctr]
                    B_number = batch['representation'][0][rand_data_ctr]

                    postfix_text = f"F={rand_A:.2f}*{var_name}+{rand_B:.2f}*{var2_name}, F({var_name},{var2_name}) equals to (digits): "
                    reference = tokenizer(str(int(rand_A*A_number+rand_B*B_number)), return_tensors="pt", add_special_tokens=False, padding=False, truncation=False)['input_ids']

                    A = linear_projector(encode_digits_to_one_hot(digits=A_number).type(linear_projector.dtype).to(linear_device)).type(model.dtype).to(model.device)
                    B = linear_projector(encode_digits_to_one_hot(digits=B_number).type(linear_projector.dtype).to(linear_device)).type(model.dtype).to(model.device)

                    Aeq = model.get_input_embeddings()(tokenizer(f"{var_name}=", return_tensors="pt", padding=False, truncation=False, add_special_tokens=False)['input_ids'].to(model.device))
                    Beq = model.get_input_embeddings()(tokenizer(f"{var2_name}=", return_tensors="pt", padding=False, truncation=False, add_special_tokens=False)['input_ids'].to(model.device))
                    # postfix
                    model_input.append(Aeq)
                    model_input.append(A.view(1, 1, -1))
                    model_input.append(Beq)
                    model_input.append(B.view(1, 1, -1))
                    # postfix
                    line_breaker = '\n' #+ tokenizer.eos_token
                    line_breaker_ids = tokenizer(line_breaker, return_tensors="pt", add_special_tokens=False, padding=False, truncation=False)['input_ids']
                    postfix = model.get_input_embeddings()(tokenizer(postfix_text, return_tensors="pt", add_special_tokens=False, padding=False, truncation=False)['input_ids']).detach().to(model.device)
                    model_input.append(postfix)
                    input_len = sum([x.shape[1] for x in model_input])

                    if reference.shape[1] <= 0:
                        # empty reference, replace with a space tok
                        reference = torch.tensor([[tokenizer.pad_token_id]])
                    model_input.append(model.get_input_embeddings()(reference).detach().to(model.device))
                    reference_ids.append(reference.to(model.device))

                    loss_idx.extend([item + input_len - 1 for item in range(model_input[-1].shape[1])])

                    input_len_breaker = sum([x.shape[1] for x in model_input])
                    model_input.append(model.get_input_embeddings()(line_breaker_ids).detach().to(model.device))
                    loss_idx.extend([item + input_len_breaker - 1 for item in range(model_input[-1].shape[1])])
                    reference_ids.append(line_breaker_ids.to(model.device))
            
            model_input = torch.cat(model_input, dim=1)
            reference_ids = torch.cat(reference_ids, dim=1)

            output = model(inputs_embeds=model_input)
            loss = CELoss(output.logits[:,loss_idx,:].contiguous().view(-1, output.logits.shape[-1]), reference_ids.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            learning_rate_scheduler.step()
            loss_lst.append(loss.detach().item())

            patience += 1
            if loss_lst[-1] < min_loss:
                min_loss = loss_lst[-1]
                patience = 0

            if (ctr+1) % 10000 == 0:
                torch.save(linear_projector.state_dict(), "../ckpts/{}_inter.pth".format(model_saving_key))
            pbar.update(1)
        print("epoch", epoch, "Loss:", np.mean(loss_lst))

    torch.save(linear_projector.state_dict(), "../ckpts/{}.pth".format(model_saving_key))


if __name__ == "__main__":
    main()