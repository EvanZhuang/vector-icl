import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import LogitsProcessor, LogitsProcessorList

import evaluate
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
import numpy as np

from scripts.prompts import get_prompts, get_verbalizer, get_preprompts, get_instructions
from embedding_merge import few_shot_embedding_icl, few_shot_embedding_icl_generation, few_shot_embedding_icl_graph
from modeling_projector import LinearProjector, ProjectorConfig, MLPProjector

from transformers import DataCollatorWithPadding
from torch.cuda.amp import autocast
from string import punctuation

from transformers.models.deprecated.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def none_type(string):
    if string == "None":
        return None
    return string

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=none_type,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--n_prompt",
        type=str,
        default="max",
        help="Number of prompts to use for the ensemble.",
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=0,
        help="Number of few shot example to use for the ensemble.",
    )
    parser.add_argument(
        "--pad_style",
        type=str,
        default="mean_pad",
        choices=["mean_pad", "interpolation_linear", "interpolation_exact", "interpolation_pad"],
        help="Number of prompts to use for the ensemble.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Sequence Length for the model.",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="right",
        choices=["right", "left"],
        help="Pre or Post Prompt.",
    )
    parser.add_argument(
        "--projector_model_id",
        type=str,
        default=None,
        help="The HuggingFace ID to load from.",
    )
    parser.add_argument("--task_type", choices=["sentiment_analysis", "summarization", "molecule", "translation", "graph"], default="classification", help="The type of task to use")
    parser.add_argument("--max_gen_length", type=int, default=512, help="The maximum length of the generated text")
    parser.add_argument("--dataset_subsample", action="store_true", help="Whether to subsample the dataset")


    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args

class TokenFilterLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        mask[:, self.allowed_token_ids] = 0  # Allow only the tokens in the list
        return scores + mask

def none_type(string):
    if string == "None":
        return None
    return string

def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_prompt_merge", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Load dataset
        raw_datasets = datasets.load_dataset(args.dataset_name)
        raw_datasets = raw_datasets.map(preprocess_item, batched=False, num_proc=48)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    device_map = "balanced_low_0"
    if accelerator.num_processes > 1:
        device_map = "cuda"

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation = "flash_attention_2",
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)

    if args.dataset_subsample:
        random_10percent = np.random.choice(len(raw_datasets['validation']), int(len(raw_datasets['validation']) * 0.2))
        raw_datasets['validation'] = raw_datasets['validation'].select(random_10percent)

    # load embedding model
    embedding_model_name_or_path = "yzhuang/graph-classification"
    model_device_id = model.device.index
    config = transformers.AutoConfig.from_pretrained(embedding_model_name_or_path)
    graph_model = transformers.GraphormerForGraphClassification.from_pretrained(embedding_model_name_or_path, config=config, torch_dtype=torch.bfloat16).to(model.device)
    graph_model = graph_model.encoder
    in_features = 768
    expansion_ratio = 1

    # Initialize the projector
    out_features = model.get_input_embeddings().weight.shape[1]
    linear_config = ProjectorConfig(in_features=in_features, out_features=out_features, expansion_ratio=expansion_ratio)
    if "linear" in args.projector_model_id and "nonlinear" not in args.projector_model_id:
        Projector = LinearProjector
    elif "nonlinear" in args.projector_model_id:
        Projector = MLPProjector

    if args.projector_model_id is not None:
        linear_projector = Projector.from_pretrained(args.projector_model_id, config=linear_config, dtype=model.dtype).to(model.device).type(model.dtype)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    completed_steps = 0
    starting_epoch = 0

    graph_collator = GraphormerDataCollator()
    train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=graph_collator, num_workers=64, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=graph_collator, num_workers=8, shuffle=True)
    eval_dataloader = accelerator.prepare(eval_dataloader)

    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    #metric_auc = evaluate.load("roc_auc")
    look_in_the_middle_metric = evaluate.load("f1")
    model_preds_per_prompt = []
    output_labels = []
    if accelerator.is_main_process:
        pbar = tqdm(total=len(eval_dataloader))

    prefix = "("
    postfix = ")'s class (positive or negative) is: "
    breaking_text = "\n"

    allowed_tokens = ["positive.", "negative.", " "]
    allowed_token_ids = []
    for token in allowed_tokens:
        token_ids = tokenizer(token, add_special_tokens=False).input_ids
        allowed_token_ids.extend(token_ids)  # Append all token ids to the list

    # Remove duplicates in case some tokens overlap or result in the same token ID
    allowed_token_ids = list(set(allowed_token_ids))
    logits_processor = LogitsProcessorList([TokenFilterLogitsProcessor(allowed_token_ids)])

    model_module = model

    #check the number of data points per rank
    print("Number of data points per rank: ", len(eval_dataloader), accelerator.num_processes)
    for data_ctr, data in enumerate(eval_dataloader): #.select(random.sample(range(len(eval_dataset)), 100)):
        # we do a prompt merge per data 
        # if data['labels'].item() == 0:
        #     continue
        data = {k: v.to(model.device) for k, v in data.items()}
        model_preds = []
        input_embeddings, attn_masks, input_locs = [], [], []
        with torch.no_grad() and autocast(dtype=torch.bfloat16):
            prompt_embed, attn_masks, input_lens_lst = few_shot_embedding_icl_graph(model_module, tokenizer, few_shot_dataset=train_dataloader, n_shot=args.n_shot, n_prompt=args.n_prompt, max_seq_length=args.max_seq_length,  embed_model=graph_model, projector=linear_projector, prefix=prefix, postfix=postfix)
        
            inputs_embeds = graph_model(**data)['last_hidden_state'][:, 0:1, :]
            inputs_embeds = linear_projector(inputs_embeds)
            pre_indicator_embeds = model.get_input_embeddings()(tokenizer(prefix, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False)['input_ids'].to(model.device))
            post_indicator_embeds = model.get_input_embeddings()(tokenizer(postfix, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False)['input_ids'].to(model.device))
            inputs_embeds = torch.cat([prompt_embed, pre_indicator_embeds, inputs_embeds, post_indicator_embeds], dim=1)

            outputs = model.generate(inputs_embeds=inputs_embeds, max_new_tokens=3, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=2e-1, num_beams=1, return_dict_in_generate=True, logits_processor=logits_processor)

            model_pred = tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True)

            # Trim the generated shots
            model_pred = model_pred.split(postfix)[0].split(prefix)[0].strip()
            if "positive" in model_pred.lower():
                model_pred = torch.tensor([1])
            elif "negative" in model_pred.lower():
                model_pred = torch.tensor([0])
            else:
                print("Model Prediction: ", model_pred)
                continue
        metric_acc.add(prediction=model_pred, reference=data["labels"])
        metric_f1.add(prediction=model_pred, reference=data["labels"])
        look_in_the_middle_metric.add(prediction=model_pred, reference=data["labels"])

        if (data_ctr+1) % 50 == 0 and accelerator.is_main_process:
            # look in the middle
            middle_results = look_in_the_middle_metric.compute()
            print("Middle Results: ", middle_results)
            
        if accelerator.is_main_process:
            pbar.update(1)
    accelerator.wait_for_everyone()
    eval_results = metric_acc.compute()
    eval_results_f1 = metric_f1.compute()
    eval_results.update(eval_results_f1)
    
    if accelerator.is_main_process:
        print(eval_results)
        with open(os.path.join(args.output_dir, "{}_eval_results.json".format(args.dataset_name.split("/")[-1])), "w") as f:
            json.dump(eval_results, f)
        
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()