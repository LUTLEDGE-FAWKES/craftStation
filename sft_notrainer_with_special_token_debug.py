#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import math
import os
import copy
import random
from itertools import chain
from functools import partial
from pathlib import Path
from typing import Dict

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
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    DataCollatorForSeq2Seq,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from peft import LoraConfig, TaskType, get_peft_model

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.41.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
DEFAULT_PAD_TOKEN = '[PAD]'

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
        type=str,
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
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--trainable",
        type=str,
        default="q_proj,v_proj",
        help="lora target module",
    )
    parser.add_argument(
        "--modules_to_save",
        type=str,
        default=None,
        help="extra target module",
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
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
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
        "--added_tokens",
        type=str,
        default=None,
        help=(
            "需要加入的的special token"
        ),
    )
    parser.add_argument(
        "--skip_tokens",
        type=str,
        default=None,
        help=(
            "训练时忽略的special token"
        ),
    )

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

    return args


"""
暂不支持从checkpoint恢复训练
"""
def main():
    args = parse_args()
    
    """
    TODO 为模型添加猴子补丁，适配flash-attn或者添加其他功能
    """

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
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()


    """
    初始化模型与分词器
    """
    if args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )
    
    if args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        # logger.info("Training new model from scratch")
        # model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)
        raise ValueError(
            "You are instantiating a new model from scratch, which is not recommended."
        )
    

    """
    添加special token并扩充词表, 尽量确保reshape后emb形状为64倍数
    """
    # 参考自self-rag
    def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
    ):
        """Resize tokenizer and embedding.
        Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
        """
        model_vocab_size = model.get_input_embeddings().num_embeddings
        logger.warning("模型的词表大小: {}".format(model_vocab_size))
        vocab_size_before = len(tokenizer)
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

        # 如果词表大小超过模型的词表大小，需要调整模型的词表大小
        # 当词表大小没有超过模型的词表大小时，不需要调整模型的词表大小（其实也可以调整，不会影响训练，但没有必要）
        if num_new_tokens + vocab_size_before > model_vocab_size:
            logger.warning("词表大小超过模型的词表大小，需要reshape embedding")
            model.resize_token_embeddings(len(tokenizer))
            if num_new_tokens > 0: # 初始化添加的embedding的权重，这里简单的取了已有embedding的均值
                input_embeddings = model.get_input_embeddings().weight.data
                output_embeddings = model.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg
        else:
            logger.warning("词表大小没有超过模型的词表大小，不需要reshape embedding")

        if accelerator.is_main_process:
            logger.warning("调整前的词表大小: {}".format(vocab_size_before))
            vocab_size_after = len(tokenizer)
            logger.warning("调整后的词表大小: {}".format(vocab_size_after))
            logger.warning("添加的special tokens 个数: {}".format(num_new_tokens))
            new_token_ids = tokenizer.convert_tokens_to_ids(special_tokens_dict['additional_special_tokens'])
            for token, token_id in zip(special_tokens_dict['additional_special_tokens'], new_token_ids):
                logger.warning("添加的 special token: '{}', ID: {}".format(token, token_id))


    if args.added_tokens is not None: 
        special_token_dict = {
            # "additional_special_tokens": ["QUERY_GENERATION", "TITLE_GENERATION"]}
            "additional_special_tokens": args.added_tokens.split(',')}
        if tokenizer.pad_token is None:
            special_token_dict["pad_token"] = DEFAULT_PAD_TOKEN
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_token_dict,
            tokenizer=tokenizer,
            model=model,
        )
    else:
        logger.warning('并没有传入任何special token，将根据原始词表进行训练')

    """
    # 初始化LoRA
    # TODO 添加kwarg
    """

    if args.use_lora:
        logger.info("Initializing LORA model...")
        modules_to_save = ["embed_tokens"]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_rank,
            target_modules=args.trainable.split(','),
            modules_to_save=args.modules_to_save.split(','),
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    """
    加载并处理数据集
    """

    data_files = {}
    dataset_args = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = (
        args.train_file.split(".")[-1]
        if args.train_file is not None
        else args.validation_file.split(".")[-1]
    )
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        **dataset_args,
    )
    column_names = list(raw_datasets["train"].features)


    def _tokenize_fn(text: str, tokenizer: transformers.AutoTokenizer, max_seq_length: int):
        """Tokenize a list of strings."""
        input_ids = labels = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                max_length=max_seq_length,
                truncation=True,
        ).input_ids
        input_ids_lens = labels_lens = input_ids.ne(tokenizer.pad_token_id).sum().item()
        # print(input_ids_lens)

        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )
    
    # 目前版本训练数据中直接包含了prompt template，而不是在分词预处理时才加入
    def sft_data_processor(example, tokenizer, max_seq_length, skip_tokens):

        source_text = example['text']
        target_text = example['answer'] + tokenizer.eos_token
        examples_tokenized = _tokenize_fn(source_text + target_text, tokenizer, max_seq_length)
        sources_tokenized = _tokenize_fn(source_text, tokenizer, max_seq_length)

        input_ids = examples_tokenized["input_ids"].flatten()
        source_len = sources_tokenized["input_ids_lens"]
        labels = copy.deepcopy(input_ids)
        labels[ :source_len-1] = -100

        # special token mask
        if skip_tokens is not None:
            for i, input_id_list in enumerate(input_ids):
                for j, orig_token in enumerate(input_id_list):
                    if orig_token in skip_tokens:
                        labels[i][j] = -100

        attention_mask = torch.ones_like(input_ids)

        return {
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
            'attention_mask': attention_mask.flatten()
        }

    encode_function = partial(
        sft_data_processor,
        tokenizer=tokenizer,
        max_seq_length=512,
        skip_tokens=args.skip_tokens.split(',') if args.skip_tokens else None,
    )

    if accelerator.is_main_process:
        logger.info('被忽略的special token：{}'.format(args.skip_tokens))

    with accelerator.main_process_first():
        sft_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in column_names if name not in ["input_ids", "labels", "attention_mask"]],
            desc=f"making sft data with special token",
        )

    train_dataset = sft_datasets["train"]
    eval_dataset = sft_datasets["validation"]

    """
    数据示例，如果不显示，请把log level调到20
    """
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    """
    训练初始化
    """
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)
    
    """
    训练
    """
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running special token training in FAWKES loop no trainer *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0

        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if accelerator.is_main_process and completed_steps %5 == 1:
                    print(f"\nstep {completed_steps} train loss: {loss.detach().float()}")

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()


