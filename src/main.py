#!/usr/bin/env python

import logging
import os
import sys
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import random
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk

import transformers
from transformers import (
    AutoConfig,
    AutoModel, 
    AutoProcessor,
    HfArgumentParser, 
    TrainingArguments, 
    Trainer, 
    set_seed,
    EvalPrediction,
    BitsAndBytesConfig
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from peft import LoraConfig, PeftModel, get_peft_model

# Import our custom classes
from trainer import LateInteractionTrainer
from multimodal_collator import MultimodalCollator
from util import PROCESSOR_MAPPING, MODEL_MAPPING, LOSS_MAPPING, AGGREGATION_MAPPING, compute_metrics

from functools import partial

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: str = field(
        default="colqwen", metadata={"help": "Type of model to use (colgemma, colqwen, colqwenomni)"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the encoder parameters or not."}
    )
    projection_dim: int = field(
        default=128, metadata={"help": "Dimensionality of the projection space."}
    )
    temperature_for_loss: float = field(
        default=0.02, metadata={"help": "Temperature for the loss function."}
    )
    pretrained_peft_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained PEFT model or model identifier from huggingface.co/models"}
    )
    do_peft: bool = field(
        default=False, metadata={"help": "Whether to use PEFT for the model."}
    )
    encoder_model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    # Add quantization parameters
    quantize_4bit: bool = field(
        default=False, metadata={"help": "Whether to quantize the model to 4 bits."}
    )
    quantize_8bit: bool = field(
        default=False, metadata={"help": "Whether to quantize the model to 8 bits."}
    )
    # Add LoRA parameters
    lora_r: int = field(
        default=8, metadata={"help": "Lora attention dimension."}
    )
    lora_alpha: int = field(
        default=16, metadata={"help": "Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "Lora dropout."}
    )
    lora_target_modules: str = field(
        default="(.*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)",
        metadata={"help": "List of module names to apply LoRA to."}
    )
    siglip_loss: bool = field(
        default=False, metadata={"help": "Whether to use siglip loss."}
    )
    init_logit_scale: float = field(
        default=np.log(10), metadata={"help": "Initial value for the logit scale. For siglip, it is usually np.log(10)"}
    )
    init_logit_bias: float = field(
        default=-10, metadata={"help": "Initial value for the logit bias. For siglip, it is usually -10"}
    )
    modality_types: List[str] = field(
        default_factory=lambda: ["video", "ocr", "asr", "description"], 
        metadata={"help": "Modality types to use."}
    )
    query_max_length: int = field(
        default=64, metadata={"help": "Maximum length of the query."}
    )
    ocr_max_length: int = field(
        default=512, metadata={"help": "Maximum length of the ocr embeddings."}
    )
    asr_max_length: int = field(
        default=512, metadata={"help": "Maximum length of the asr embeddings."}
    )
    description_max_length: int = field(
        default=512, metadata={"help": "Maximum length of the description embeddings."}
    )
    text_max_length: int = field(
        default=512, metadata={"help": "Maximum length of the text embeddings."}
    )
    combine_modalities: bool = field(
        default=False, metadata={"help": "Whether to combine modalities."}
    )
    max_combined_length: int = field(
        default=2048, metadata={"help": "Maximum length of the combined embeddings."}
    )
    loss_type: str = field(
        default="contrastive", metadata={"help": "Type of loss to use (contrastive, contrastive_hard_positive, contrastive_hard_negative)"}
    )
    aggregation_method: str = field(
        default="token_max", metadata={"help": "Aggregation method to use (token_max, modality_max)"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The data directory containing input files."}
    )
    frames_column: Optional[str] = field(
        default="images",
        metadata={"help": "The name of the column in the datasets containing the images."},
    )
    query_column: Optional[str] = field(
        default="caption",
        metadata={"help": "The name of the column in the datasets containing the text query."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_frames: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of frames to use for the retrieval."}
    )

    video_only: bool = field(
        default=False, metadata={"help": "Whether to only use video data."}
    )

    ocr_only: bool = field(
        default=False, metadata={"help": "Whether to only use ocr data."}
    )

    asr_only: bool = field(
        default=False, metadata={"help": "Whether to only use asr data."}
    )

    description_only: bool = field(
        default=False, metadata={"help": "Whether to only use description data."}
    )

    frames_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to the directory containing the frames."}
    )
    video_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to the directory containing the videos."}
    )
    preprocess_with_processor: bool = field(
        default=False, metadata={"help": "Whether to preprocess examples with the processor. Note: This can significantly increase dataset size."}
    )
    audio_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to the directory containing the audio files (for omni model)."}
    )

def main():
    # 1. Parse input arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry
    send_example_telemetry("run_multimodal_retriever", model_args, data_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and setting seed
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        
    set_seed(training_args.seed)

    # 4. Initialize processor
    logger.info(f"Initializing processor")

    processor = PROCESSOR_MAPPING[model_args.model_type].from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_fast=True,
    )

    # 5. Load dataset
    logger.info(f"Loading local dataset")
    
    # Load and process a regular dataset
    logger.info(f"Loading dataset from local path: {data_args.data_dir}")
    
    # Prepare the dataset with preprocessing
    dataset = load_from_disk(data_args.data_dir)
    
    logger.info(f"Dataset loaded, train_dataset: {dataset['train'] if 'train' in dataset else None}, eval_dataset: {dataset['validation'] if 'validation' in dataset else None}, test_dataset: {dataset['test'] if 'test' in dataset else None}")
    train_dataset = dataset["train"] if 'train' in dataset else None
    eval_dataset = dataset["validation"] if "validation" in dataset else None
    test_dataset = dataset["test"] if "test" in dataset else None

    if test_dataset is not None and eval_dataset is None:
        eval_dataset = test_dataset
    
    if training_args.do_eval and eval_dataset is None:
        assert train_dataset is not None
        eval_dataset = train_dataset

    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
    
    if data_args.video_only:
        if training_args.do_train:
            train_dataset = train_dataset.filter(lambda x: x["query_type"] == "video") if train_dataset is not None else None
            eval_dataset = eval_dataset.filter(lambda x: x["query_type"] == "video") if eval_dataset is not None else None
        # test_dataset = test_dataset.filter(lambda x: x["query_type"] == "video") if test_dataset is not None else None
        model_args.modality_types = ["video"]
    
    if data_args.ocr_only:
        if training_args.do_train:
            train_dataset = train_dataset.filter(lambda x: x["query_type"] == "ocr") if train_dataset is not None else None
            eval_dataset = eval_dataset.filter(lambda x: x["query_type"] == "ocr") if eval_dataset is not None else None
        # test_dataset = test_dataset.filter(lambda x: x["query_type"] == "ocr") if test_dataset is not None else None
        model_args.modality_types = ["ocr"]
    
    if data_args.asr_only:
        if training_args.do_train:
            train_dataset = train_dataset.filter(lambda x: x["query_type"] == "speech") if train_dataset is not None else None
            eval_dataset = eval_dataset.filter(lambda x: x["query_type"] == "asr") if eval_dataset is not None else None
        # test_dataset = test_dataset.filter(lambda x: x["query_type"] == "asr") if test_dataset is not None else None
        model_args.modality_types = ["asr"]
    
    if data_args.description_only:
        if training_args.do_train:
            train_dataset = train_dataset.filter(lambda x: x["query_type"] == "description") if train_dataset is not None else None
            eval_dataset = eval_dataset.filter(lambda x: x["query_type"] == "description") if eval_dataset is not None else None
        # test_dataset = test_dataset.filter(lambda x: x["query_type"] == "description") if test_dataset is not None else None
        model_args.modality_types = ["description"]
    
    if training_args.do_train and eval_dataset is None:
        # split train dataset into train and eval
        dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    
    # add idx
    if train_dataset is not None:
        train_dataset = train_dataset.add_column("idx", list(range(len(train_dataset))))
    if eval_dataset is not None:
        eval_dataset = eval_dataset.add_column("idx", list(range(len(eval_dataset))))
    if test_dataset is not None:
        test_dataset = test_dataset.add_column("idx", list(range(len(test_dataset))))
    

    if "omni" in model_args.model_type:
        if "msrvtt" in data_args.data_dir:
            model_args.modality_types = ["video", "audio"]
        else:
            model_args.modality_types = ["video", "audio", "ocr", "description"]
    else:
        if "msrvtt" in data_args.data_dir:
            model_args.modality_types = ["video", "asr"]
        else:
            model_args.modality_types = ["video", "asr", "ocr", "description"]

    # 7. Initialize model
    logger.info(f"Initializing model")

    quantization_config = None
    if model_args.quantize_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    elif model_args.quantize_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )

    model = MODEL_MAPPING[model_args.model_type].from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        quantization_config=quantization_config,
    )

    if model_args.do_peft:
        if model_args.pretrained_peft_model_name_or_path is not None:
            print("Loading pretrained PEFT model")
            model.load_adapter(model_args.pretrained_peft_model_name_or_path, is_trainable=True)
        else:
            peft_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                bias="none",
                task_type="FEATURE_EXTRACTION",
                target_modules=model_args.lora_target_modules,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    logging.info(model)

    if "msrvtt" in data_args.data_dir:
        model_args.modality_types = ["video", "asr"]
    else:
        model_args.modality_types = ["video", "ocr", "asr", "description"]
    
    # 8. Initialize trainer based on loss type
    agg = AGGREGATION_MAPPING[model_args.aggregation_method]
    loss_func = LOSS_MAPPING[model_args.loss_type](
        aggregation_method=agg,
        modality_types=model_args.modality_types,
    )
    
    max_lengths = {
        "query": model_args.query_max_length,
        "ocr": model_args.ocr_max_length,
        "asr": model_args.asr_max_length,
        "description": model_args.description_max_length,
        "text": model_args.text_max_length,
        "combined": model_args.max_combined_length,
    }

    collator_kwargs = dict(
        processor=processor,
        max_lengths={
            "query": model_args.query_max_length,
            "ocr": model_args.ocr_max_length,
            "asr": model_args.asr_max_length,
            "description": model_args.description_max_length,
            "text": model_args.text_max_length,
            "combined": model_args.max_combined_length,
        },
        modality_types=model_args.modality_types,
        combine_modalities=model_args.combine_modalities,
        audio_dir=data_args.audio_dir,
        frames_dir=data_args.frames_dir,
        video_dir=data_args.video_dir,
    )
    collator = MultimodalCollator(**collator_kwargs)

    training_args.label_names = ["labels"]
    training_args.remove_unused_columns = False
    trainer = LateInteractionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=partial(
            compute_metrics,
            agg=agg,
            device=model.device,
            batch_size=8,
            modality_types=model_args.modality_types,
            output_dir=training_args.output_dir
        ),
        loss_func=loss_func,
        data_collator=collator,
        modality_types=model_args.modality_types,
        agg=agg
    )
    
    # 9. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()
        model.save_pretrained(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)
        
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 10. Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 11. Write training arguments to disk
    if trainer.is_world_process_zero():
        with open(os.path.join(training_args.output_dir, "training_args.json"), "w") as f:
            import json
            json.dump(training_args.to_dict(), f, indent=2)

if __name__ == "__main__":
    main()