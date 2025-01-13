import logging
import time
import psutil
from dataclasses import dataclass, field
import os
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import setup_chat_format
from peft import LoraConfig
from trl import SFTTrainer
import mlflow
from accelerate import Accelerator

# Anthropic/Vicuna like template without the need for special tokens
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)


@dataclass
class ScriptArguments:
    dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the dataset"},
    )
    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )


def b2mb(x):
    return int(x / 2**20)


class TorchTracemalloc:
    def __enter__(self):
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()
        self.cpu_begin = self.process.memory_info().rss
        return self

    def __exit__(self, *exc):
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)
        self.cpu_end = self.process.memory_info().rss
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)


def training_function(script_args, training_args):
    accelerator = Accelerator()

    if accelerator.is_main_process:
        mlflow.set_tracking_uri("")  # add your mlflow endpoint
        experiment_name = "Mi300x_llama3_70B_scrolls_govt_report_experimemt_FSDP_1"
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        mlflow.log_params(
            {
                "model_id": script_args.model_id,
                "dataset": "tau/scrolls",
                "subset": "gov_report",
                "max_seq_length": script_args.max_seq_length,
            }
        )

    train_dataset = load_dataset("tau/scrolls", "gov_report", split="train")
    test_dataset = load_dataset("tau/scrolls", "gov_report", split="test")

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE

    def template_dataset(examples):
        return {"text": examples["input"]}

    train_dataset = train_dataset.map(template_dataset, remove_columns=["input"])
    test_dataset = test_dataset.map(template_dataset, remove_columns=["input"])

    with training_args.main_process_first(
        desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(train_dataset)), 2):
            print(train_dataset[index]["text"])

    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=quant_storage_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        eval_dataset=test_dataset,
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
    )
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    train_start_time = time.time()

    for epoch in range(int(training_args.num_train_epochs)):
        with TorchTracemalloc() as tracemalloc:
            train_output = trainer.train()

            train_loss = train_output.training_loss
            train_runtime = train_output.metrics["train_runtime"]
            train_samples_per_second = train_output.metrics["train_samples_per_second"]
            train_steps_per_second = train_output.metrics["train_steps_per_second"]

            eval_output = trainer.evaluate()
            eval_loss = eval_output["eval_loss"]
            eval_perplexity = eval_output["eval_perplexity"]

            if accelerator.is_main_process:
                mlflow.log_metrics(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_runtime": train_runtime,
                        "train_samples_per_second": train_samples_per_second,
                        "train_steps_per_second": train_steps_per_second,
                        "eval_loss": eval_loss,
                        "eval_perplexity": eval_perplexity,
                        "gpu_memory_used": tracemalloc.used,
                        "gpu_peak_memory": tracemalloc.peaked,
                        "cpu_memory_used": tracemalloc.cpu_used,
                    },
                    step=epoch + 1,
                )

    total_train_runtime = time.time() - train_start_time

    if accelerator.is_main_process:
        mlflow.log_metrics(
            {
                "total_train_runtime": total_train_runtime,
                "final_train_loss": train_loss,
                "final_eval_loss": eval_loss,
                "final_eval_perplexity": eval_perplexity,
            }
        )
        mlflow.end_run()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    set_seed(training_args.seed)

    training_function(script_args, training_args)
