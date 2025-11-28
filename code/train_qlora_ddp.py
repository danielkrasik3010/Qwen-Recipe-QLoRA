"""
Fine-tune a model on recipe generation dataset using QLoRA (Quantized Low-Rank Adaptation).
Uses recipe-specific preprocessing with assistant-only masking.
Fully integrated with shared utilities and config.yaml.
"""

import os
import wandb
import torch
from dotenv import load_dotenv
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    TrainingArguments,
    Trainer,
)
from utils.config_utils import load_config
from utils.data_utils import load_and_prepare_dataset, build_messages_for_sample
from utils.model_utils import setup_model_and_tokenizer
from paths import OUTPUTS_DIR


# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class PaddingCollator:
    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch):
        # Convert lists to tensors
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in batch]
        attn_masks = [
            torch.tensor(f["attention_mask"], dtype=torch.long) for f in batch
        ]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in batch]

        # Pad to the max length in this batch
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=self.label_pad_token_id
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": labels,
        }


def preprocess_samples(examples, tokenizer, task_instruction, max_length, cfg):
    """
    Tokenize recipe samples and apply assistant-only masking for causal LM.
    
    Args:
        examples: Batch dictionary with 'title', 'ingredients', 'directions', 'NER' fields
        tokenizer: Tokenizer with chat template support
        task_instruction: Task instruction (kept for compatibility)
        max_length: Maximum sequence length
        cfg: Configuration dictionary (required for model config and field_map)
    """
    input_ids_list, labels_list, attn_masks = [], [], []

    # Process each sample in the batch (EXACT from notebook)
    for title, ingredients, directions, ner in zip(
        examples.get("title", []),
        examples.get("ingredients", []),
        examples.get("directions", []),
        examples.get("NER", [])
    ):
        sample = {
            "title": title,
            "ingredients": ingredients,
            "directions": directions,
            "NER": ner
        }

        # Build messages using our build_messages_for_sample() with cfg
        # (Instead of notebook's build_recipe_messages() with model_name)
        msgs_full = build_messages_for_sample(
            sample, task_instruction, include_assistant=True, cfg=cfg
        )
        msgs_prompt = build_messages_for_sample(
            sample, task_instruction, include_assistant=False, cfg=cfg
        )

        # Apply chat template (EXACT from notebook)
        text_full = tokenizer.apply_chat_template(
            msgs_full,
            tokenize=False,
            add_generation_prompt=False
        )
        text_prompt = tokenizer.apply_chat_template(
            msgs_prompt,
            tokenize=False,
            add_generation_prompt=True
        )

        # Get prompt length in characters (EXACT from notebook)
        prompt_len = len(text_prompt)

        # Tokenize with offset mapping (EXACT from notebook)
        tokens = tokenizer(
            text_full,
            max_length=max_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        # Find where assistant response starts (EXACT from notebook)
        start_idx = len(tokens["input_ids"])
        for i, (start_char, _) in enumerate(tokens["offset_mapping"]):
            if start_char >= prompt_len:
                start_idx = i
                break

        # Create labels: mask prompt tokens (-100), keep assistant tokens (EXACT from notebook)
        labels = [-100] * start_idx + tokens["input_ids"][start_idx:]

        # Ensure labels match input_ids length (EXACT from notebook)
        if len(labels) > len(tokens["input_ids"]):
            labels = labels[:len(tokens["input_ids"])]

        input_ids_list.append(tokens["input_ids"])
        labels_list.append(labels)
        attn_masks.append(tokens["attention_mask"])

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attn_masks,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(cfg, model, tokenizer, train_data, val_data, accelerator=None, num_gpus=1):
    """
    Tokenize datasets, configure Trainer, and run LoRA fine-tuning.
    
    Args:
        accelerator: Accelerator instance for DDP-safe operations (optional)
        num_gpus: Number of GPUs used for training (for Hugging Face model naming)
    """
    task_instruction = cfg["task_instruction"]

    print("\n📚 Tokenizing datasets...")
    tokenized_train = train_data.map(
        lambda e: preprocess_samples(
            e, tokenizer, task_instruction, cfg["sequence_len"], cfg  # ADD cfg
        ),
        batched=True,
        remove_columns=train_data.column_names,
    )

    tokenized_val = val_data.map(
        lambda e: preprocess_samples(
            e, tokenizer, task_instruction, cfg["sequence_len"], cfg  # ADD cfg
        ),
        batched=True,
        remove_columns=val_data.column_names,
    )

    collator = PaddingCollator(tokenizer=tokenizer)

    # Use output_dir from config (already set in main() for DDP)
    output_dir = cfg.get("output_dir", os.path.join(OUTPUTS_DIR, "lora_recipe"))
    # Directory creation is DDP-safe (already created in main() on main process)
    # Only create if it doesn't exist (safety check)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["num_epochs"],
        max_steps=cfg.get("max_steps", 500),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_steps=cfg.get("warmup_steps", 100),
        bf16=cfg.get("bf16", True),
        optim=cfg.get("optim", "paged_adamw_8bit"),
        eval_strategy="steps",
        eval_steps=cfg.get("save_steps", 100),  # Evaluate every N steps (same as save_steps)
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 100),  # Save checkpoint every N steps
        logging_steps=cfg.get("logging_steps", 25),
        save_total_limit=cfg.get("save_total_limit", 2),
        report_to="wandb",
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
    )

    print("\n🎯 Starting LoRA fine-tuning...")
    trainer.train()
    print("\n✅ Training complete!")

    # DDP-safe model saving: only save on main process
    is_main_process = accelerator.is_main_process if accelerator else True
    if is_main_process:
        save_dir = os.path.join(output_dir, "lora_adapters")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"💾 Saved LoRA adapters to {save_dir}")
        
        # Optional: Push to Hugging Face Hub (only on main process)
        hf_username = os.getenv("HF_USERNAME")
        hub_model_name = cfg.get("hub_model_name", None)
        
        # Add DDP suffix to model name if using multiple GPUs
        if hub_model_name and num_gpus > 1:
            # Append DDP info: e.g., "Qwen2.5-1.5B-QLoRA-Recipe" -> "Qwen2.5-1.5B-QLoRA-Recipe-DDP-4GPU"
            hub_model_name = f"{hub_model_name.strip()}-DDP-{num_gpus}GPU"
            print(f"\n📝 Using DDP-specific model name: {hub_model_name}")
        elif hub_model_name and num_gpus == 1:
            # Single GPU: append baseline suffix
            hub_model_name = f"{hub_model_name.strip()}-1GPU"
            print(f"\n📝 Using single GPU model name: {hub_model_name}")
        
        if hf_username and hub_model_name:
            push_to_hub(model, tokenizer, hub_model_name, hf_username)
        elif hf_username:
            # Default model name if not specified
            default_name = f"Qwen2.5-1.5B-QLoRA-Recipe-DDP-{num_gpus}GPU" if num_gpus > 1 else "Qwen2.5-1.5B-QLoRA-Recipe-1GPU"
            push_to_hub(model, tokenizer, default_name, hf_username)
        else:
            print("\n💡 To push to Hugging Face Hub, set HF_USERNAME in .env file")
    else:
        print("\n⏭️  Skipping model save (not main process)")


def push_to_hub(model, tokenizer, model_name, hf_username):
    """
    Push LoRA adapters and merged model to Hugging Face Hub.
    Similar to Colab version but uses environment variables.
    
    Args:
        model: The trained PEFT model (with LoRA adapters)
        tokenizer: The tokenizer
        model_name: Model name (e.g., "Qwen2.5-1.5B-QLoRA-Recipe0")
        hf_username: Your Hugging Face username
    """
    model_id = f"{hf_username}/{model_name}"
    
    try:
        print(f"\n📤 Pushing to Hugging Face Hub: {model_id}")
        
        # Push LoRA adapters
        print("  → Pushing LoRA adapters...")
        model.push_to_hub(f"{model_id}-adapters", private=False)
        
        # Merge and push full model
        print("  → Merging adapters and pushing full model...")
        merged_model = model.merge_and_unload()
        merged_model.push_to_hub(model_id, private=False)
        
        # Push tokenizer
        print("  → Pushing tokenizer...")
        tokenizer.push_to_hub(model_id)
        
        print(f"\n✅ Successfully pushed to: https://huggingface.co/{model_id}")
        print(f"   Adapters: https://huggingface.co/{model_id}-adapters")
        
    except Exception as e:
        print(f"\n❌ Error pushing to Hugging Face: {e}")
        print("   Make sure you're logged in with: huggingface-cli login")
        print("   Or set HF_TOKEN in your .env file")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = load_config()

    # Initialize Accelerator for DDP
    accelerator = Accelerator()
    num_gpus = accelerator.num_processes
    print(f"Using {num_gpus} GPUs")
    
    # Get model name for folder structure
    model_name = cfg["base_model"].split("/")[-1].lower()

    # Create output directory: outputs/ddp_{n}gpu/{model_name}/
    if num_gpus == 1:
        config_folder = "baseline_1gpu"
    else:
        config_folder = f"ddp_{num_gpus}gpu"

    run_output_dir = os.path.join(OUTPUTS_DIR, config_folder, model_name)
    run_name = f"{config_folder}-{model_name}"

    print(f"\n🔧 Training mode: DDP with {num_gpus} GPU(s)")
    print(f"📁 Output directory: {run_output_dir}")
    
    # Create output directory (only on main process for DDP)
    if accelerator.is_main_process:
        os.makedirs(run_output_dir, exist_ok=True)
    # Wait for main process to create directory
    accelerator.wait_for_everyone()
    
    # Update config with DDP-specific output directory
    cfg["output_dir"] = run_output_dir
    
    # Load dataset
    train_data, val_data, _ = load_and_prepare_dataset(cfg)
    # Reuse unified model setup (quantization + LoRA)
    model, tokenizer = setup_model_and_tokenizer(
        cfg, use_4bit=True, use_lora=True, padding_side="right", device_map=accelerator.local_process_index,
    )

    if accelerator.is_main_process:
        # Initialize W&B with config values
        wandb.init(
            project=cfg.get("wandb_project", "qwen_recipe"),
            name=run_name,  # Use DDP-specific run name
            config={
                "model": cfg["base_model"],  # Qwen/Qwen2.5-1.5B-Instruct
                "learning_rate": cfg.get("learning_rate", 2e-4),
                "epochs": cfg.get("num_epochs", 1),
                "lora_r": cfg.get("lora_r", 16),
                "lora_alpha": cfg.get("lora_alpha", 32),
                "num_gpus": num_gpus,
                "training_mode": "DDP",
            },
    )

    # Pass accelerator and num_gpus to train_model for DDP-safe operations
    train_model(cfg, model, tokenizer, train_data, val_data, accelerator=accelerator, num_gpus=num_gpus)
    # Finish W&B run on main process
    if accelerator.is_main_process:
        wandb.finish()

    # Wait for all processes to complete
    accelerator.wait_for_everyone()
    accelerator.free_memory()

    # Properly destroy the process group before exit
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
