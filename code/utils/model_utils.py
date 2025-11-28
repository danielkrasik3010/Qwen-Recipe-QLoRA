"""
Utilities for model loading, preparation, and checkpoint management.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ---------------------------------------------------------------------------
# Model & Tokenizer Setup
# ---------------------------------------------------------------------------


def setup_model_and_tokenizer(
    cfg, use_4bit: bool = None, use_lora: bool = None, padding_side: str = "right", device_map=None
):
    """
    Load model, tokenizer, and apply quantization + LoRA config if specified.

    Args:
        cfg (dict): Configuration dictionary containing:
            - base_model
            - quantization parameters
            - lora parameters (optional)
            - bf16 or fp16 precision
        use_4bit (bool, optional): Override whether to load in 4-bit mode.
        use_lora (bool, optional): Override whether to apply LoRA adapters.
        padding_side (str): Tokenizer padding side ("left" or "right").
        device_map (str, int, dict, optional): Device placement strategy.
            - "auto": Automatic device placement (default for single GPU)
            - int: Specific GPU device index (for DDP, use accelerator.local_process_index)
            - None: Uses "auto" as default

    Returns:
        tuple: (model, tokenizer)
    """
    model_name = cfg["base_model"]
    print(f"\nLoading model: {model_name}")

    # ------------------------------
    # Tokenizer setup
    # ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side

    # Determine quantization + LoRA usage
    load_in_4bit = use_4bit if use_4bit is not None else cfg.get("load_in_4bit", False)
    apply_lora = use_lora if use_lora is not None else ("lora_r" in cfg)

    # ------------------------------
    # Quantization setup (optional)
    # ------------------------------
    quant_cfg = None
    if load_in_4bit:
        print("⚙️  Enabling 4-bit quantization...")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=getattr(
                torch, cfg.get("bnb_4bit_compute_dtype", "bfloat16")
            ),
        )
    else:
        print("⚙️  Loading model in full precision (no quantization).")

    # ------------------------------
    # Model loading
    # ------------------------------
    # Handle device_map for DDP (when device_map is an integer) vs single GPU (when "auto")
    if device_map is None:
        device_map = "auto"  # Default: automatic device placement
    
    # For DDP: if device_map is an integer, convert to dict format for bitsandbytes compatibility
    # For 4-bit quantization with DDP, we use dict format: {"": device_index}
    if isinstance(device_map, int):
        # DDP mode: place model on specific GPU
        # Use dict format for bitsandbytes compatibility: {"": device_index}
        device_map_dict = {"": device_map}
        print(f"🔧 DDP mode: Loading model on GPU {device_map}")
        device_map = device_map_dict
    elif device_map == "auto":
        print("🔧 Auto device placement mode")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        device_map=device_map,
        dtype=(
            torch.bfloat16
            if cfg.get("bf16", True) and torch.cuda.is_available()
            else torch.float32
        ),
    )

    # ------------------------------
    # LoRA setup (optional)
    # ------------------------------
    if apply_lora:
        print("🔧 Applying LoRA configuration...")
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=cfg.get("lora_r", 8),
            lora_alpha=cfg.get("lora_alpha", 16),
            target_modules=cfg.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        print("🔹 Skipping LoRA setup — using base model only.")

    return model, tokenizer


def get_last_checkpoint_path(checkpoints_dir):
    """
    Return the path to the most recent checkpoint in a directory.

    Args:
        checkpoints_dir (str): Directory containing checkpoints.

    Returns:
        str: Full path to the last checkpoint.
    """
    checkpoints = [
        int(f.replace("checkpoint-", ""))
        for f in os.listdir(checkpoints_dir)
        if f.startswith("checkpoint")
    ]
    if not checkpoints:
        return None
    last_ckpt = max(checkpoints)
    return os.path.join(checkpoints_dir, f"checkpoint-{last_ckpt}")


def count_trainable_params(model):
    """Return total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_gb(model):
    """Calculate approximate model size in GB."""
    total_bytes = 0
    for param in model.parameters():
        dtype_size = torch.tensor([], dtype=param.dtype).element_size()
        total_bytes += param.numel() * dtype_size
    return total_bytes / (1024**3)
