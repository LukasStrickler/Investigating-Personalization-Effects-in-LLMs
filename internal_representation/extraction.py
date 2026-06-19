"""
Step 2 — Hidden State Extraction (Forward Pass).

Loads the frozen LLM and extracts residual-stream hidden states for each
conversation in the dataset.  Hidden states are captured at every layer and
at the selected token position (last user token by default).
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
import transformers.tokenization_utils_base as _tub
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from config import ModelConfig, ProbeConfig


# Monkey-patch: transformers assumes extra_special_tokens is a dict, but some models
# (e.g. Gemma 4) store it as a list in their tokenizer_config.json. This fixes the
# AttributeError: 'list' object has no attribute 'keys'.
_orig_set_special = _tub.PreTrainedTokenizerBase._set_model_specific_special_tokens


def _patched_set_special_tokens(self, special_tokens):
    if isinstance(special_tokens, list):
        pass  # Token values, not attribute names — skip (e.g. Gemma 4 multimodal tokens)
    else:
        _orig_set_special(self, special_tokens)


_tub.PreTrainedTokenizerBase._set_model_specific_special_tokens = _patched_set_special_tokens


def load_model_and_tokenizer(
    config: ModelConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a frozen HuggingFace causal LM and its tokenizer."""

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)

    print(f"[Model] Loading {config.model_name} (dtype={config.torch_dtype}) ...")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        token=config.hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "flash_attention_2" if config.use_flash_attention else "eager"

    # Load model — try AutoModelForCausalLM first, fall back to AutoModelForImageTextToText
    # (required for multimodal models like Gemma 4).
    for model_cls in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            model = model_cls.from_pretrained(
                config.model_name,
                device_map=config.device_map,
                torch_dtype=torch_dtype,
                token=config.hf_token,
                trust_remote_code=True,
                attn_implementation=attn_impl,
            )
            break
        except Exception as e:
            print(f"[Model] {model_cls.__name__} failed ({e}), trying next ...")
    else:
        raise RuntimeError(f"Could not load model {config.model_name} with any known class.")
    model.eval()
    # Freeze all parameters (safety measure — we never call .backward())
    for param in model.parameters():
        param.requires_grad_(False)

    text_cfg = getattr(model.config, "text_config", model.config)
    num_layers = text_cfg.num_hidden_layers
    hidden_dim = text_cfg.hidden_size
    print(f"[Model] Loaded — {num_layers} layers, hidden_dim={hidden_dim}")
    return model, tokenizer


def _find_last_user_token_idx(
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    attention_mask: Optional[torch.Tensor] = None,
) -> int:
    """
    Find the index of the last token that belongs to the **user's** final turn.

    Heuristic: locate the last occurrence of "User:" in the decoded tokens and
    return the index of the last non-padding token in that segment.  Falls back
    to the very last non-pad token if it can't find a "User:" marker.
    """
    ids = input_ids.squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    # Build the decoded text to find the character offset of the last "User:"
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    last_user_pos = decoded.rfind("User:")
    if last_user_pos == -1:
        last_user_pos = decoded.rfind("user")  # Fallback for chat templates

    if last_user_pos != -1:
        # Find the token index closest to the last "User:" segment end
        # We want the last token *before* the next "Assistant:" reply.
        next_assistant_pos = decoded.find("Assistant:", last_user_pos)
        if next_assistant_pos == -1:
            # The user turn is the very last segment
            target_char = len(decoded) - 1
        else:
            target_char = next_assistant_pos - 1

        # Binary search to find the token index corresponding to target_char
        low, high = 0, len(ids)
        token_idx = len(ids) - 1
        while low < high:
            mid = (low + high) // 2
            prefix_str = tokenizer.decode(ids[:mid], skip_special_tokens=False)
            if len(prefix_str) > target_char:
                token_idx = mid - 1
                high = mid
            else:
                low = mid + 1
        return min(token_idx, len(ids) - 1)

    # Fallback: last non-padding token
    if attention_mask is not None:
        mask = attention_mask.squeeze().tolist()
        for i in range(len(mask) - 1, -1, -1):
            if mask[i] == 1:
                return i
    return len(ids) - 1


@torch.no_grad()
def extract_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    conversations: List[str],
    model_config: ModelConfig,
    probe_config: ProbeConfig,
    batch_size: int = 1,
) -> Dict[int, np.ndarray]:
    """
    Run the forward pass on every conversation and collect hidden states.

    Parameters
    ----------
    model : frozen CausalLM with output_hidden_states=True
    tokenizer : matching tokenizer
    conversations : list of conversation strings
    model_config : model configuration
    probe_config : probe configuration (token position selection)
    batch_size : kept at 1 to avoid padding artefacts

    Returns
    -------
    hidden_states : dict mapping layer_index -> np.ndarray of shape
                    (n_conversations, hidden_dim)
    """
    text_cfg = getattr(model.config, "text_config", model.config)
    num_layers = text_cfg.num_hidden_layers
    layers_to_extract = (
        probe_config.layers if probe_config.layers else list(range(num_layers + 1))
    )  # +1 because HF returns embedding layer (index 0) + N transformer layers

    # Pre-allocate collection lists
    collected: Dict[int, List[np.ndarray]] = {l: [] for l in layers_to_extract}

    device = next(model.parameters()).device

    for conv in tqdm(conversations, desc="Extracting hidden states"):
        # Tokenize
        inputs = tokenizer(
            conv,
            return_tensors="pt",
            truncation=True,
            max_length=model_config.max_seq_length,
            padding=False,
        ).to(device)

        outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple of (num_layers+1) tensors,
        # each of shape (batch=1, seq_len, hidden_dim)
        all_hidden = outputs.hidden_states

        # Determine the token position to extract
        if probe_config.token_position == "last":
            tok_idx = _find_last_user_token_idx(
                inputs["input_ids"], tokenizer, inputs.get("attention_mask")
            )
        else:
            tok_idx = None  # will mean-pool below

        for layer_idx in layers_to_extract:
            # Convert to float32 early to avoid float16 overflow/inf
            h = all_hidden[layer_idx].float()  # (1, seq_len, hidden_dim)
            if tok_idx is not None:
                vec = h[0, tok_idx, :].cpu().numpy()
            else:
                # Mean pool over non-padding tokens
                mask = inputs["attention_mask"][0].unsqueeze(-1).float()
                vec = (h[0] * mask).sum(dim=0) / mask.sum(dim=0)
                vec = vec.cpu().numpy()
            collected[layer_idx].append(vec)

    hidden_states = {l: np.stack(vecs) for l, vecs in collected.items()}

    # Sanitize: replace any residual inf/NaN with 0
    n_bad = 0
    for l in hidden_states:
        bad_mask = ~np.isfinite(hidden_states[l])
        n_bad += bad_mask.sum()
        hidden_states[l] = np.nan_to_num(hidden_states[l], nan=0.0, posinf=0.0, neginf=0.0)
    if n_bad > 0:
        print(f"  [WARN] Replaced {n_bad} inf/NaN values in hidden states with 0")

    print(
        f"[Extraction] Collected hidden states for {len(layers_to_extract)} layers, "
        f"{len(conversations)} samples, dim={hidden_states[layers_to_extract[0]].shape[1]}"
    )
    return hidden_states


def save_hidden_states(hidden_states: Dict[int, np.ndarray], path: str) -> None:
    """Save extracted hidden states to a compressed .npz file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **{str(k): v for k, v in hidden_states.items()})
    print(f"[Extraction] Saved hidden states to {path}")


def load_hidden_states(path: str) -> Dict[int, np.ndarray]:
    """Load hidden states from a .npz file."""
    data = np.load(path)
    return {int(k): data[k] for k in data.files}
