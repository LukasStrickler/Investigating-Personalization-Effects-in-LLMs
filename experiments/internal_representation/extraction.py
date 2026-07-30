"""
Step 2 — Hidden State Extraction (Forward Pass).

Loads the frozen LLM and extracts residual-stream hidden states for each
conversation in the dataset.  Hidden states are captured at every layer and
at the selected token position (last user token by default).
"""

import os

import numpy as np
import torch
import transformers.tokenization_utils_base as _tub
from config import ModelConfig, ProbeConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

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
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a frozen HuggingFace causal LM and its tokenizer."""

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)

    print(f"[Model] Loading {config.model_name} (dtype={config.torch_dtype}) ...")

    # Tokenizer load with a fallback: some repos (notably Mistral/Ministral) ship
    # a mistral-format tokenizer that the fast AutoTokenizer can't parse. Retry
    # with the slow tokenizer, which reads the underlying sentencepiece/tekken
    # model directly.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            token=config.hf_token,
            trust_remote_code=True,
        )
    except Exception as tok_exc:  # noqa: BLE001
        print(f"[Model] Fast tokenizer failed ({tok_exc}); retrying with use_fast=False ...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            token=config.hf_token,
            trust_remote_code=True,
            use_fast=False,
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
    attention_mask: torch.Tensor | None = None,
) -> int:
    """
    Find the index of the last token that belongs to the **user's** final turn.

    Heuristic: locate the last occurrence of "User:" in the decoded tokens and
    return the index of the last non-padding token in that segment.  Falls back
    to the very last non-pad token if it can't find a "User:" marker.
    """
    ids = input_ids.squeeze().tolist()

    # Build the decoded text to find the character offset of the last "User:"
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    last_user_pos = decoded.rfind("User:")
    if last_user_pos == -1:
        last_user_pos = decoded.rfind("user")  # Fallback for chat templates

    if last_user_pos != -1:
        # Find the token index closest to the last "User:" segment end
        # We want the last token *before* the next "Assistant:" reply.
        next_assistant_pos = decoded.find("Assistant:", last_user_pos)
        # If there is no following "Assistant:", the user turn is the very last segment
        target_char = len(decoded) - 1 if next_assistant_pos == -1 else next_assistant_pos - 1

        # Map character offset back to a token index (approximate)
        char_count = 0
        token_idx = 0
        for i, tok in enumerate(ids):
            tok_str = tokenizer.decode([tok])
            char_count += len(tok_str)
            if char_count >= target_char:
                token_idx = i
                break
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
    conversations: list[str],
    model_config: ModelConfig,
    probe_config: ProbeConfig,
    batch_size: int = 1,
) -> dict[int, np.ndarray]:
    """
    Run the forward pass on every conversation and collect hidden states.

    Parameters
    ----------
    model : frozen CausalLM with output_hidden_states=True
    tokenizer : matching tokenizer
    conversations : list of conversation strings
    model_config : model configuration
    probe_config : probe configuration (token position selection)
    batch_size : number of histories per forward pass. Token positions are
                 computed before right-padding, so batching does not change
                 which representation is extracted.

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
    collected: dict[int, list[np.ndarray]] = {layer: [] for layer in layers_to_extract}

    device = next(model.parameters()).device

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    for start in tqdm(range(0, len(conversations), batch_size), desc="Extracting hidden states"):
        batch = conversations[start : start + batch_size]
        token_positions = []
        if probe_config.token_position == "last":
            for conv in batch:
                single = tokenizer(
                    conv,
                    return_tensors="pt",
                    truncation=True,
                    max_length=model_config.max_seq_length,
                    padding=False,
                )
                token_positions.append(
                    _find_last_user_token_idx(
                        single["input_ids"], tokenizer, single.get("attention_mask")
                    )
                )

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=model_config.max_seq_length,
            padding=True,
        ).to(device)

        outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple of (num_layers+1) tensors,
        # each of shape (batch=1, seq_len, hidden_dim)
        all_hidden = outputs.hidden_states

        for layer_idx in layers_to_extract:
            # Convert to float32 early to avoid float16 overflow/inf
            h = all_hidden[layer_idx].float()  # (1, seq_len, hidden_dim)
            if probe_config.token_position == "last":
                vec = (
                    torch.stack([h[row, token_positions[row], :] for row in range(len(batch))])
                    .cpu()
                    .numpy()
                )
            else:
                # Mean pool over non-padding tokens
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                vec = ((h * mask).sum(dim=1) / mask.sum(dim=1)).cpu().numpy()
            collected[layer_idx].extend(vec)

    tokenizer.padding_side = original_padding_side

    hidden_states = {layer: np.stack(vecs) for layer, vecs in collected.items()}

    # Sanitize: replace any residual inf/NaN with 0
    n_bad = 0
    for layer in hidden_states:
        bad_mask = ~np.isfinite(hidden_states[layer])
        n_bad += bad_mask.sum()
        hidden_states[layer] = np.nan_to_num(hidden_states[layer], nan=0.0, posinf=0.0, neginf=0.0)
    if n_bad > 0:
        print(f"  [WARN] Replaced {n_bad} inf/NaN values in hidden states with 0")

    print(
        f"[Extraction] Collected hidden states for {len(layers_to_extract)} layers, "
        f"{len(conversations)} samples, dim={hidden_states[layers_to_extract[0]].shape[1]}"
    )
    return hidden_states


def save_hidden_states(hidden_states: dict[int, np.ndarray], path: str) -> None:
    """Save extracted hidden states to a compressed .npz file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **{str(k): v for k, v in hidden_states.items()})
    print(f"[Extraction] Saved hidden states to {path}")


def load_hidden_states(path: str) -> dict[int, np.ndarray]:
    """Load hidden states from a .npz file."""
    data = np.load(path)
    return {int(k): data[k] for k in data.files}
