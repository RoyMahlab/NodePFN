import torch
from transformers import AutoTokenizer, AutoModel, modeling_utils
from collections import defaultdict
import os
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Fix for missing ALL_PARALLEL_STYLES in modeling_utils
# This is a workaround for compatibility with older versions of transformers
# where ALL_PARALLEL_STYLES might not be defined.
# It ensures that the code runs without errors even if this attribute is missing.
# This is particularly useful for environments where the transformers library
# might not have the latest updates or features.
if (
    not hasattr(modeling_utils, "ALL_PARALLEL_STYLES")
    or modeling_utils.ALL_PARALLEL_STYLES is None
):
    modeling_utils.ALL_PARALLEL_STYLES = [
        "tp",
        "none",
        "colwise",
        "rowwise",
        "colwise_rep",
        "rowwise_rep",
    ]

def find_prompts_file(prompts_file: str, cache_dir: Optional[str] = None) -> str:
    """
    Search for the prompts file in multiple locations.
    
    Searches in the following order:
    1. The exact path provided (if it's an absolute path or exists as-is)
    2. Current working directory
    3. Cache directory (if provided)
    
    Args:
        prompts_file (str): The prompts file name or path
        cache_dir (Optional[str]): Additional directory to search in
        
    Returns:
        str: The full path to the found prompts file
        
    Raises:
        FileNotFoundError: If the prompts file cannot be found in any location
    """
    # List of directories to search
    search_paths = []
    
    # If it's already an absolute path or relative path that exists, try it first
    if os.path.exists(prompts_file):
        return os.path.abspath(prompts_file)
    
    # Add current working directory
    search_paths.append(os.getcwd())
    
    # Add cache directory if provided
    if cache_dir:
        search_paths.append(cache_dir)
    
    # Search in each directory
    for directory in search_paths:
        candidate = os.path.join(directory, prompts_file)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    
    # If not found, raise an error with helpful message
    searched_locations = "\n  - ".join(search_paths)
    raise FileNotFoundError(
        f"Prompts file '{prompts_file}' not found.\n"
        f"Searched in:\n  - {searched_locations}"
    )


def load_hf_llm(model_name) -> tuple:
    """
    Load a Hugging Face language model and its corresponding tokenizer.
    Args:
        model_name (str): The name or path of the Hugging Face model to load.
            This can be a model identifier from the Hugging Face Hub or a local path.
    Returns:
        tuple: A tuple containing:
            - tokenizer (AutoTokenizer): The loaded tokenizer with pad_token set to eos_token
            - model (AutoModel): The loaded Hugging Face model
    Note:
        The function sets the tokenizer's pad_token to the eos_token to ensure
        proper padding behavior during tokenization.
    """

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def embed_text(text: str, tokenizer, model, device) -> torch.Tensor:
    """
    Generate text embeddings using a pre-trained transformer model.
    Args:
        text (str): The input text to be embedded.
        tokenizer: The tokenizer associated with the model, used to convert text to tokens.
        model: The pre-trained transformer model used to generate embeddings.
        device: The device  where the model computations will be performed.
    Returns:
        torch.Tensor: The embeddings from the last hidden state of the model with shape
                      (batch_size, sequence_length, hidden_size). The tensor is detached
                      from the computation graph.
    """

    # Tokenize and prepare for the model
    inputs = tokenizer(text, return_tensors="pt")
    inputs.to(device)

    # Get model output (using no_grad for efficiency)
    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.detach()

    return embeddings


def embed_prompts(
    prompts: Iterable[str], hf_model: str, device: torch.device
) -> torch.Tensor:
    """
    Embed a collection of text prompts using a HuggingFace language model.

    This function efficiently processes multiple prompts by grouping them by token length
    to enable batched processing, which improves performance compared to processing
    prompts individually.

    Args:
        prompts (Iterable[str]): An iterable of text prompts to embed.
        hf_model (str): The HuggingFace model identifier or path to use for embedding.
        device (torch.device): The device to run the model on.

    Returns:
        torch.Tensor: A list of embedded tensors, one for each input prompt.
                     Each tensor contains the last hidden state embeddings for the
                     corresponding prompt.

    Note:
        - The function groups prompts by token length to enable efficient batched processing
        - Memory is freed by explicitly deleting the tokenizer and model after use
        - All output tensors are moved to CPU before returning
    """

    tokenizer, llm = load_hf_llm(hf_model)
    llm.to(device)

    # Tokenize all prompts without padding
    tokenized_prompts = []
    embedded_tensors = []

    for prompt in prompts:
        tokenized_prompts.append(
            tokenizer(prompt, padding=False, return_tensors="pt", truncation=False)
        )

    # Split the tokenized tensors by length
    length_groups = defaultdict(list)
    for item in tokenized_prompts:
        length = item["input_ids"].shape[1]
        length_groups[length].append(item)

    # Embed each group as a batch
    for group in length_groups.values():
        input_ids = torch.cat([item["input_ids"] for item in group], dim=0).to(
            llm.device
        )
        attention_mask = torch.cat(
            [item["attention_mask"] for item in group], dim=0
        ).to(llm.device)

        outputs = llm(input_ids=input_ids, attention_mask=attention_mask)
        embedded_tensors.extend(list(outputs.last_hidden_state.detach().cpu()))

    # # Move model back to CPU for memory efficiency
    # llm.to("cpu")

    del tokenizer, llm  # Free memory
    torch.cuda.empty_cache()
    return embedded_tensors


def add_group_embeddings(embedded_tensors: List[torch.Tensor], scaler) -> List[torch.Tensor]:
    """
    Add scaled group mean embeddings to each tensor in the input list.
    For each tensor in the input list, this function computes the mean across the sequence
    length dimension and adds it back to the original tensor scaled by the given factor.
    This operation can help enhance group-level information in the embeddings.
    Args:
        embedded_tensors (List[torch.Tensor]): List of embedding tensors, each with shape
            (batch_size, sequence_length, embedding_dim) or similar.
        scaler: Scaling factor to apply to the group mean before adding it back to the
            original tensor. Can be any numeric type, fixed or learnable.
    Returns:
        List[torch.Tensor]: List of updated tensors with the same shape as input tensors,
            where each tensor has the scaled group mean added to it.
    """

    updated_tensors = []
    
    for tensor in embedded_tensors:
        group_mean = tensor.mean(dim=1)  # Mean over sequence length
        updated_group = tensor + group_mean * scaler  # Broadcasting addition
        updated_tensors.append(updated_group)

    return updated_tensors


def get_cache_filename(prompts_file: str, model_name: str) -> str:
    """
    Generate a cache filename based on the prompts file and model name.
    
    Args:
        prompts_file (str): Path to the prompts JSON file
        model_name (str): Name of the HuggingFace model (may contain slashes)
        
    Returns:
        str: A sanitized filename for the cached embeddings
    """
    # Get the base name of the prompts file without extension
    prompts_basename = Path(prompts_file).stem
    
    # Sanitize model name by replacing slashes with underscores
    model_basename = model_name.replace('/', '_')
    
    # Create cache filename
    cache_filename = f"{prompts_basename}_{model_basename}_embeddings.pt"
    
    return cache_filename


def should_reembed(prompts_file: str, cache_file: str) -> bool:
    """
    Check if embeddings should be regenerated by comparing file modification times.
    
    Args:
        prompts_file (str): Path to the prompts JSON file
        cache_file (str): Path to the cached embeddings file
        
    Returns:
        bool: True if embeddings should be regenerated, False if cache is fresh
    """
    # If cache doesn't exist, need to embed
    if not os.path.exists(cache_file):
        return True
    
    # If prompts file doesn't exist, something is wrong
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    # Compare modification times
    prompts_mtime = os.path.getmtime(prompts_file)
    cache_mtime = os.path.getmtime(cache_file)
    
    # Need to re-embed if prompts file is newer than cache
    return prompts_mtime > cache_mtime


def load_or_embed_prompts(
    prompts_file: str,
    hf_model: str,
    device: torch.device,
    cache_dir: Optional[str] = None
) -> List[torch.Tensor]:
    """
    Load cached embeddings or generate new ones if cache is invalid or missing.
    
    This function implements an intelligent caching mechanism:
    1. Locates the prompts file in current directory or cache directory
    2. Checks if cached embeddings exist in the specified cache directory
    3. Compares timestamps to determine if cache is stale (prompts file modified)
    4. Loads cached embeddings if fresh, or generates and caches new ones if needed
    
    Args:
        prompts_file (str): Path or filename of the prompts JSON file
        hf_model (str): The HuggingFace model identifier to use for embedding
        device (torch.device): The device to run the model on
        cache_dir (Optional[str]): Directory to store/load cached embeddings.
                                   If None, caching is disabled.
    
    Returns:
        List[torch.Tensor]: List of embedded tensors, one for each prompt
        
    Note:
        - Cache files are named based on prompts filename and model name
        - Timestamps are compared to detect when prompts file has been updated
        - Cache directory is created automatically if it doesn't exist
        - Prompts file is searched in current directory and cache directory
    """
    # Find the prompts file in available locations
    prompts_path = find_prompts_file(prompts_file, cache_dir)
    print(f"Found prompts file at: {prompts_path}")
    
    # Load prompts from file (only once)
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
    
    # If no cache directory specified, always embed
    if cache_dir is None:
        print(f"No cache directory specified, embedding prompts")
        return embed_prompts(prompts, hf_model, device)
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filename
    cache_filename = get_cache_filename(prompts_path, hf_model)
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Check if we need to re-embed
    if should_reembed(prompts_path, cache_path):
        print(f"Cache miss or stale cache for {prompts_path} with model {hf_model}")
        print(f"Embedding prompts and saving to {cache_path}")
        
        # Embed prompts
        embedded_tensors = embed_prompts(prompts, hf_model, device)
        
        # Save to cache
        torch.save(embedded_tensors, cache_path)
        print(f"Saved embedded prompts to cache: {cache_path}")
        
        return embedded_tensors
    else:
        print(f"Loading cached embeddings from {cache_path}")
        embedded_tensors = torch.load(cache_path, weights_only=False)
        return embedded_tensors
    

def collate_prompts(batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for prompts with variable length sequences.
    
    Takes a batch of tensors with shape [L, features] where L varies across samples,
    pads them to the same length, and creates a mask indicating valid tokens.
    
    Args:
        batch: List of tensors, each with shape [L_i, features] where L_i varies
        
    Returns:
        tuple: (padded_batch, mask)
            - padded_batch: Tensor of shape [batch_size, max_L, features]
            - mask: Boolean tensor of shape [batch_size, max_L] where True indicates valid tokens
    """
    if not batch:
        return torch.empty(0), torch.empty(0, dtype=torch.bool)
    
    # Get dimensions
    batch_size = len(batch)
    
    if isinstance(batch[0], tuple):
        batch = torch.stack([item[0] for item in batch], dim=0)
    
    feature_dim = batch[0].shape[-1]
    lengths = [tensor.shape[0] for tensor in batch]
    max_length = max(lengths)
    
    # Create padded tensor and mask
    padded_batch = torch.zeros(batch_size, max_length, feature_dim, dtype=batch[0].dtype)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    
    # Fill in the data and mask
    for i, tensor in enumerate(batch):
        length = lengths[i]
        padded_batch[i, :length] = tensor
        mask[i, :length] = True
    
    return padded_batch, mask
