import gc
from functools import partial
from typing import Callable, List, Union

import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from eap.eap_graph import EAPGraph

def EAP_corrupted_forward_hook(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"], 
    graph: EAPGraph
):
    """Hook for capturing corrupted activations during forward pass."""
    hook_slice = graph.get_hook_slice(hook.name)
    if activations.ndim == 3:
        # For residual layer or MLP
        upstream_activations_difference[:, :, hook_slice, :] = -activations.unsqueeze(-2)
    elif activations.ndim == 4:
        # For attention layer
        upstream_activations_difference[:, :, hook_slice, :] = -activations

def EAP_clean_forward_hook(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"], 
    graph: EAPGraph
):
    """Hook for capturing clean activations during forward pass."""
    hook_slice = graph.get_hook_slice(hook.name)
    if activations.ndim == 3:
        upstream_activations_difference[:, :, hook_slice, :] += activations.unsqueeze(-2)
    elif activations.ndim == 4:
        upstream_activations_difference[:, :, hook_slice, :] += activations

def EAP_clean_backward_hook(
    grad: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"],
    graph: EAPGraph
):
    """Hook for computing EAP scores during backward pass."""
    hook_slice = graph.get_hook_slice(hook.name)
    earlier_upstream_nodes_slice = graph.get_slice_previous_upstream_nodes(hook)

    if grad.ndim == 3:
        grad_expanded = grad.unsqueeze(-2)
    else:
        grad_expanded = grad
        
    # Computing EAP scores for attention components
    result = torch.matmul(
        upstream_activations_difference[:, :, earlier_upstream_nodes_slice],
        grad_expanded.transpose(-1, -2)
    ).sum(dim=0).sum(dim=0)

    graph.eap_scores[earlier_upstream_nodes_slice, hook_slice] += result

def apply_log_transform(tensor: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
    """
    Apply log transform to the absolute values of the tensor with scaling.
    
    Args:
        tensor: Input tensor
        scale: Scaling factor to apply before taking log
    
    Returns:
        Log-transformed tensor of absolute values
    """
    # Print original value ranges
    print(f"Original value range: [{tensor.min().item():.2e}, {tensor.max().item():.2e}]")
    
    abs_values = torch.abs(tensor)
    print(f"After abs value range: [{abs_values.min().item():.2e}, {abs_values.max().item():.2e}]")
    
    scaled_values = abs_values * scale
    print(f"After scaling (x{scale}) range: [{scaled_values.min().item():.2e}, {scaled_values.max().item():.2e}]")
    
    log_values = torch.log(scaled_values)
    print(f"After log range: [{log_values.min().item():.2e}, {log_values.max().item():.2e}]")
    
    # Check for -inf values
    if torch.any(torch.isinf(log_values)):
        print(f"Warning: Found {torch.sum(torch.isinf(log_values)).item()} -inf values")
        # Print a few example values that led to -inf
        inf_mask = torch.isinf(log_values)
        original_values = tensor[inf_mask][:5]
        print(f"Example original values that led to -inf: {original_values.tolist()}")
    
    return log_values

def EAP(
    model: HookedTransformer,
    clean_tokens: Int[Tensor, "batch_size seq_len"],
    corrupted_tokens: Int[Tensor, "batch_size seq_len"],
    metric: Callable,
    upstream_nodes: List[str] = None,
    downstream_nodes: List[str] = None,
    batch_size: int = 1,
    apply_log: bool = True,
    scale: float = 1000.0,
) -> EAPGraph:
    """
    Run Exploratory Attribution Patching (EAP) analysis.
    
    Args:
        model: HookedTransformer model to analyze
        clean_tokens: Clean input tokens
        corrupted_tokens: Corrupted input tokens
        metric: Metric function for evaluation
        upstream_nodes: List of upstream node types to analyze
        downstream_nodes: List of downstream node types to analyze
        batch_size: Batch size for processing
        apply_log: Whether to apply log transform to scores
        scale: Scaling factor for log transform
    
    Returns:
        EAPGraph containing the analysis results
    """
    # Input validation
    assert clean_tokens.shape == corrupted_tokens.shape, "Shape mismatch between clean and corrupted tokens"
    num_prompts, seq_len = clean_tokens.shape[0], clean_tokens.shape[1]
    assert num_prompts % batch_size == 0, "Number of prompts must be divisible by batch size"

    # Create graph and initialize
    graph = EAPGraph(model.cfg, upstream_nodes, downstream_nodes)
    graph.reset_scores()

    # Initialize activation difference tensor
    upstream_activations_difference = torch.zeros(
        (batch_size, seq_len, graph.n_upstream_nodes, model.cfg.d_model),
        device=model.cfg.device,
        dtype=model.cfg.dtype,
        requires_grad=False
    )

    # Set up hook filters and functions
    upstream_hook_filter = lambda name: any(name.endswith(hook) for hook in graph.upstream_hooks)
    downstream_hook_filter = lambda name: any(name.endswith(hook) for hook in graph.downstream_hooks)

    corrupted_upstream_hook_fn = partial(
        EAP_corrupted_forward_hook,
        upstream_activations_difference=upstream_activations_difference,
        graph=graph
    )
    clean_upstream_hook_fn = partial(
        EAP_clean_forward_hook,
        upstream_activations_difference=upstream_activations_difference,
        graph=graph
    )
    clean_downstream_hook_fn = partial(
        EAP_clean_backward_hook,
        upstream_activations_difference=upstream_activations_difference,
        graph=graph
    )

    # Process batches
    for idx in tqdm(range(0, num_prompts, batch_size)):
        batch_slice = slice(idx, idx + batch_size)
        
        # Corrupted forward pass
        model.add_hook(upstream_hook_filter, corrupted_upstream_hook_fn, "fwd")
        with torch.no_grad():
            model(corrupted_tokens[batch_slice].to(model.cfg.device), return_type=None)

        # Clean forward and backward pass
        model.reset_hooks()
        model.add_hook(upstream_hook_filter, clean_upstream_hook_fn, "fwd")
        model.add_hook(downstream_hook_filter, clean_downstream_hook_fn, "bwd")

        value = metric(model(clean_tokens[batch_slice].to(model.cfg.device), return_type="logits"))
        value.backward()
        
        model.zero_grad()
        upstream_activations_difference.zero_()

    # Cleanup
    del upstream_activations_difference
    gc.collect()
    torch.cuda.empty_cache()
    model.reset_hooks()

    # Post-process scores
    graph.eap_scores /= num_prompts
    
    if apply_log:
        graph.eap_scores = apply_log_transform(graph.eap_scores, scale=scale)
    
    graph.eap_scores = graph.eap_scores.cpu()

    return graph
