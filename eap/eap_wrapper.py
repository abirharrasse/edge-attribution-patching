def EAP_clean_backward_hook(
    grad: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"],
    graph: EAPGraph
):
    hook_slice = graph.get_hook_slice(hook.name)
    earlier_upstream_nodes_slice = graph.get_slice_previous_upstream_nodes(hook)

    if grad.ndim == 3:
        grad_expanded = grad.unsqueeze(-2)
    else:
        grad_expanded = grad
        
    # Debug gradient magnitudes
    print(f"\nGradient stats for {hook.name}:")
    print(f"Gradient shape: {grad_expanded.shape}")
    print(f"Gradient mean abs: {torch.abs(grad_expanded).mean().item():.6f}")
    print(f"Gradient std: {grad_expanded.std().item():.6f}")
    print(f"Gradient min: {grad_expanded.min().item():.6f}")
    print(f"Gradient max: {grad_expanded.max().item():.6f}")
        
    # Debug activation differences
    print(f"\nActivation difference stats for {hook.name}:")
    diff_slice = upstream_activations_difference[:, :, earlier_upstream_nodes_slice]
    print(f"Activation difference shape: {diff_slice.shape}")
    print(f"Activation difference mean abs: {torch.abs(diff_slice).mean().item():.6f}")
    print(f"Activation difference std: {diff_slice.std().item():.6f}")
    print(f"Activation difference min: {diff_slice.min().item():.6f}")
    print(f"Activation difference max: {diff_slice.max().item():.6f}")

    # Computing EAP scores for attention components
    result = torch.matmul(
        upstream_activations_difference[:, :, earlier_upstream_nodes_slice],
        grad_expanded.transpose(-1, -2)
    ).sum(dim=0).sum(dim=0)
    
    # Debug EAP score components
    print(f"\nEAP score stats for {hook.name}:")
    print(f"EAP score shape: {result.shape}")
    print(f"EAP score mean abs: {torch.abs(result).mean().item():.6f}")
    print(f"EAP score std: {result.std().item():.6f}")
    print(f"EAP score min: {result.min().item():.6f}")
    print(f"EAP score max: {result.max().item():.6f}")

    graph.eap_scores[earlier_upstream_nodes_slice, hook_slice] += result

def EAP(
    model: HookedTransformer,
    clean_tokens: Int[Tensor, "batch_size seq_len"],
    corrupted_tokens: Int[Tensor, "batch_size seq_len"],
    metric: Callable,
    upstream_nodes: List[str]=None,
    downstream_nodes: List[str]=None,
    batch_size: int=1,
    debug: bool=False,
):
    # Create graph with updated node handling
    graph = EAPGraph(model.cfg, upstream_nodes, downstream_nodes)

    assert clean_tokens.shape == corrupted_tokens.shape, "Shape mismatch between clean and corrupted tokens"
    num_prompts, seq_len = clean_tokens.shape[0], clean_tokens.shape[1]
    assert num_prompts % batch_size == 0, "Number of prompts must be divisible by batch size"

    upstream_activations_difference = torch.zeros(
        (batch_size, seq_len, graph.n_upstream_nodes, model.cfg.d_model),
        device=model.cfg.device,
        dtype=model.cfg.dtype,
        requires_grad=False
    )

    if debug:
        print("\nInitial upstream_activations_difference stats:")
        print(f"Shape: {upstream_activations_difference.shape}")
        print(f"Device: {upstream_activations_difference.device}")
        print(f"Dtype: {upstream_activations_difference.dtype}")

    graph.reset_scores()

    # Hook filters updated for new hook names
    upstream_hook_filter = lambda name: any(name.endswith(hook) for hook in graph.upstream_hooks)
    downstream_hook_filter = lambda name: any(name.endswith(hook) for hook in graph.downstream_hooks)

    for idx in tqdm(range(0, num_prompts, batch_size)):
        if debug and idx == 0:  # Only print for first batch to avoid spam
            print(f"\nProcessing batch {idx}/{num_prompts}")
            
        # Corrupted input forward pass
        model.add_hook(upstream_hook_filter, corruped_upstream_hook_fn, "fwd")
        with torch.no_grad(): 
            corrupted_tokens = corrupted_tokens.to(model.cfg.device)
            model(corrupted_tokens[idx:idx+batch_size], return_type=None)        

        if debug and idx == 0:
            print("\nAfter corrupted forward pass:")
            print(f"Activation difference mean abs: {torch.abs(upstream_activations_difference).mean().item():.6f}")
            print(f"Activation difference std: {upstream_activations_difference.std().item():.6f}")

        # Clean input forward and backward pass
        model.reset_hooks()
        model.add_hook(upstream_hook_filter, clean_upstream_hook_fn, "fwd")
        model.add_hook(downstream_hook_filter, clean_downstream_hook_fn, "bwd")

        clean_tokens = clean_tokens.to(model.cfg.device)
        value = metric(model(clean_tokens[idx:idx+batch_size], return_type="logits"))
        
        if debug and idx == 0:
            print(f"\nMetric value: {value.item():.6f}")
            
        value.backward()
        
        if debug and idx == 0:
            print("\nAfter backward pass:")
            print(f"EAP scores mean abs: {torch.abs(graph.eap_scores).mean().item():.6f}")
            print(f"EAP scores std: {graph.eap_scores.std().item():.6f}")
        
        model.zero_grad()
        upstream_activations_difference *= 0

    del upstream_activations_difference
    gc.collect()
    torch.cuda.empty_cache()
    model.reset_hooks()

    graph.eap_scores /= num_prompts
    graph.eap_scores = graph.eap_scores.cpu()

    if debug:
        print("\nFinal EAP scores stats:")
        print(f"Mean abs: {torch.abs(graph.eap_scores).mean().item():.6f}")
        print(f"Std: {graph.eap_scores.std().item():.6f}")
        print(f"Min: {graph.eap_scores.min().item():.6f}")
        print(f"Max: {graph.eap_scores.max().item():.6f}")

    return graph
