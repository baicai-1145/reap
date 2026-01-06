from __future__ import annotations
import time
import pickle
import logging
import dataclasses
import pathlib
import re
import time
from typing import Any
import gc
import yaml
import shutil

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset, Features, Value
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser

from accelerate.utils import set_seed
from accelerate.hooks import remove_hook_from_module


from reap.args import (
    ReapArgs,
    ModelArgs,
    DatasetArgs,
    ObserverArgs,
    ClusterArgs,
    KdArgs,
    EvalArgs,
    MergeArgs,
)
from reap.merge import MergeMethod, MoEExpertMerger
from reap.data import DATASET_REGISTRY
from reap.observer import OBSERVER_CONFIG_REGISTRY, MoETransformerObserver
from reap.cluster import (
    get_penalty_vector,
    hierarchical_clustering,
    dynamic_frequency_penalized_clustering,
    multi_layer_hierarchical_clustering,
    mc_smoe_clustering,
    multi_layer_kmeans_clustering,
    multi_layer_kmeans_clustering_on_ca,
    restricted_hierarchical_clustering,
    kmeans_clustering
)
from reap.model_util import get_moe, assert_merge, MODEL_ATTRS, patched_model_map, get_super_expert_indices
from reap.eval import run_evaluate
from reap.cluster_plots import plot_cluster_analysis
from reap.metrics import get_distance_fn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> tuple[dataclasses.Dataclass]:
    parser = HfArgumentParser(
        (
            ReapArgs,
            ModelArgs,
            DatasetArgs,
            ObserverArgs,
            ClusterArgs,
            KdArgs,
            EvalArgs,
            MergeArgs,
        )
    )
    args = parser.parse_args_into_dataclasses()
    return args


def str_to_directory_name(s: str) -> str:
    """Convert a string to a valid directory name by replacing special characters."""
    return re.sub(r"[^\w\-_.]", "_", s)


def create_results_directory(model_name: str, dataset_name: str) -> pathlib.Path:
    """Create a clean directory name from model and dataset names."""
    model_clean = model_name.split("/")[-1]
    dataset_clean = dataset_name.split("/")[-1]

    # Create clean directory name by removing special characters
    model_clean = str_to_directory_name(model_clean)
    dataset_clean = str_to_directory_name(dataset_clean)

    results_dir = pathlib.Path("./artifacts") / model_clean / dataset_clean

    if results_dir.exists():
        logger.warning(f"Directory '{results_dir}' already exists")
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created artifacts directory: {results_dir}")

    return results_dir


def record_activations(
    model, tokenizer, reap_args, model_args, ds_args, obs_args, results_dir
):
    # Fast path: if not splitting by category, we only use the "all" bucket.
    # If the observations file already exists and overwrite is disabled, load it
    # directly to avoid re-loading the dataset and re-running the model.
    if not obs_args.split_by_category:
        cat_dir = results_dir / "all"
        f_name = cat_dir / obs_args.output_file_name
        if f_name.exists() and not obs_args.overwrite_observations:
            logger.info("Found existing observations at %s. Loading...", f_name)
            return torch.load(f_name, weights_only=False)

    if ds_args.dataset_name == "combined":
        # just return the combined data
        cat_dir = results_dir / "all"
        f_name = cat_dir / obs_args.output_file_name
        if f_name.exists():
            return torch.load(f_name, weights_only=False)
        else:
            raise RuntimeError(
                f"Combined dataset requested but no pre-recorded data found at {f_name}"
            )
    try:
        ds_path = pathlib.Path(ds_args.dataset_name)
        if ds_path.exists():
            local_features: Features | None = None
            if ds_args.dataset_name == "artifacts/openthoughts_114k_samples_by_domain":
                # HuggingFace Datasets may infer null columns when a field is always null
                # in the first jsonl file(s). Later files with string values then fail to
                # cast (string -> null). Pin a stable schema for our local jsonl bundle.
                local_features = Features(
                    {
                        "problem": Value("string"),
                        "deepseek_reasoning": Value("string"),
                        "deepseek_solution": Value("string"),
                        "ground_truth_solution": Value("string"),
                        "domain": Value("string"),
                        "source": Value("string"),
                        "test_cases": Value("string"),
                        "starter_code": Value("string"),
                    }
                )
            if ds_path.is_dir():
                jsonl_files = sorted(
                    [str(p) for p in ds_path.glob("*.jsonl")]
                    + [str(p) for p in ds_path.glob("**/*.jsonl")]
                )
                # De-duplicate while preserving order
                seen = set()
                jsonl_files = [p for p in jsonl_files if not (p in seen or seen.add(p))]
                if not jsonl_files:
                    raise RuntimeError(
                        f"Local dataset directory has no .jsonl files: {ds_path}"
                    )
                raw_ds = load_dataset(
                    "json",
                    data_files=jsonl_files,
                    split="train",
                    streaming=False,
                    features=local_features,
                )
            else:
                raw_ds = load_dataset(
                    "json",
                    data_files={"train": str(ds_path)},
                    split="train",
                    streaming=False,
                    features=local_features,
                )
        elif ds_args.dataset_name == "allenai/c4":
            file_url = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"
            c4_single_file_dataset = load_dataset(
                "json", data_files={"train": file_url}, split="train", streaming=False
            )
            raw_ds = c4_single_file_dataset
        else:
            raw_ds = load_dataset(ds_args.dataset_name, split=ds_args.split)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{ds_args.dataset_name}': {e}")

    # load dataset processor
    proc_cls = DATASET_REGISTRY.get(ds_args.dataset_name)
    if proc_cls is None:
        raise ValueError(
            f"No DatasetProcessor registered for '{ds_args.dataset_name}'. "
            f"Supported: {list(DATASET_REGISTRY.keys())}"
        )

    # init processor & process dataset
    processor = proc_cls(
        dataset=raw_ds,
        tokenizer=tokenizer,
        max_input_len=obs_args.model_max_length,
        split=ds_args.split,
        split_by_category=obs_args.split_by_category,
        return_vllm_tokens_prompt=obs_args.return_vllm_tokens_prompt,
        truncate=obs_args.truncate,
    )
    category_data_batches = processor.get_processed_dataset(
        samples_per_category=obs_args.samples_per_category,
    )
    logger.info(
        "Loaded and processed data for categories: %s",
        str(list(category_data_batches.keys())),
    )

    # load observer and hook model
    try:
        renormalize_router_weights = getattr(model.config, "norm_topk_prob", False) and obs_args.renormalize_router_weights
        if renormalize_router_weights:
            logger.info("Renormalizing topk router weights to sum to 1.")
        observer_config = OBSERVER_CONFIG_REGISTRY[model.__class__.__name__](
            # distance_measure=obs_args.distance_measure,
            distance_measure='cosine',
            renormalize_router_weights=renormalize_router_weights,
            record_pruning_metrics_only=obs_args.record_pruning_metrics_only,
        )
    except KeyError:
        raise ValueError(
            f"No observer configuration registered for model '{model.__class__.__name__}'. "
            f"Supported: {list(OBSERVER_CONFIG_REGISTRY.keys())}"
        )
    observer = MoETransformerObserver(
        model=model,
        hook_config=observer_config,
    )

    if reap_args.profile:
        # profile at max len
        with torch.no_grad():
            try:
                model_max_length = obs_args.model_max_length
                if model_max_length is None:
                    model_max_length = tokenizer.model_max_length
                logger.info(f"Profiling at model max length: {model_max_length}.")
                s = "hello " * model_max_length
                tokenized = tokenizer(
                    [s],
                    return_tensors="pt",
                    truncation=True,
                    max_length=model_max_length,
                )
                tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
                for _ in range(2):
                    _ = model(**tokenized)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to run model with max input length {model_max_length}: {e}"
                )
        logger.info(
            f"Model {model_args.model_name} successfully loaded and profiled at max length {model_max_length}."
        )
        observer.reset()

    # run samples over model and save observer state
    with torch.no_grad():
        last_saved_path: pathlib.Path | None = None
        for category, cat_data in category_data_batches.items():
            logger.info(f"Processing category: {category}...")
            cat_dir = results_dir / str_to_directory_name(category)
            cat_dir.mkdir(parents=True, exist_ok=True)
            f_name = cat_dir / obs_args.output_file_name
            if f_name.exists() and not obs_args.overwrite_observations:
                logger.info(f"Category '{category}' previously processed. Skipping...")
                last_saved_path = f_name
                continue

            logger.info("No previous data found. Starting LEO (Layer-wise Expert-wise Observation) optimized path...")
            
            # 1. Initial Hidden States (Embeddings)
            all_samples = torch.cat(cat_data, dim=0) # (TotalSamples, SeqLen)
            num_total_samples = all_samples.shape[0]
            batch_size = obs_args.batch_size # Dynamically use the batch size from args
            seq_len = all_samples.shape[1]
            
            # Prepare Attention Mask (typically all ones for calibration if no padding)
            # and position embeddings
            # Base causal mask (2D: SeqLen x SeqLen)
            target_device = getattr(model, "target_device", torch.device("cuda:0"))
            causal_mask = torch.full((seq_len, seq_len), fill_value=float("-inf"), device=target_device, dtype=model.dtype)
            causal_mask = torch.triu(causal_mask, diagonal=1)

            h_states = []
            logger.info("Computing initial embeddings...")
            for i in range(0, num_total_samples, batch_size):
                batch = all_samples[i:i+batch_size].to(model.device)
                h = model.model.embed_tokens(batch)
                h_states.append(h.cpu())
                del batch
            h_states = torch.cat(h_states, dim=0)
            target_device = getattr(model, "target_device", torch.device("cuda:0"))
            
            # Buffer Optimization: Use two GPU buffers to eliminate ALL CPU-GPU traffic
            # 32GB (Input) + 32GB (Output) = 64GB. With 1 layer (3.3GB) = 67.3GB. 
            # Safe for H100 (80GB).
            logger.info(f"Moving activations to GPU buffers (64GB total)...")
            h_states = h_states.to(target_device)
            next_h_states_buffer = torch.empty_like(h_states)
            
            # Stream for asynchronous weight loading
            load_stream = torch.cuda.Stream(device=target_device)
            
            # 2. Iterate through layers
            num_layers = len(model.model.layers)
            num_experts = model.config.num_experts
            
            # Initial load for first layer
            model.model.layers[0].to(target_device)

            # Use tqdm for the outer loop so you can see progress
            pbar = tqdm(range(num_layers), desc="Processing Layers")
            for l_idx in pbar:
                layer = model.model.layers[l_idx]
                moe_block = layer.mlp
                
                # PREFETCH NEXT LAYER: While computing current, load next
                if l_idx + 1 < num_layers:
                    with torch.cuda.stream(load_stream):
                        model.model.layers[l_idx + 1].to(target_device, non_blocking=True)

                total_tokens = num_total_samples * seq_len
                # Reduced chunk size to 48k to give GPU more "breathing room" (prevents fragmentation slowdown)
                token_chunk_size = 128 * 1024 
                flat_h_states = h_states.view(total_tokens, -1)
                flat_next_h_states = next_h_states_buffer.view(total_tokens, -1)

                for c_start in range(0, total_tokens, token_chunk_size):
                    c_end = min(c_start + token_chunk_size, total_tokens)
                    chunk_len = c_end - c_start
                    chunk_h = flat_h_states[c_start:c_end]
                    
                    temp_batch_size = (chunk_len + seq_len - 1) // seq_len
                    chunk_h_padded = torch.nn.functional.pad(chunk_h, (0, 0, 0, temp_batch_size * seq_len - chunk_len))
                    chunk_h_reshaped = chunk_h_padded.view(temp_batch_size, seq_len, -1)
                    
                    with torch.no_grad():
                        # Attention Stage
                        residual = chunk_h_reshaped
                        hidden_states = layer.input_layernorm(chunk_h_reshaped)
                        
                        current_mask = causal_mask.view(1, 1, seq_len, seq_len).expand(temp_batch_size, 1, -1, -1)
                        position_ids = torch.arange(seq_len, device=target_device).expand(temp_batch_size, -1)
                        pos_emb = model.model.rotary_emb(hidden_states, position_ids)

                        if hasattr(layer, "linear_attn"):
                            hidden_states = layer.linear_attn(hidden_states, attention_mask=current_mask)[0]
                        else:
                            hidden_states = layer.self_attn(hidden_states, attention_mask=current_mask, position_embeddings=pos_emb)[0]
                        
                        hidden_states = residual + hidden_states
                        residual = hidden_states
                        moe_input = layer.post_attention_layernorm(hidden_states)
                        
                        # MoE Stage
                        moe_input_flat = moe_input.view(-1, model.config.hidden_size)[:chunk_len]
                        router_logits = moe_block.gate(moe_input_flat)
                        
                        top_k = model.config.num_experts_per_tok
                        routing_weights = torch.nn.functional.softmax(
                            router_logits, dim=-1, dtype=torch.float32
                        )
                        _, selected_experts = torch.topk(router_logits, top_k, dim=-1)
                        if renormalize_router_weights:
                            topk_weights = torch.gather(
                                routing_weights, 1, selected_experts
                            )
                            routing_weights = routing_weights / topk_weights.sum(
                                dim=-1, keepdim=True
                            )
                            routing_weights = torch.clamp(
                                routing_weights,
                                min=torch.finfo(routing_weights.dtype).eps,
                            )
                        
                        moe_out_chunk = torch.zeros_like(moe_input_flat)
                        
                        c_ean_sum = torch.zeros(
                            num_experts, device=target_device, dtype=torch.float64
                        )
                        c_w_ean_sum = torch.zeros(
                            num_experts, device=target_device, dtype=torch.float64
                        )
                        c_w_ean_sum_l2 = torch.zeros(
                            num_experts, device=target_device, dtype=torch.float64
                        )
                        c_w_freq_sum = torch.zeros(
                            num_experts, device=target_device, dtype=torch.float64
                        )
                        c_freq = torch.zeros(
                            num_experts, device=target_device, dtype=torch.long
                        )
                        c_max_act = torch.zeros(num_experts, device=target_device)
                        c_routed_ca_sum = torch.zeros(
                            (num_experts, model.config.hidden_size),
                            device=target_device,
                            dtype=torch.float32,
                        )

                        if hasattr(moe_block, "shared_expert"):
                            shared_out = moe_block.shared_expert(moe_input_flat)
                            shared_gate = torch.sigmoid(moe_block.shared_expert_gate(moe_input_flat))
                            moe_out_chunk += (shared_out * shared_gate)

                        # Expert Sort Optimization
                        flat_selected_experts = selected_experts.view(-1)
                        expert_ids, token_indices_in_chunk = torch.sort(flat_selected_experts)
                        tokens_per_expert = torch.bincount(expert_ids, minlength=num_experts)
                        expert_offsets = torch.cumsum(tokens_per_expert, dim=0) - tokens_per_expert

                        for exp_idx in range(num_experts):
                            num_tokens = tokens_per_expert[exp_idx].item()
                            if num_tokens == 0: continue
                            
                            start_off = expert_offsets[exp_idx].item()
                            chunk_token_idx = token_indices_in_chunk[start_off : start_off + num_tokens] // top_k
                            
                            expert_out = moe_block.experts[exp_idx](moe_input_flat[chunk_token_idx])
                            exp_weights = routing_weights[chunk_token_idx, exp_idx].unsqueeze(-1)
                            
                            moe_out_chunk.index_add_(
                                0,
                                chunk_token_idx,
                                expert_out * exp_weights.to(expert_out.dtype),
                            )
                            
                            ean_norm = torch.linalg.norm(expert_out, dim=-1).to(
                                torch.float32
                            )
                            exp_w = exp_weights.squeeze(-1).to(torch.float32)
                            ean_norm_l2 = ean_norm * ean_norm
                            c_ean_sum[exp_idx] += ean_norm.sum().to(torch.float64)
                            c_w_ean_sum[exp_idx] += (ean_norm * exp_w).sum().to(
                                torch.float64
                            )
                            c_w_ean_sum_l2[exp_idx] += (ean_norm_l2 * exp_w).sum().to(
                                torch.float64
                            )
                            c_w_freq_sum[exp_idx] += exp_w.sum().to(torch.float64)
                            c_freq[exp_idx] += num_tokens
                            c_max_act[exp_idx] = torch.max(c_max_act[exp_idx], expert_out.max())
                            c_routed_ca_sum[exp_idx] += expert_out.to(torch.float32).sum(
                                dim=0
                            )

                        if l_idx not in observer.state:
                            observer.state[l_idx] = observer._initialize_state((moe_input_flat[0:1],), num_experts)
                        obs_state = observer.state[l_idx]
                        obs_state["ean_sum"] += c_ean_sum.cpu()
                        obs_state["weighted_ean_sum"] += c_w_ean_sum.cpu()
                        obs_state["expert_frequency"] += c_freq.cpu()
                        obs_state["weighted_expert_frequency_sum"] += c_w_freq_sum.cpu()
                        if "weighted_ean_sum_l2" not in obs_state:
                            obs_state["weighted_ean_sum_l2"] = torch.zeros(
                                (num_experts,),
                                device="cpu",
                                dtype=torch.float64,
                                requires_grad=False,
                            )
                        obs_state["weighted_ean_sum_l2"] += c_w_ean_sum_l2.cpu()

                        if "routed_characteristic_activation_sum" not in obs_state:
                            obs_state["routed_characteristic_activation_sum"] = torch.zeros(
                                (num_experts, model.config.hidden_size),
                                device="cpu",
                                dtype=torch.float32,
                                requires_grad=False,
                            )
                            obs_state["routed_characteristic_activation_count"] = torch.zeros(
                                (num_experts,),
                                device="cpu",
                                dtype=torch.long,
                                requires_grad=False,
                            )
                        obs_state["routed_characteristic_activation_sum"] += (
                            c_routed_ca_sum.cpu()
                        )
                        obs_state["routed_characteristic_activation_count"] += c_freq.cpu()
                        c_ean_mean = torch.zeros(
                            num_experts, device=target_device, dtype=torch.float32
                        )
                        c_reap = torch.zeros(
                            num_experts, device=target_device, dtype=torch.float32
                        )
                        nonzero = c_freq > 0
                        c_ean_mean[nonzero] = (c_ean_sum[nonzero] / c_freq[nonzero]).to(
                            torch.float32
                        )
                        c_reap[nonzero] = (c_w_ean_sum[nonzero] / c_freq[nonzero]).to(
                            torch.float32
                        )
                        obs_state["ean_mean"].update(c_ean_mean.cpu(), c_freq.cpu())
                        obs_state["reap"].update(c_reap.cpu(), c_freq.cpu())
                        obs_state["max_activations"] = torch.max(obs_state["max_activations"], c_max_act.cpu())
                            
                        flat_next_h_states[c_start:c_end] = (residual.view(-1, model.config.hidden_size)[:chunk_len] + moe_out_chunk)

                # Swap Buffers
                h_states, next_h_states_buffer = next_h_states_buffer, h_states
                obs_state["total_tokens"] += total_tokens
                
                # Offload current layer asynchronously
                with torch.cuda.stream(load_stream):
                    layer.to("cpu", non_blocking=True)
                
                # Periodic cache clearing to combat fragmentation every 8 layers
                if l_idx % 8 == 0:
                    torch.cuda.empty_cache()
                
                # Synchronize before moving to next layer
                torch.cuda.synchronize() 
                # torch.cuda.empty_cache() # Avoid excessive syncing

            # Materialize lightweight derived fields before saving.
            # We store sums/counts during LEO to minimize per-step work, then convert
            # to means here so downstream pruning/clustering can consume tensors.
            for layer_idx, layer_state in observer.state.items():
                if "routed_characteristic_activation_sum" in layer_state:
                    cnt = layer_state["routed_characteristic_activation_count"]
                    denom = cnt.clamp(min=1).unsqueeze(-1).to(torch.float32)
                    mean = layer_state["routed_characteristic_activation_sum"] / denom
                    mean[cnt == 0] = 0
                    layer_state["routed_characteristic_activation"] = mean
                    del layer_state["routed_characteristic_activation_sum"]
                    del layer_state["routed_characteristic_activation_count"]

                if "weighted_ean_sum_l2" in layer_state:
                    ef = layer_state["expert_frequency"].to(torch.float64)
                    denom = ef.clone()
                    denom[denom == 0] = 1
                    reap_l2 = (layer_state["weighted_ean_sum_l2"] / denom).to(
                        torch.float32
                    )
                    reap_l2[ef == 0] = 0
                    layer_state["reap_l2"] = reap_l2

            observer.save_state(f_name)
            logger.info(f"Category '{category}' finished via LEO path.")
            observer.reset()
            last_saved_path = f_name

    observer.close_hooks()
    if last_saved_path is None:
        raise RuntimeError("No observations were produced or found on disk.")
    return torch.load(last_saved_path, weights_only=False)


def cluster(
    data: dict[int, dict[str, Any]],
    num_clusters: int,
    cluster_args: ClusterArgs,
    distance_measure: str,
    results_dir: pathlib.Path,
) -> dict[int, torch.Tensor]:
    """Cluster the model's experts based on the specified clustering method."""
    logger.info(f"Clustering experts using settings:\n{cluster_args.__str__()}\n")

    cluster_labels = {}
    distances = {}
    all_layer_expert_proba = {}
    if cluster_args.singleton_super_experts or cluster_args.singleton_outlier_experts:
        super_expert_idx = get_super_expert_indices(data, include_last_layers=cluster_args.singleton_outlier_experts)
    for layer in tqdm(data, "Clustering experts..."):
        expert_prob = data[layer]["expert_frequency"] / data[layer]["total_tokens"]
        ttm_sim_matrix = None
        try:
            ttm_sim_matrix = data[layer]["ttm_similarity_matrix"]
        except KeyError:
            pass
        online_characteristic_activation_dist = None
        try:
            online_characteristic_activation_dist = data[layer][
                "online_characteristic_activation_dist"
            ]
        except KeyError:
            pass
        ca = data[layer]["characteristic_activation"]
        routed_ca = None
        try:
            routed_ca = data[layer]["routed_characteristic_activation"]
        except KeyError:
            pass
        router_logits = data[layer]["router_logit_similiarity"]

        expert_similarity_scores = {
            "ttm": ttm_sim_matrix,
            "dynamic_ttm": ttm_sim_matrix,
            "characteristic_activation": ca,
            "routed_characteristic_activation": routed_ca,
            "router_logits": router_logits,
            "online_characteristic_activation_dist": online_characteristic_activation_dist,
        }
        distance = expert_similarity_scores[cluster_args.expert_sim]

        if cluster_args.expert_sim in [
            "characteristic_activation",
            "routed_characteristic_activation",
            "router_logits",
        ] and cluster_args.cluster_method != "kmeans":
            # get NxN similarity matrix for vector metrics
            distance_fn = get_distance_fn(distance_measure)
            distance = distance_fn(distance.unsqueeze(0), distance.unsqueeze(1))

        
        if cluster_args.singleton_super_experts:
            # set super expert distance to max
            super_experts_in_layer = super_expert_idx[super_expert_idx[:, 0] == layer][:, 1]
            if len(super_experts_in_layer) > 0:
                max_value = torch.finfo(distance.dtype).max
                distance[:, super_experts_in_layer] = max_value
                distance[super_experts_in_layer, :] = max_value

        distances[layer] = distance
        all_layer_expert_proba[layer] = expert_prob
        if cluster_args.multi_layer or cluster_args.cluster_method == "mc_smoe":
            continue
        if cluster_args.frequency_penalty and cluster_args.expert_sim != "dynamic_ttm":
            penalty = get_penalty_vector(
                expert_prob,
                cluster_args.softmax_temperature,
            )
            penalty_matrix = penalty.unsqueeze(0) + penalty.unsqueeze(1)
            penalized_distance = distance * penalty_matrix
            penalized_distance[penalized_distance.isnan()] = float("inf")
            distance = penalized_distance

        if cluster_args.expert_sim == "dynamic_ttm":
            cluster_label = dynamic_frequency_penalized_clustering(
                distance,
                expert_prob,
                num_clusters,
                cluster_args.softmax_temperature,
            )

        elif cluster_args.cluster_method == "agglomerative":
            if (
                hasattr(cluster_args, "max_cluster_size")
                and cluster_args.max_cluster_size is None
            ):
                cluster_label = hierarchical_clustering(
                    distance,
                    cluster_args.linkage_method,
                    num_clusters,
                )
            else:
                cluster_label = restricted_hierarchical_clustering(
                    distance,
                    cluster_args.linkage_method,
                    num_clusters,
                    max_cluster_size=cluster_args.max_cluster_size,
                )
            if isinstance(cluster_label, np.ndarray):
                cluster_label = torch.tensor(cluster_label)

        elif cluster_args.cluster_method == "kmeans":
            cluster_label = kmeans_clustering(distance, num_clusters)

        else:
            raise NotImplementedError(
                f"Clustering method '{cluster_args.cluster_method}' is not implemented."
            )
        cluster_labels[layer] = cluster_label

    if cluster_args.multi_layer:
        # we have parsed distances, time to cluster across layers]
        logger.info(
            f"Multi layer clustering with multi_layer={cluster_args.multi_layer}"
        )
        if cluster_args.cluster_method == "agglomerative":
            cluster_labels = multi_layer_hierarchical_clustering(
                distances,
                cluster_args.multi_layer,
                cluster_args.linkage_method,
                num_clusters,
            )
        elif cluster_args.cluster_method == "kmeans": 
            # try v2:
            if cluster_args.expert_sim != 'characteristic_activation':
                raise ValueError("multi_layer kmeans clustering on ca only implemented for characteristic_activation expert sim")
            cluster_labels = multi_layer_kmeans_clustering_on_ca(
                distances,
                num_layers=cluster_args.multi_layer,
                n_clusters=num_clusters,
            )
            
            # cluster_labels = multi_layer_kmeans_clustering(
            #     distances,
            #     num_layers=cluster_args.multi_layer,
            #     n_clusters=num_clusters,
            # )

    if cluster_args.cluster_method == "mc_smoe":
        logger.info(f"Performing MC-SMoE adpative layer-wise merging...")
        cluster_labels = mc_smoe_clustering(
            distances,
            all_layer_expert_proba,
            total_clusters=len(distances) * num_clusters,
        )
    return cluster_labels


def merge(
    model: nn.Module,
    cluster_labels: dict[int, torch.Tensor],
    observer_data: dict[int, dict[str, Any]],
    merge_args: MergeArgs,
):
    """Merge experts based on the clustering results."""
    logger.info(f"Merging experts using method '{merge_args.merge_method}'")
    model_attrs = MODEL_ATTRS[model.__class__.__name__]

    try:
        merge_method = MergeMethod(merge_args.merge_method)
    except ValueError:
        raise NotImplementedError(
            f"Merge method '{merge_args.merge_method}' is not implemented. "
            f"Supported methods: {[method.value for method in MergeMethod]}"
        )

    for layer_idx, layer in enumerate(tqdm(cluster_labels, "Merging layers...")):
        if merge_args.skip_first and layer_idx == 0:
            logger.info(
                f"Skipping merging for layer {layer_idx} as per 'skip_first' argument."
            )
            continue

        if merge_args.skip_last and layer_idx == len(cluster_labels) - 1:
            logger.info(
                f"Skipping merging for layer {layer_idx} as per 'skip_last' argument."
            )
            continue

        expert_proba = (
            observer_data[layer]["expert_frequency"]
            / observer_data[layer]["total_tokens"]
        )
        cluster_label = cluster_labels[layer]
        moe = get_moe(model, layer)
        merger = MoEExpertMerger(
            moe=moe,
            cluster_label=cluster_label,
            expert_proba=expert_proba,
            model_attrs=model_attrs,
            merge_method=merge_method,
            dom_as_base=merge_args.dom_as_base,
            select_top_k=merge_args.select_top_k,
            permute=merge_args.permute,
            tie_tensors=merge_args.save_as_tied_params,
        )
        merger.merge_experts()
        # in case of non-uniform compression, update num_experts
        # TODO deal with router too
        # setattr(getattr(moe, model_attrs["num_experts"]), model_attrs["num_experts"], len(cluster_label.unique()))
        assert_merge(model, moe, cluster_label)


def save_merged_model(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    merged_model_dir: pathlib.Path,
    safe_serialization,
) -> pathlib.Path:
    logger.info("Saving merged model...")
    merged_model_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        model.save_pretrained(merged_model_dir, safe_serialization=safe_serialization)
        tokenizer.save_pretrained(merged_model_dir)
    except Exception as e:
        import pdb; breakpoint()
    end = time.time()
    logger.info(
        f"Merged model saved to {merged_model_dir} in {end - start:.2f} seconds"
    )
    return merged_model_dir


@torch.no_grad()
def smoke_test(model: nn.Module, tokenizer: AutoTokenizer):
    """Run a smoke test to ensure the model is functioning correctly."""
    target_device = getattr(model, "target_device", torch.device("cuda:0"))
    logger.info(f"Moving model to {target_device} for fast smoke test...")
    model.to(target_device)
    
    prompt = "What is your name?"
    test_input = [
        {"role": "user", "content": prompt},
    ]
    # Use return_dict=True to get both input_ids and attention_mask
    inputs = tokenizer.apply_chat_template(
        test_input,
        return_tensors="pt",
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    ).to(target_device)
    
    logger.info("Generating response (max 20 tokens)...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    logger.info("Smoke test response: %s", response[0])
    
    model.to("cpu")
    torch.cuda.empty_cache()


def get_model_dir(
    results_dir, num_clusters, cluster_labels, cluster_args, obs_args, merge_args
) -> pathlib.Path:
    cluster_desc = cluster_args.cluster_description
    if not cluster_desc:
        cluster_desc = (
            f"{cluster_args.expert_sim}_{obs_args.distance_measure}_{num_clusters}_"
            f"{cluster_args.linkage_method}_freq-penalty-{cluster_args.frequency_penalty}"
            f"_softmax-{cluster_args.softmax_temperature}_multi_layer-{cluster_args.multi_layer}"
        )
        if cluster_args.max_cluster_size is not None:
            cluster_desc += f"_max_size-{cluster_args.max_cluster_size}"
    merge_model_subdir_name = merge_args.merged_model_dir_name
    
    if not merge_model_subdir_name:
        merge_model_subdir_name = f"{merge_args.merge_method}-permute_{merge_args.permute}-skip_first_{merge_args.skip_first}-skip_last_{merge_args.skip_last}-multilayer_{cluster_args.multi_layer}"

    # Check for non uniform compression
    non_uniform_cluster_labels = (
        len(
            torch.unique(
                torch.tensor(
                    [
                        len(torch.unique(clusters))
                        for clusters in cluster_labels.values()
                    ]
                )
            )
        )
        > 1
    )
    if (
        non_uniform_cluster_labels
        or cluster_args.multi_layer
        or merge_args.skip_first
        or merge_args.skip_last
    ):
        logger.info("Detected non-uniform compression across layers.")
        merge_model_parent_dir_name = "non_uniform_merged_models"
    else:
        merge_model_parent_dir_name = "merged_models"

    merged_model_dir = (
        results_dir
        / merge_model_parent_dir_name
        / merge_model_subdir_name
        / cluster_desc
    )
    return merged_model_dir


def dump_args_to_yaml(
    merged_model_dir: pathlib.Path,
    reap_args: ReapArgs,
    model_args: ModelArgs,
    ds_args: DatasetArgs,
    obs_args: ObserverArgs,
    cluster_args: ClusterArgs,
    kd_args: KdArgs,
    eval_args: EvalArgs,
    merge_args: MergeArgs,
):
    """Dump all arguments to a YAML file."""
    all_args = {
        "reap_args": dataclasses.asdict(reap_args),
        "model_args": dataclasses.asdict(model_args),
        "ds_args": dataclasses.asdict(ds_args),
        "obs_args": dataclasses.asdict(obs_args),
        "cluster_args": dataclasses.asdict(cluster_args),
        "kd_args": dataclasses.asdict(kd_args),
        "eval_args": dataclasses.asdict(eval_args),
        "merge_args": dataclasses.asdict(merge_args),
    }

    def convert_paths_to_str(data):
        if isinstance(data, dict):
            return {k: convert_paths_to_str(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_paths_to_str(i) for i in data]
        elif isinstance(data, pathlib.Path):
            return str(data)
        else:
            return data

    serializable_args = convert_paths_to_str(all_args)

    output_path = merged_model_dir / "reap_args.yaml"
    with open(output_path, "w") as f:
        yaml.dump(serializable_args, f, default_flow_style=False)
    logger.info(f"All arguments saved to {output_path}")


def main():
    (
        reap_args,
        model_args,
        ds_args,
        obs_args,
        cluster_args,
        kd_args,
        eval_args,
        merge_args,
    ) = parse_args()
    set_seed(reap_args.seed)
    results_dir = create_results_directory(model_args.model_name, ds_args.dataset_name)

    if cluster_args.singleton_super_experts and cluster_args.singleton_outlier_experts:
        raise ValueError(
            "Both 'singleton_super_experts' in clustering and 'perserve_super_experts' in merging cannot be set to True."
        )
    # get local patched model if req'd
    model_name = patched_model_map(model_args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # load model
    logger.info("Loading model into RAM (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": "cpu"},
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    model.target_device = target_device
    
    # PERMANENTLY move shared modules to GPU to save PCIe bandwidth
    logger.info("Moving shared modules (Embeddings/Norm) to GPU permanently...")
    model.model.embed_tokens.to(target_device)
    model.model.rotary_emb.to(target_device)
    
    logger.info(f"Model loaded into RAM. Target compute device: {model.target_device}")

    # record activations or load previously recorded activations
    logger.info(
        f"Running observer to collect activation data for model {model_args.model_name} on dataset {ds_args.dataset_name}."
    )
    observer_data = record_activations(
        model,
        tokenizer,
        reap_args,
        model_args,
        ds_args,
        obs_args,
        results_dir,
    )
    if reap_args.run_observer_only:
        logger.info(
            "Observer run completed. Exiting after collecting activation data since "
            "`run_observer_only` is set to True."
        )
        return

    # clustering
    logger.info("Start of clustering")
    num_clusters = cluster_args.num_clusters
    if num_clusters is None:
        if cluster_args.compression_ratio is None:
            raise ValueError(
                "Either num_clusters or compression_ratio must be set for clustering."
            )
        else:
            # Calculate num_clusters from compression_ratio
            if not merge_args.skip_first and not merge_args.skip_last:
                total_experts = len(
                    observer_data[next(iter(observer_data))]["expert_frequency"]
                )
                num_clusters = int(total_experts * (1 - cluster_args.compression_ratio))
            else:
                # If skipping first or last layer, adjust total_experts accordingly
                experts_per_layer = len(
                    observer_data[next(iter(observer_data))]["expert_frequency"]
                )
                layers = len(observer_data)
                total_experts = layers * experts_per_layer
                total_clusters = int(
                    total_experts * (1 - cluster_args.compression_ratio)
                )
                total_layers = len(observer_data)
                if merge_args.skip_first:
                    total_layers -= 1
                if merge_args.skip_last:
                    total_layers -= 1
                num_clusters = int(total_clusters / total_layers)
            logger.info(
                f"Calculated num_clusters: {num_clusters} from compression_ratio: {cluster_args.compression_ratio}"
            )
    cluster_labels = cluster(
        observer_data,
        num_clusters,
        cluster_args,
        obs_args.distance_measure,
        results_dir,
    )
    logger.info("Clustering completed.")

    # merging
    logging.info("Start of merging")
    merged_model_dir = get_model_dir(
        results_dir,
        num_clusters,
        cluster_labels,
        cluster_args,
        obs_args,
        merge_args,
    )
    if (
        merged_model_dir.exists()
        and list(merged_model_dir.glob("*.safetensors"))
        and not merge_args.overwrite_merged_model
    ):
        logger.info(
            f"Merged model files already exist in {merged_model_dir}. Skipping merging."
        )
    else:
        merge(
            model,
            cluster_labels,
            # num_clusters,
            observer_data,
            merge_args,
        )
        logger.info("Merging completed.")
        logger.info("Saving merged model...")
        merged_model_dir = save_merged_model(
            model,
            tokenizer,
            merged_model_dir,
            safe_serialization=True if not merge_args.save_as_tied_params else False,
        )
        logger.info(f"Merged model saved to {merged_model_dir}.")

        # save clustering results
        logger.info("Saving clustering results...")
        cluster_analysis_dir = merged_model_dir / "clusters"
        cluster_analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(cluster_analysis_dir / "clusters.pkl", "wb") as f:
            pickle.dump(cluster_labels, f)

        if reap_args.plot_clusters:
            logger.info("Plotting clusters analysis...")
            plot_cluster_analysis(
                cluster_labels,
                cluster_analysis_dir,
                merge_args.skip_first,
                merge_args.skip_last,
            )
        logger.info(
            f"Clustering results saved to {merged_model_dir / cluster_analysis_dir}"
        )

        # smoke test
        if reap_args.smoke_test:
            logger.info("Running smoke test on the merged model...")
            try:
                smoke_test(model, tokenizer)
            except Exception as e:
                logger.error(f"Smoke test failed: {e}")
                pass

        dump_args_to_yaml(
            merged_model_dir,
            reap_args,
            model_args,
            ds_args,
            obs_args,
            cluster_args,
            kd_args,
            eval_args,
            merge_args,
        )

        if model_name == "artifacts/models/GLM-4.5-Air":
            # move modelling file
            source_file = pathlib.Path(model_name) / "modeling_glm4_moe.py"
            target_file = merged_model_dir / "modeling_glm4_moe.py"
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                logger.info(f"Copied modeling_glm4_moe.py to {merged_model_dir}")
            else:
                raise RuntimeError(
                    f"Source file {source_file} does not exist. Cannot copy to {target_file}."
                )

    # eval
    if reap_args.do_eval:
        remove_hook_from_module(model, recurse=True)
        model.to("cpu")
        del model
        del observer_data
        del cluster_labels
        torch.cuda.empty_cache()
        gc.collect()
        model_args.model_name = merged_model_dir
        run_evaluate(model_args, merged_model_dir / "eval", eval_args, reap_args.seed)


if __name__ == "__main__":
    main()
