import torch
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


MODEL_ATTRS = {
    "Qwen3MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Qwen3NextForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Qwen3-Coder-30B-A3B-Instruct": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "NonUniformQwen3MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Llama4ForCausalLM": {
        "moe_block": "feed_forward",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "num_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "MixtralForCausalLM": {
        "moe_block": "block_sparse_moe",
        "gate_proj": "w3",
        "up_proj": "w1",
        "down_proj": "w2",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "DeepseekV2ForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Ernie4_5_MoEForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "moe_num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Ernie4_5_MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "moe_num_experts",
        "num_experts_per_tok": "moe_k",
    },
    "gpt-oss-20b": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Glm4MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
}


@dataclass(frozen=True)
class PruneMoESpec:
    moe_block: str
    experts: str
    router: str
    fused: bool


_AUTO_PRUNE_SPEC_CACHE: dict[str, PruneMoESpec] = {}


def _infer_prune_spec_from_layer(layer_module: Any) -> PruneMoESpec:
    # Find the MoE block within a decoder layer by scanning direct children first.
    moe_block_name = None
    moe_block = None
    for child_name, child in layer_module.named_children():
        if hasattr(child, "experts"):
            experts = getattr(child, "experts", None)
            if experts is None:
                continue
            try:
                nE = len(experts)
            except Exception:
                nE = None
            if nE is None or nE < 2:
                continue
            moe_block_name = child_name
            moe_block = child
            break
    if moe_block is None or moe_block_name is None:
        # Fall back to a shallow search (depth<=2) within the layer.
        for full_name, mod in layer_module.named_modules():
            if full_name == "":
                continue
            if hasattr(mod, "experts"):
                experts = getattr(mod, "experts", None)
                if experts is None:
                    continue
                try:
                    nE = len(experts)
                except Exception:
                    nE = None
                if nE is None or nE < 2:
                    continue
                # Only accept a 1-hop attribute path for now (keeps pruning logic simple).
                if "." in full_name:
                    continue
                moe_block_name = full_name
                moe_block = mod
                break
    if moe_block is None or moe_block_name is None:
        raise ValueError(
            "Unable to infer MoE block within decoder layer (no module with multi-expert `experts` found)."
        )

    # Experts attribute name (prefer `.experts`)
    experts_attr = None
    experts_obj = getattr(moe_block, "experts", None)
    if isinstance(experts_obj, torch.nn.ModuleList) and len(experts_obj) >= 2:
        experts_attr = "experts"
        num_experts = len(experts_obj)
        fused = False
    else:
        # Try to find a ModuleList child that looks like experts
        candidates = []
        for child_name, child in moe_block.named_children():
            if isinstance(child, torch.nn.ModuleList) and len(child) >= 2:
                candidates.append((child_name, child))
        if candidates:
            experts_attr, experts_obj = candidates[0]
            num_experts = len(experts_obj)
            fused = False
        else:
            # Fused/special MoE (not supported by generic pruning)
            fused = True
            experts_attr = "experts" if hasattr(moe_block, "experts") else "<unknown>"
            num_experts = None

    # Router attribute name (prefer `.gate`/`.router`)
    router_attr = None
    for cand in ("gate", "router"):
        r = getattr(moe_block, cand, None)
        if isinstance(r, torch.nn.Linear):
            router_attr = cand
            break
    if router_attr is None and not fused and num_experts is not None:
        # Best-effort: find a Linear child whose out_features equals num_experts
        for child_name, child in moe_block.named_children():
            if isinstance(child, torch.nn.Linear) and getattr(child, "out_features", None) == num_experts:
                router_attr = child_name
                break
    if router_attr is None:
        router_attr = "gate" if hasattr(moe_block, "gate") else "router" if hasattr(moe_block, "router") else "<unknown>"

    return PruneMoESpec(
        moe_block=str(moe_block_name),
        experts=str(experts_attr),
        router=str(router_attr),
        fused=bool(fused),
    )


def get_prune_moe_spec(model) -> PruneMoESpec:
    """
    Return a minimal MoE spec required for pruning (moe_block/experts/router/fused).
    Falls back to best-effort inference for unknown models.
    """
    model_cls = model.__class__.__name__
    cached = _AUTO_PRUNE_SPEC_CACHE.get(model_cls)
    if cached is not None:
        return cached

    attrs = MODEL_ATTRS.get(model_cls)
    if attrs is not None:
        spec = PruneMoESpec(
            moe_block=attrs["moe_block"],
            experts=attrs["experts"],
            router=attrs["router"],
            fused=bool(attrs.get("fused", False)),
        )
        _AUTO_PRUNE_SPEC_CACHE[model_cls] = spec
        return spec

    # Infer from first layer as a representative template.
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError(
            f"Auto MoE pruning spec inference failed for {model_cls}: expected model.model.layers."
        )
    layer0 = model.model.layers[0]
    spec = _infer_prune_spec_from_layer(layer0)
    _AUTO_PRUNE_SPEC_CACHE[model_cls] = spec
    logger.warning(
        "Inferred MoE pruning spec for unknown model %s: moe_block=%s experts=%s router=%s fused=%s",
        model_cls,
        spec.moe_block,
        spec.experts,
        spec.router,
        spec.fused,
    )
    return spec


def get_moe(model, layer):
    spec = get_prune_moe_spec(model)
    return getattr(model.model.layers[layer], spec.moe_block)


def assert_merge(model, merged_moe, cluster_label):
    model_attr = MODEL_ATTRS.get(model.__class__.__name__)
    assert hasattr(merged_moe, "experts"), (
        "The merged module must have an 'experts' attribute."
    )

    gate_proj = model_attr["gate_proj"]
    down_proj = model_attr["down_proj"]

    if model_attr["fused"]:
        for cluster_id in cluster_label.unique():
            expert_indices = torch.where(cluster_label == cluster_id)[0]
            dom_expert = expert_indices[0]
            for expert in expert_indices[1:]:
                assert torch.allclose(
                    getattr(merged_moe.experts, gate_proj)[dom_expert],
                    getattr(merged_moe.experts, gate_proj)[expert],
                ), f"Experts {expert_indices} are not merged correctly."
                assert torch.allclose(
                    getattr(merged_moe.experts, down_proj)[dom_expert],
                    getattr(merged_moe.experts, down_proj)[expert],
                ), f"Experts {expert_indices} are not merged correctly."
    else:
        up_proj = model_attr["up_proj"]
        for cluster_id in cluster_label.unique():
            expert_indices = torch.where(cluster_label == cluster_id)[0]
            dom_expert = expert_indices[0]
            for expert in expert_indices[1:]:
                assert (
                    getattr(merged_moe.experts[dom_expert], up_proj).weight
                    == getattr(merged_moe.experts[expert], up_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."
                assert (
                    getattr(merged_moe.experts[dom_expert], down_proj).weight
                    == getattr(merged_moe.experts[expert], down_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."
                assert (
                    getattr(merged_moe.experts[dom_expert], gate_proj).weight
                    == getattr(merged_moe.experts[expert], gate_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."


def patched_model_map(model: str):
    patched = False
    model_name = model

    if model == "deepseek-ai/DeepSeek-V2-Lite-Chat":
        patched = True
        model_name = "artifacts/models/DeepSeek-V2-Lite-Chat"

    # until hf version lands
    if model == "baidu/ERNIE-4.5-21B-A3B-PT":
        patched = True
        model_name = "artifacts/models/ERNIE-4.5-21B-A3B-PT"

    if model == "Qwen/NonUniformQwen3-30B-A3B":
        patched = True
        model_name = "artifacts/models/NonUniformQwen3-30B-A3B"

    if model == "zai-org/GLM-4.5-Air":
        patched = True
        model_name = "artifacts/models/GLM-4.5-Air"

    if model == "zai-org/GLM-4.5-Air-FP8":
        patched = True
        model_name = "artifacts/models/GLM-4.5-Air-FP8"

    if model == "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8":
        patched = True
        model_name = "artifacts/models/Qwen3-Coder-480B-A35B-Instruct-FP8"

    if patched:
        logger.info(f"Using patched model for {model} from: {model_name}")
    return model_name


def assert_tied_weights(model, clusters_labels):
    model_attrs = MODEL_ATTRS.get(model.__class__.__name__)
    for layer_idx in clusters_labels:
        clusters = clusters_labels[layer_idx]
        moe = get_moe(model, layer_idx)
        experts = getattr(moe, model_attrs["experts"])
        for cluster_idx in torch.unique(clusters):
            experts_in_cluster = torch.where(clusters == cluster_idx)[0].tolist()
            dom_expert = experts[experts_in_cluster[0]]
            for attr in ["up_proj", "down_proj", "gate_proj"]:
                for expert_idx in experts_in_cluster:
                    if expert_idx == dom_expert:
                        continue
                    expert = experts[expert_idx]
                    proj = getattr(expert, attr)
                    weight = proj.weight
                    dom_proj = getattr(dom_expert, attr)
                    dom_weight = dom_proj.weight
                    if not torch.allclose(weight, dom_weight):
                        print(
                            f"Weights for expert {expert_idx} in cluster {cluster_idx} for layer {layer_idx} and attr {attr} are not tied!"
                        )
                        print(f"Max diff: {torch.abs(weight - dom_weight).max()}")
                    # check adapters
                    for lora_adapter in ["lora_A", "lora_B"]:
                        if hasattr(proj, lora_adapter):
                            lora_weight = getattr(proj, lora_adapter).default.weight
                            dom_lora_weight = getattr(
                                dom_proj, lora_adapter
                            ).default.weight
                            if not torch.allclose(lora_weight, dom_lora_weight):
                                print(
                                    f"LoRA Weights for expert {expert_idx} in cluster {cluster_idx} for layer {layer_idx} and adapter {lora_adapter} are not tied!"
                                )
                                print(
                                    f"Max diff: {torch.abs(lora_weight - dom_lora_weight).max()}"
                                )

def get_super_expert_indices(observer_data, include_last_layers: bool = False):
    logger.info("Identifying super experts to preserve...")
    quantile = 99.5
    times = 10
    all_max_activations = [layer['max_activations'] for layer in observer_data.values()]
    num_layers = len(all_max_activations)
    all_max_activations = torch.cat(all_max_activations).flatten()
    percentile_threshold = torch.quantile(all_max_activations, quantile / 100.0).item()
    abs_threshold = all_max_activations.max().item() / times
    final_threshold = max(percentile_threshold, abs_threshold)
    # reshape back into per layer data
    all_max_activations = all_max_activations.reshape(num_layers, -1)
    super_experts_mask = all_max_activations > final_threshold
    if not include_last_layers:
        # only consider first 75% of layers for super experts
        logger.info(
            "Only considering first 75% of layers for super expert "
            "identification since perserve_outliers is False"
        )
        num_layers = int(num_layers * 0.75)
        super_experts_mask[num_layers:, :] = False
    super_expert_idx = torch.argwhere(super_experts_mask)
    logger.info(f"Identified {super_experts_mask.sum().item()} super experts with threshold: {final_threshold:.4f}")
    return super_expert_idx

def register_llama_with_vllm():
    from vllm.model_executor.models import ModelRegistry
    print("Registering Llama4ForCausalLM with vLLM")
    ModelRegistry.register_model("Llama4ForCausalLM", "vllm.model_executor.models.llama4:Llama4ForCausalLM")
