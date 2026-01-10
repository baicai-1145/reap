from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
import gc
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from dataclasses import dataclass
import logging
import pathlib
from functools import reduce

from reap.metrics import (
    ttm_online,
    get_routed_characteristic_activation,
    ca_dist_online,
    OnlineStatsTracker,
    get_distance_fn,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTransformerObserverHookConfig:
    state_attr_name: str = "hook_state"
    hook_attr_name: str = "hooks"
    module_name_to_hook_regex: Optional[str] = None
    module_class_name_to_hook_regex: Optional[str] = None


class BaseTransformerObserver(ABC):
    def __init__(
        self,
        model,
        hook_config: Optional[BaseTransformerObserverHookConfig] = None,
    ):
        self.model = model
        self.hook_config = hook_config
        self.hooks = []
        self.state: dict[Any, Any] = {}
        self._hook_model()
        logger.info(
            "%s initialized for %s.",
            self.__class__.__name__,
            self.model.__class__.__name__,
        )

    @abstractmethod
    def _hook_factory(self, module: nn.Module, layer_number: int) -> callable:
        """
        Factory method to create a hook function for the given module.
        This method should be implemented by subclasses to define how the
        hook function should behave.
        """
        raise NotImplementedError("Subclasses must implement _hook_factory method.")

    def report_state(self) -> dict[str, Any]:
        """
        Method to report the current state of the observer. Can be overridden to inject
        custom behaviours.
        """
        return self.state

    def close_hooks(self):
        """Close all hooks registered to the model."""
        self.reset()  # Reset the state before closing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.debug("All hooks closed for %s.", self.model.__class__.__name__)

    def reset(self):
        """Reset the observer state."""
        del self.state
        gc.collect()
        self.state = {}
        logger.debug("Observer state reset for %s.", self.model.__class__.__name__)

    def save_state(self, file_path: str | pathlib.Path):
        self._move_state_tensors_to_cpu()
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = self.report_state()
        with open(file_path, "wb") as f:
            torch.save(state_dict, f)
        logger.info("State saved to %s", file_path)

    def _move_state_tensors_to_cpu(self):
        """
        Move all tensors in the state dictionary to CPU.
        This is useful before saving the state to avoid GPU memory issues.
        """
        for layer_number, layer_state in self.state.items():
            for key, value in layer_state.items():
                if isinstance(value, torch.Tensor):
                    self.state[layer_number][key] = value.cpu()

    def _validate_hook_config(self):
        if self.hook_config is None:
            raise ValueError("hook_config must be provided.")
        if (
            self.hook_config.module_name_to_hook_regex is None
            and self.hook_config.module_class_name_to_hook_regex is None
        ):
            raise ValueError(
                "At least one of 'module_name_to_hook_regex' or "
                "'module_class_name_to_hook_regex' must be provided in the hook config."
            )
        if (
            self.hook_config.module_name_to_hook_regex is not None
            and self.hook_config.module_class_name_to_hook_regex is not None
        ):
            logger.warning(
                "Both 'module_name_to_hook_regex' and 'module_class_name_to_hook_regex' are "
                "provided. Both conditions must be satisfied to hook the module."
            )

    def _hook_model(self):
        self._validate_hook_config()

        def _infer_layer_number(module_name: str) -> int | None:
            # Prefer explicit "layers.<idx>" patterns.
            m = re.search(r"(?:^|\\.)layers\\.(\\d+)(?:\\.|$)", module_name)
            if m:
                return int(m.group(1))
            # Fall back to first integer (best-effort).
            m = re.search(r"\\d+", module_name)
            if m:
                return int(m.group(0))
            return None

        for name, module in self.model.named_modules():
            name_ok = True
            cls_ok = True
            if self.hook_config.module_name_to_hook_regex:
                name_ok = re.search(self.hook_config.module_name_to_hook_regex, name) is not None
            if self.hook_config.module_class_name_to_hook_regex:
                cls_ok = re.search(
                    self.hook_config.module_class_name_to_hook_regex,
                    module.__class__.__name__,
                ) is not None
            hook_module = name_ok and cls_ok
            if hook_module:
                layer_number = _infer_layer_number(name)
                if layer_number is None:
                    logger.warning(
                        "Matched module '%s' but could not infer layer number; skipping hook.",
                        name,
                    )
                    continue
                hook_fn = self._hook_factory(module, layer_number)
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)
                logger.info("Hooked module: %s at layer %d", name, layer_number)
        if len(self.hooks) == 0:
            raise ValueError(
                "No modules matched the provided hook configuration. "
                "Check your hook configuration settings."
            )

    @classmethod
    def _get_registry_for_cls(cls) -> dict[str, type[BaseTransformerObserver]]:
        """Helper to get the registry from the specific class 'cls'."""
        if not hasattr(cls, "_architecture_registry") or not isinstance(
            cls._architecture_registry, dict
        ):
            raise AttributeError(
                f"Class {cls.__name__} must define its own "
                "`_architecture_registry: dict[str, type] = {{}}` "
                f"to use the common registration/creation methods."
            )
        return cls._architecture_registry

    @classmethod
    def register_implementation(cls, *arch_names: str):
        """
        Class method decorator to register a concrete observer implementation.
        'cls' is the class on which this decorator's factory is called (e.g.,
        MoEExpertObserver) 'sub_cls' is the class being decorated
        (e.g., Llama4MoEExpertObserver).
        """

        def decorator(sub_cls: type[BaseTransformerObserver]):
            registry = cls._get_registry_for_cls()

            for name in arch_names:
                if name in registry:
                    raise RuntimeError(
                        f"Architecture {name} already registered with "
                        f"{registry[name].__name__} for {cls.__name__}."
                    )
                registry[name] = sub_cls
            return sub_cls

        return decorator

    @classmethod
    def create_from_registry(
        cls,
        model: nn.Module,
        hook_config: Optional[BaseTransformerObserverHookConfig] = None,
        return_rank_0_only: bool = True,
        **kwargs: Any,
    ) -> BaseTransformerObserver:
        registry = cls._get_registry_for_cls()
        model_cls_name = model.__class__.__name__

        specific_observer_cls = registry.get(model_cls_name)

        if specific_observer_cls:
            return specific_observer_cls(
                model,
                hook_config=hook_config,
                return_rank_0_only=return_rank_0_only,
                **kwargs,
            )
        else:
            raise ValueError(
                "Unsupported architecture for "
                f"{cls.__name__}: {model_cls_name}. "
                "Registered architectures in "
                f"{cls.__name__}._architecture_registry: "
                f"{list(registry.keys())}"
            )


# --- MoE Transformer Observer ---------------------------------------------------------


@dataclass
class MoETransformerObserverConfig(BaseTransformerObserverHookConfig):
    num_experts_attr_name: str = "num_experts"
    top_k_attr_name: str = "top_k"
    fused_experts: bool = False
    distance_measure: str = "angular"
    renormalize_router_weights: bool = False
    record_pruning_metrics_only: bool = False
    # Best-effort fallbacks for unknown model implementations.
    num_experts_attr_candidates: tuple[str, ...] = (
        "num_experts",
        "num_local_experts",
        "experts_per_rank",
        "config.n_routed_experts",
        "experts",
    )
    top_k_attr_candidates: tuple[str, ...] = (
        "top_k",
        "num_experts_per_tok",
        "k",
        "config.num_experts_per_tok",
        "config.num_experts_per_tok",
    )


class MoETransformerObserver(BaseTransformerObserver):
    """MoE Transformer Observer for all methods including both pruning and merging."""

    def _resolve_int_attr(self, module: nn.Module, primary: str, fallbacks: tuple[str, ...]) -> int | None:
        def _resolve_path(obj: Any, path: str) -> Any | None:
            cur = obj
            for part in path.split("."):
                if cur is None:
                    return None
                if not hasattr(cur, part):
                    return None
                cur = getattr(cur, part)
            return cur

        def _coerce_int(v: Any) -> int | None:
            if v is None:
                return None
            if isinstance(v, bool):
                return int(v)
            if isinstance(v, int):
                return int(v)
            if torch.is_tensor(v) and v.numel() == 1:
                return int(v.item())
            if isinstance(v, nn.ModuleList):
                return len(v)
            if hasattr(v, "__len__") and not isinstance(v, (str, bytes, dict)):
                try:
                    return int(len(v))  # type: ignore[arg-type]
                except Exception:
                    return None
            try:
                return int(v)
            except Exception:
                return None

        # Try primary then fallbacks on module
        for p in (primary, *fallbacks):
            v = _resolve_path(module, p)
            out = _coerce_int(v)
            if out is not None:
                return out

        # Fallback to model.config if available
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            for p in (primary, *fallbacks):
                v = _resolve_path(cfg, p.replace("config.", ""))
                out = _coerce_int(v)
                if out is not None:
                    return out
        return None

    def report_state(self) -> dict[str, Any]:
        """
        Method to report the current state of the observer. Can be overridden to inject
        custom behaviours.
        """
        return {
            layer_num: {
                k: v.mean if isinstance(v, OnlineStatsTracker) else v
                for k, v in layer_state.items()
            }
            for layer_num, layer_state in self.state.items()
        }

    def _initialize_state(self, output: torch.Tensor, num_experts: int):
        # get device and shape info
        output_hidden_states = output[0]
        device = "cpu"
        hidden_dim = output_hidden_states.shape[-1]
        layer_state = {}

        # unnormalized states (counts)
        layer_state["total_tokens"] = torch.tensor(0, device=device, dtype=torch.long)
        layer_state["expert_frequency"] = torch.zeros(
            num_experts, device=device, dtype=torch.long
        )
        layer_state["pairwise_expert_frequency"] = torch.zeros(
            num_experts, num_experts, dtype=torch.long, device=device
        )

        if not self.hook_config.record_pruning_metrics_only:
            # per routed token normalized states
            layer_state["ttm_similarity_matrix"] = OnlineStatsTracker(
                shape=(num_experts, num_experts),
                count_shape=(num_experts, num_experts),
                device=device,
                dtype=torch.float32,
            )
            layer_state["routed_characteristic_activation"] = OnlineStatsTracker(
                shape=(num_experts, hidden_dim),
                count_shape=(num_experts, hidden_dim),
                device=device,
                dtype=torch.float32,
            )
            # HC-SMoE
            layer_state["characteristic_activation"] = OnlineStatsTracker(
                shape=(num_experts, hidden_dim),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )
            # SubMoE
            layer_state["online_characteristic_activation_dist"] = OnlineStatsTracker(
                shape=(num_experts, num_experts),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )
            # per total token normalized states -> MC-SMoE
            layer_state["router_logit_similiarity"] = OnlineStatsTracker(
                shape=(num_experts, num_experts),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )

        # Expert Activation Norm
        layer_state["ean_sum"] = torch.zeros(
            (num_experts,), device=device, dtype=torch.float64, requires_grad=False
        )
        layer_state["weighted_ean_sum"] = torch.zeros(
            (num_experts,), device=device, dtype=torch.float64, requires_grad=False
        )
        layer_state["ean_mean"] = OnlineStatsTracker(
            shape=(num_experts,),
            count_shape=(num_experts,),
            device=device,
            dtype=torch.float32,
        )
        layer_state["reap"] = OnlineStatsTracker(
            shape=(num_experts,),
            count_shape=(num_experts,),
            device=device,
            dtype=torch.float32,
        )

        # Weighted frequency
        layer_state["weighted_expert_frequency_sum"] = torch.zeros(
            (num_experts,), device=device, dtype=torch.float64, requires_grad=False
        )

        # super experts
        layer_state["max_activations"] = torch.zeros(
            (num_experts,), device=device, dtype=torch.float32, requires_grad=False
        )

        return layer_state

    def _hook_factory(self, module: nn.Module, layer_number: int) -> callable:
        distance_fn = get_distance_fn("cosine") # always use cosine for online dist. metrics
        num_experts = self._resolve_int_attr(
            module,
            self.hook_config.num_experts_attr_name,
            getattr(self.hook_config, "num_experts_attr_candidates", tuple()),
        )
        top_k = self._resolve_int_attr(
            module,
            self.hook_config.top_k_attr_name,
            getattr(self.hook_config, "top_k_attr_candidates", tuple()),
        )
        if num_experts is None or top_k is None:
            raise ValueError(
                f"Module {module.__class__.__name__} at layer {layer_number} "
                "does not have expected 'num_experts' or 'top_k' attributes. Check "
                "HookConfig settings."
            )

        @torch.no_grad()
        def _hook_fn(module, args, output):
            if not len(output) >= 2:
                raise ValueError(
                    f"Expected output of module {module.__class__.__name__} at layer "
                    f"{layer_number} to be a tuple of at least length 2, got {len(output)}."
                )
            input = args[0]  # (batch_size, seq_len, hidden_dim)
            device = input.device
            batch_size, sequence_length, hidden_dim = input.shape
            flat_input = input.view(-1, hidden_dim)  # total_seq_len, hidden
            
            num_tokens = batch_size * sequence_length
            num_tokens_tensor = torch.tensor(num_tokens, device="cpu", dtype=torch.long)

            if layer_number not in self.state:
                self.state[layer_number] = self._initialize_state(output, num_experts)

            # --- PRUNE/MERGE SALIENCY CRITERIA PREP ---------------------------
            *_, router_logits = output  # (total_tokens, num_experts)
            _, selected_experts = torch.topk(router_logits, top_k, dim=-1)
            
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float).to(device)
            if self.hook_config.renormalize_router_weights:
                topk_weights = torch.gather(routing_weights, 1, selected_experts)
                routing_weights = routing_weights / topk_weights.sum(dim=-1, keepdim=True)
                routing_weights = torch.clamp(routing_weights, min=torch.finfo(routing_weights.dtype).eps)

            expert_frequency = torch.bincount(
                selected_experts.view(-1), minlength=num_experts
            ).to(device)
            pairwise_expert_frequency = (expert_frequency.unsqueeze(0) + expert_frequency.unsqueeze(1)).to(device)

            self.state[layer_number]["total_tokens"] += num_tokens_tensor
            self.state[layer_number]["expert_frequency"] += expert_frequency.to("cpu", torch.long)
            self.state[layer_number]["pairwise_expert_frequency"] += pairwise_expert_frequency.to("cpu", torch.long)

            # --- PROCESS EXPERTS ----------------------------------------------
            if self.hook_config.record_pruning_metrics_only and not self.hook_config.fused_experts:
                # OPTIMIZED PATH: Stream metrics and only compute for active tokens to save memory and time
                
                # Pre-calculate token indices for all experts to avoid 512 torch.where calls
                # selected_experts shape: (total_tokens, top_k)
                total_tokens = selected_experts.shape[0]
                expert_to_token_indices = [[] for _ in range(num_experts)]
                
                # Faster way to group tokens by expert
                flat_selected = selected_experts.flatten()
                token_indices_repeated = torch.arange(total_tokens, device=device).unsqueeze(1).expand(-1, top_k).flatten()
                
                # Use a loop but it's faster than 512 where() calls on a large tensor
                # Actually, for 512 experts, let's just use a more efficient grouping if possible
                # But even better: just iterate and use the mask
                
                layer_ean_means = torch.zeros(num_experts, device=device)
                layer_reaps = torch.zeros(num_experts, device=device)
                
                for i in range(num_experts):
                    # Only find tokens assigned to expert i
                    mask = (selected_experts == i).any(dim=-1)
                    if not mask.any():
                        continue
                    
                    token_indices = mask.nonzero(as_tuple=True)[0]
                    expert = module.experts[i]
                    
                    # Only compute expert output for the tokens it's assigned to
                    expert_input = flat_input[token_indices]
                    expert_out = expert(expert_input) # (num_active_tokens, hidden_dim)
                    
                    # active_router_weights shape: (num_active_tokens,)
                    active_router_weights = routing_weights[token_indices, i]
                    ean_norm = torch.linalg.norm(expert_out, dim=-1)
                    
                    # Store results in local GPU tensors
                    self.state[layer_number]["ean_sum"][i] += ean_norm.sum().cpu()
                    self.state[layer_number]["weighted_ean_sum"][i] += (ean_norm * active_router_weights).sum().cpu()
                    self.state[layer_number]["weighted_expert_frequency_sum"][i] += active_router_weights.sum().cpu()
                    
                    layer_ean_means[i] = ean_norm.mean()
                    layer_reaps[i] = (ean_norm * active_router_weights).mean()

                    # super experts
                    selected_activations_max = expert_out.max().cpu()
                    if selected_activations_max > self.state[layer_number]["max_activations"][i]:
                        self.state[layer_number]["max_activations"][i] = selected_activations_max
                    
                    del expert_out, ean_norm, expert_input
                
                # Single update call per layer instead of 512
                self.state[layer_number]["ean_mean"].update(layer_ean_means.cpu(), expert_frequency.cpu())
                self.state[layer_number]["reap"].update(layer_reaps.cpu(), expert_frequency.cpu())
                
                # Cleanup
                del routing_weights, selected_experts, expert_frequency, pairwise_expert_frequency
                del layer_ean_means, layer_reaps
                gc.collect()
                return # End of optimized _hook_fn

            # LEGACY PATH: Allocates large activations tensor (used for merging or fused experts)
            activations = torch.zeros((num_experts, *flat_input.shape), device=device)
            if self.hook_config.fused_experts:
                _, router_scores = output  # (num_experts, total_tokens)
                # ... rest of fused logic ...

            del flat_input
            num_tokens = batch_size * sequence_length
            num_tokens = torch.tensor(num_tokens, device="cpu", dtype=torch.long)

            # --- PRUNE/MERGE SALIENCY CRITERIA --------------------------------
            # expert frequency
            expert_frequency = torch.bincount(
                selected_experts.view(-1), minlength=num_experts
            ).to(device)
            pairwise_expert_frequency = expert_frequency.unsqueeze(
                0
            ) + expert_frequency.unsqueeze(1)
            pairwise_expert_frequency = pairwise_expert_frequency.to(device)

            self.state[layer_number]["total_tokens"] += num_tokens
            self.state[layer_number]["expert_frequency"] += expert_frequency.to(
                "cpu", torch.long
            )
            self.state[layer_number]["pairwise_expert_frequency"] += (
                pairwise_expert_frequency.to("cpu", torch.long)
            )

            # Merging critera
            if not self.hook_config.record_pruning_metrics_only:
                ttm_similarity_matrix = ttm_online(
                    activations,
                    selected_experts,
                    distance_callable=distance_fn,
                    num_experts=num_experts,
                    pairwise_expert_frequency=pairwise_expert_frequency,
                )

                # ttm_similarity_matrix with pairwise frequency counts
                self.state[layer_number]["ttm_similarity_matrix"].update(
                    ttm_similarity_matrix, pairwise_expert_frequency
                )
                del ttm_similarity_matrix

                routed_characteristic_activation = get_routed_characteristic_activation(
                    activations,
                    selected_experts,
                    expert_frequency,
                    device,
                    hidden_dim,
                    num_experts,
                )

                # routed_characteristic_activation with expert frequency counts
                expert_freq_expanded = expert_frequency.unsqueeze(-1).expand(
                    (-1, hidden_dim)
                )
                self.state[layer_number]["routed_characteristic_activation"].update(
                    routed_characteristic_activation, expert_freq_expanded
                )
                del expert_freq_expanded, routed_characteristic_activation

                online_characteristic_activation_dist = ca_dist_online(
                    activations,
                    distance_callable=distance_fn,
                ).to(device="cpu")

                # online_characteristic_activation_dist with expert frequency counts
                self.state[layer_number]["online_characteristic_activation_dist"].update(
                    online_characteristic_activation_dist, num_tokens
                )
                del online_characteristic_activation_dist

                # router logit similarity -> must align with distance_fn shape expectations
                # dim 0 "batch" dim, dims 1,2 expert pairwise, dim 3 token logits
                router_logit_sim = (
                    distance_fn(
                        router_logits.permute(1, 0).view(
                            1, num_experts, 1, -1
                        ),  # 1, num_experts, 1, logits
                        router_logits.permute(1, 0).view(
                            1, 1, num_experts, -1
                        ),  # 1, 1, num_experts, logits
                    )
                    .squeeze()
                    .to(device="cpu")
                )  # yields (num_experts, num_experts)

                # router_logit_similarity with total tokens count
                self.state[layer_number]["router_logit_similiarity"].update(
                    router_logit_sim, num_tokens
                )
                del router_logit_sim

                # characteristic_activation with total tokens count
                self.state[layer_number]["characteristic_activation"].update(
                    activations.mean(dim=1), num_tokens
                )

            # Pruning criteria
            ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
            ean_mean = torch.zeros(num_experts, device=device, dtype=torch.float32)
            weighted_ean_sum = torch.zeros(
                num_experts, device=device, dtype=torch.float64
            )
            reap = torch.zeros(
                num_experts, device=device, dtype=torch.float32
            )
            weighted_expert_frequency_sum = torch.zeros(
                num_experts, device=device, dtype=torch.float64
            )
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float).to(
                device
            )  # tok, num_experts
            prior_max_activations = self.state[layer_number]["max_activations"]
            # renormalize
            if self.hook_config.renormalize_router_weights:
                topk_weights = torch.gather(
                    routing_weights,
                    1,
                    selected_experts,
                )  # (total_tokens, top_k)
                routing_weights = routing_weights / topk_weights.sum(
                    dim=-1, keepdim=True
                )
                routing_weights = torch.clamp(
                    routing_weights, min=torch.finfo(routing_weights.dtype).eps
                )
                # routing_weights = routing_weights.to(device)

            for i in range(num_experts):
                active_mask = (selected_experts == i).any(dim=-1).to(device)
                if not active_mask.any():
                    continue
                active_router_weights = routing_weights[active_mask, i]
                ean_norm = torch.linalg.norm(activations[i, active_mask, :], dim=-1)
                ean_sum[i] = ean_norm.sum().to(device)
                ean_mean[i] = ean_norm.mean().to(device)
                weighted_expert_frequency_sum[i] = active_router_weights.sum().to(
                    device
                )
                weighted_ean_sum[i] = (
                    (ean_norm * active_router_weights).sum().to(device)
                )
                reap[i] = (
                    (ean_norm * active_router_weights).mean().to(device)
                )

                # super experts
                selected_activations = activations[i, active_mask, :]
                selected_activations_max = selected_activations.max().to(device="cpu")
                if selected_activations_max > prior_max_activations[i]:
                    self.state[layer_number]["max_activations"][i] = (
                        selected_activations_max
                    )
                    prior_max_activations[i] = selected_activations_max

            # ean
            self.state[layer_number]["ean_sum"] += ean_sum.to(device="cpu")
            self.state[layer_number]["ean_mean"].update(ean_mean, expert_frequency)
            self.state[layer_number]["weighted_ean_sum"] += weighted_ean_sum.to(
                device="cpu"
            )
            if reap.sum() == 0:
                print("debug")
            self.state[layer_number]["reap"].update(
                reap, expert_frequency
            )

            # weighted_expert_frequency_sum
            
            self.state[layer_number]["weighted_expert_frequency_sum"] += (
                weighted_expert_frequency_sum.to(device="cpu")
            )

            # --- CLEAN UP -------------------------------------------------------------
            del (
                activations,
                selected_experts,
                router_logits,
                expert_frequency,
                pairwise_expert_frequency,
                prior_max_activations,
            )
            gc.collect()

        return _hook_fn


# --- Concrete Config Implementations ----


@dataclass
class Qwen3MoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "Qwen3MoeSparseMoeBlock"


@dataclass
class Qwen3NextMoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "Qwen3NextSparseMoeBlock"


@dataclass
class Llama4MoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "Llama4TextMoe"
    fused_experts: bool = True  # Llama4 uses fused experts


@dataclass
class MixtralMoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "MixtralSparseMoeBlock"


@dataclass
class DeepSeekMoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "DeepseekV2MoE"
    num_experts_attr_name: str = "experts_per_rank"  # only for ep=1!
    top_k_attr_name: str = "num_experts_per_tok"
    fused_experts: bool = False


@dataclass
class Ernie4_5MoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "Ernie4_5_MoeMLP"
    num_experts_attr_name: str = "num_local_experts"
    top_k_attr_name: str = "k"

    # hf in tree implementation below:
    # module_class_name_to_hook_regex: Optional[str] = "Ernie4_5_MoESparseMoeBlock"
    # num_experts_attr_name: str = "num_experts"
    # top_k_attr_name: str = "top_k"
    fused_experts: bool = False


@dataclass
class Glm44MoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "Glm4MoeMoE"
    num_experts_attr_name: str = "config.n_routed_experts"
    top_k_attr_name: str = "config.num_experts_per_tok"
    fused_experts: bool = False


OBSERVER_CONFIG_REGISTRY = {
    "Qwen3MoeForCausalLM": Qwen3MoEObserverHookConfig,
    "NonUniformQwen3MoeForCausalLM": Qwen3MoEObserverHookConfig,
    "Qwen3NextForCausalLM": Qwen3NextMoEObserverHookConfig,
    "Llama4ForCausalLM": Llama4MoEObserverHookConfig,
    "MixtralForCausalLM": MixtralMoEObserverHookConfig,
    "DeepseekV2ForCausalLM": DeepSeekMoEObserverHookConfig,
    "Ernie4_5_MoEForCausalLM": Ernie4_5MoEObserverHookConfig,
    "Ernie4_5_MoeForCausalLM": Ernie4_5MoEObserverHookConfig,
    "Glm4MoeForCausalLM": Glm44MoEObserverHookConfig,
}


def infer_moe_observer_hook_config(
    model: nn.Module,
    *,
    distance_measure: str = "cosine",
    renormalize_router_weights: bool = False,
    record_pruning_metrics_only: bool = False,
) -> MoETransformerObserverConfig:
    """
    Best-effort MoE auto-detection to avoid manual registry edits.

    This is intended for typical HF MoE blocks that expose an `experts` ModuleList and
    a router (often named `gate`/`router`) and whose forward outputs include router logits.
    If the discovered modules do not match the expected output signature, observation will
    fail with a clear error at runtime.
    """
    moe_class_names: list[str] = []
    fused_hint = False

    for _name, mod in model.named_modules():
        if not hasattr(mod, "experts"):
            continue
        experts = getattr(mod, "experts", None)
        if experts is None:
            continue
        # Require there to be multiple experts (avoid false positives)
        nE = None
        if isinstance(experts, nn.ModuleList):
            nE = len(experts)
        elif hasattr(experts, "__len__"):
            try:
                nE = len(experts)
            except Exception:
                nE = None
        if nE is None or nE < 2:
            continue

        if not isinstance(experts, nn.ModuleList):
            fused_hint = True
        moe_class_names.append(mod.__class__.__name__)

    if not moe_class_names:
        raise ValueError(
            "Auto MoE detection failed: no modules with a multi-expert `experts` attribute were found. "
            "You may still need to add a custom HookConfig in src/reap/observer.py."
        )

    # Build a class-name regex that matches all detected MoE block classes.
    uniq = sorted(set(moe_class_names))
    if len(uniq) == 1:
        cls_pat = f"^{re.escape(uniq[0])}$"
    else:
        cls_pat = "^(" + "|".join(re.escape(x) for x in uniq) + ")$"

    if fused_hint:
        logger.warning(
            "Auto MoE detection found non-ModuleList experts (fused/specialized MoE). "
            "Some pruning modes may not be supported without custom adapters."
        )

    cfg = MoETransformerObserverConfig(
        module_class_name_to_hook_regex=cls_pat,
        distance_measure=distance_measure,
        renormalize_router_weights=renormalize_router_weights,
        record_pruning_metrics_only=record_pruning_metrics_only,
    )
    return cfg
