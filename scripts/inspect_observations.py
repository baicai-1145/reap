#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

import torch


def _tensor_stats(x: torch.Tensor) -> dict[str, Any]:
    x_cpu = x.detach().to("cpu")
    nz = int((x_cpu != 0).sum().item())
    return {
        "shape": tuple(x_cpu.shape),
        "dtype": str(x_cpu.dtype),
        "min": float(x_cpu.min().item()) if x_cpu.numel() else None,
        "max": float(x_cpu.max().item()) if x_cpu.numel() else None,
        "mean": float(x_cpu.float().mean().item()) if x_cpu.numel() else None,
        "nonzero": nz,
    }


def _print_tensor(name: str, x: torch.Tensor) -> None:
    s = _tensor_stats(x)
    print(
        f"- {name}: shape={s['shape']} dtype={s['dtype']} "
        f"min={s['min']:.4g} max={s['max']:.4g} mean={s['mean']:.4g} nz={s['nonzero']}"
    )


def _as_int(v: Any) -> int | None:
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return int(v.item())
    if isinstance(v, int):
        return v
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Inspect REAP observation .pt files (per-layer tensor stats)."
    )
    ap.add_argument(
        "path",
        type=pathlib.Path,
        help="Path to observations_*.pt produced by record_activations.",
    )
    ap.add_argument(
        "--layers",
        type=str,
        default="head:3",
        help=(
            "Which layers to print. Examples: 'all', 'head:3', 'tail:3', '0,1,2,10'. "
            "Default: head:3"
        ),
    )
    ap.add_argument(
        "--keys",
        type=str,
        default="total_tokens,expert_frequency,weighted_expert_frequency_sum,ean_sum,weighted_ean_sum,reap,max_activations",
        help="Comma-separated keys to print per layer.",
    )
    args = ap.parse_args()

    if not args.path.exists():
        print(f"error: file not found: {args.path}", file=sys.stderr)
        return 2

    obs = torch.load(args.path, map_location="cpu", weights_only=False)
    if not isinstance(obs, dict) or not obs:
        print("error: unexpected file content (expected non-empty dict).", file=sys.stderr)
        return 2

    layers_all = sorted(obs.keys())
    if not all(isinstance(x, int) for x in layers_all):
        print(
            f"warning: layer keys are not all int: {set(type(x) for x in layers_all)}",
            file=sys.stderr,
        )

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]

    # Global reap quick check
    reap_max = 0.0
    reap_nz = 0
    reap_present_layers = 0
    for l in layers_all:
        v = obs[l].get("reap")
        if isinstance(v, torch.Tensor):
            reap_present_layers += 1
            if v.numel():
                reap_max = max(reap_max, float(v.max().item()))
                reap_nz += int((v != 0).sum().item())

    print(f"file: {args.path}")
    print(f"num_layers: {len(layers_all)}")
    print(f"example_keys(layer0): {list(obs[layers_all[0]].keys())}")
    print(f"global_reap_present_layers: {reap_present_layers}")
    print(f"global_reap_max: {reap_max:.6g}")
    print(f"global_reap_nonzero: {reap_nz}")

    # Select layers to print
    sel: list[int]
    spec = args.layers.strip().lower()
    if spec == "all":
        sel = layers_all
    elif spec.startswith("head:"):
        n = int(spec.split(":", 1)[1])
        sel = layers_all[:n]
    elif spec.startswith("tail:"):
        n = int(spec.split(":", 1)[1])
        sel = layers_all[-n:]
    else:
        sel = [int(x.strip()) for x in spec.split(",") if x.strip()]

    for l in sel:
        if l not in obs:
            print(f"\n== layer {l} ==\n- missing")
            continue
        layer = obs[l]
        print(f"\n== layer {l} ==")
        # Consistency check: sum(expert_frequency) / total_tokens ~= top_k
        tt = _as_int(layer.get("total_tokens"))
        ef = layer.get("expert_frequency")
        if tt is not None and isinstance(ef, torch.Tensor) and ef.numel():
            s = int(ef.sum().item())
            ratio = (s / tt) if tt else None
            print(f"- check: total_tokens={tt} sum(expert_frequency)={s} ratio={ratio}")
        for k in keys:
            v = layer.get(k)
            if isinstance(v, torch.Tensor):
                _print_tensor(k, v)
            else:
                print(f"- {k}: {type(v).__name__}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

