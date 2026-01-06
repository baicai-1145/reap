#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import sys
from collections import defaultdict
from typing import Any


def _sanitize_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
    return name[:180] if name else "unknown"


def _jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    # Fallback for numpy / other scalars
    try:
        import numpy as np  # type: ignore

        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
    except Exception:
        pass
    return str(x)


def _extract_domain(example: dict[str, Any], domain_field: str | None) -> str:
    if domain_field and domain_field in example:
        v = example.get(domain_field)
        if v is None:
            return "unknown"
        return str(v)

    # Common patterns
    for k in ("domain", "category", "task", "subset", "source", "dataset"):
        if k in example and example.get(k) is not None:
            return str(example[k])

    meta = example.get("meta")
    if isinstance(meta, dict):
        for k in ("domain", "category", "task", "subset", "source"):
            if k in meta and meta.get(k) is not None:
                return str(meta[k])

    return "unknown"


def _reservoir_add(
    rng: random.Random,
    reservoir: list[dict[str, Any]],
    seen_count: int,
    k: int,
    item: dict[str, Any],
) -> None:
    """
    Classic reservoir sampling (Algorithm R):
    after processing N items, the reservoir of size k is a uniform sample.
    """
    if k <= 0:
        return
    if seen_count <= k:
        reservoir.append(item)
        return
    j = rng.randrange(seen_count)
    if j < k:
        reservoir[j] = item


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Stream OpenThoughts3-1.2M (or any HF dataset) and reservoir-sample "
            "K examples per domain into JSONL files (one pass, no full download)."
        )
    )
    ap.add_argument(
        "--dataset",
        default="open-thoughts/OpenThoughts3-1.2M",
        help="HuggingFace dataset name.",
    )
    ap.add_argument(
        "--split",
        default="train",
        help="Dataset split to stream (e.g. train).",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="Optional dataset config name (if the dataset has multiple configs).",
    )
    ap.add_argument(
        "--domain-field",
        default=None,
        help=(
            "Column name to use as domain. If omitted, tries common keys like "
            "domain/category/task/subset/source."
        ),
    )
    ap.add_argument(
        "--samples-per-domain",
        type=int,
        default=1024,
        help="Number of samples to keep for each domain.",
    )
    ap.add_argument(
        "--max-domains",
        type=int,
        default=0,
        help=(
            "If >0, only keep the first N discovered domains (others are ignored). "
            "Useful if the dataset has many tiny domains."
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reservoir sampling.",
    )
    ap.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/openthoughts3_samples_by_domain"),
        help="Output directory for *.jsonl and manifest.json.",
    )
    ap.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help=(
            "If >0, stop after streaming this many total examples (approximate sampling). "
            "If 0, streams the full split for exact reservoir sampling."
        ),
    )
    ap.add_argument(
        "--write-one-file",
        action="store_true",
        help="Write a single combined jsonl with an added '__domain' field.",
    )
    args = ap.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        print(
            "error: missing dependency 'datasets'. Install it (already in this repo's deps).",
            file=sys.stderr,
        )
        print(str(e), file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)

    load_kwargs: dict[str, Any] = {"streaming": True}
    if args.config:
        load_kwargs["name"] = args.config

    ds = load_dataset(args.dataset, **load_kwargs)[args.split]

    rng = random.Random(args.seed)

    reservoirs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    domains_seen_order: list[str] = []

    total = 0
    for ex in ds:
        total += 1
        if args.max_examples and total > args.max_examples:
            break

        domain = _extract_domain(ex, args.domain_field)
        if domain not in reservoirs and args.max_domains and len(domains_seen_order) >= args.max_domains:
            continue
        if domain not in reservoirs:
            domains_seen_order.append(domain)

        seen[domain] += 1
        item = _jsonable(dict(ex))
        _reservoir_add(
            rng=rng,
            reservoir=reservoirs[domain],
            seen_count=seen[domain],
            k=args.samples_per_domain,
            item=item,
        )

        if total % 50_000 == 0:
            done = sum(1 for d in reservoirs if len(reservoirs[d]) >= args.samples_per_domain)
            print(
                f"[progress] streamed={total} domains={len(reservoirs)} filled={done}",
                file=sys.stderr,
                flush=True,
            )

    manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "domain_field": args.domain_field,
        "samples_per_domain": args.samples_per_domain,
        "seed": args.seed,
        "max_domains": args.max_domains,
        "max_examples": args.max_examples,
        "total_streamed": total,
        "domains": [],
    }

    if args.write_one_file:
        out_path = args.outdir / "samples_by_domain.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for domain in domains_seen_order:
                for item in reservoirs[domain]:
                    item = dict(item)
                    item["__domain"] = domain
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Wrote: {out_path}")
    else:
        for domain in domains_seen_order:
            safe = _sanitize_filename(domain)
            out_path = args.outdir / f"domain={safe}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for item in reservoirs[domain]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Wrote: {out_path}")

    for domain in domains_seen_order:
        manifest["domains"].append(
            {
                "domain": domain,
                "seen": int(seen[domain]),
                "saved": int(len(reservoirs[domain])),
                "file": (
                    "samples_by_domain.jsonl"
                    if args.write_one_file
                    else f"domain={_sanitize_filename(domain)}.jsonl"
                ),
            }
        )

    with open(args.outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {args.outdir / 'manifest.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

