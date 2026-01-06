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
import time
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


def _escape_sql_string(s: str) -> str:
    # SQL-style escaping for single quotes: O'Hara -> 'O''Hara'
    return s.replace("'", "''")


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


def _viewer_get(url: str, *, token: str | None, timeout: int = 60) -> dict[str, Any]:
    import requests  # type: ignore

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, headers=headers, timeout=timeout)
    if resp.status_code == 429:
        raise RuntimeError("rate_limited")
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected response (expected JSON object).")
    return data


def _viewer_resolve_config(
    base_url: str, dataset: str, split: str, *, token: str | None
) -> str:
    url = f"{base_url}/splits?dataset={dataset}"
    data = _viewer_get(url, token=token)
    splits = data.get("splits")
    if not isinstance(splits, list) or not splits:
        raise ValueError("Could not resolve dataset splits/configs from /splits.")
    # pick first config that contains the requested split
    for item in splits:
        if not isinstance(item, dict):
            continue
        if item.get("split") == split and item.get("config"):
            return str(item["config"])
    # fallback: first config
    first = splits[0]
    if isinstance(first, dict) and first.get("config"):
        return str(first["config"])
    raise ValueError("Failed to resolve dataset config.")


def _viewer_guess_domain_field(features: list[dict[str, Any]]) -> str | None:
    # Prefer a feature whose name includes domain/category/task/subset/source.
    preferred = ("domain", "category", "task", "subset", "source")
    names = [f.get("name") for f in features if isinstance(f, dict)]
    for p in preferred:
        for n in names:
            if isinstance(n, str) and p == n.lower():
                return n
    for p in preferred:
        for n in names:
            if isinstance(n, str) and p in n.lower():
                return n
    return None


def _viewer_get_domain_frequencies(
    base_url: str,
    dataset: str,
    config: str,
    split: str,
    domain_field: str,
    *,
    token: str | None,
) -> tuple[dict[str, int], bool]:
    url = f"{base_url}/statistics?dataset={dataset}&config={config}&split={split}"
    data = _viewer_get(url, token=token)
    stats = data.get("statistics")
    partial = bool(data.get("partial", False))
    if not isinstance(stats, list):
        raise ValueError("Unexpected /statistics response structure.")
    for col in stats:
        if not isinstance(col, dict):
            continue
        if col.get("column_name") != domain_field:
            continue
        col_stats = col.get("column_statistics")
        if not isinstance(col_stats, dict):
            continue
        freqs = col_stats.get("frequencies")
        if isinstance(freqs, dict):
            out: dict[str, int] = {}
            for k, v in freqs.items():
                try:
                    out[str(k)] = int(v)
                except Exception:
                    continue
            return out, partial
    raise ValueError(
        f"Domain field '{domain_field}' not found in /statistics frequencies. "
        "Try specifying --domain-field explicitly or use streaming backend."
    )


def _viewer_sample_by_domain(
    *,
    base_url: str,
    dataset: str,
    config: str,
    split: str,
    domain_field: str,
    samples_per_domain: int,
    seed: int,
    outdir: pathlib.Path,
    max_domains: int,
    max_requests: int,
    token: str | None,
    sleep_on_429: float,
) -> None:
    # Get domain list + approximate counts
    freqs, partial = _viewer_get_domain_frequencies(
        base_url, dataset, config, split, domain_field, token=token
    )
    domains = list(freqs.keys())
    # Deterministic domain order for repeatability
    domains.sort()
    if max_domains and len(domains) > max_domains:
        domains = domains[:max_domains]

    rng = random.Random(seed)
    collected: dict[str, dict[int, dict[str, Any]]] = {d: {} for d in domains}

    def done() -> bool:
        return all(len(collected[d]) >= samples_per_domain for d in domains)

    reqs = 0
    while not done():
        if max_requests and reqs >= max_requests:
            break
        # pick a domain that still needs samples, weighted by remaining need
        need_domains = [d for d in domains if len(collected[d]) < samples_per_domain]
        d = rng.choice(need_domains)
        count = max(1, int(freqs.get(d, 1)))
        # request a slice
        length = min(100, samples_per_domain - len(collected[d]))
        # random offset within filtered rows
        offset = rng.randrange(max(1, count))
        where = f"\"{domain_field}\"='{_escape_sql_string(d)}'"
        url = (
            f"{base_url}/filter?dataset={dataset}&config={config}&split={split}"
            f"&where={where}&offset={offset}&length={length}"
        )
        try:
            data = _viewer_get(url, token=token)
        except RuntimeError as e:
            if str(e) == "rate_limited":
                time.sleep(sleep_on_429)
                continue
            raise
        reqs += 1
        rows = data.get("rows")
        if not isinstance(rows, list) or not rows:
            continue
        for r in rows:
            if not isinstance(r, dict):
                continue
            row_idx = r.get("row_idx")
            row = r.get("row")
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row_idx)
            except Exception:
                continue
            if idx in collected[d]:
                continue
            collected[d][idx] = _jsonable(row)
            if len(collected[d]) >= samples_per_domain:
                break

        if reqs % 50 == 0:
            filled = sum(1 for dd in domains if len(collected[dd]) >= samples_per_domain)
            print(
                f"[progress] requests={reqs} filled={filled}/{len(domains)} partial_stats={partial}",
                file=sys.stderr,
                flush=True,
            )

    manifest = {
        "backend": "dataset-viewer",
        "base_url": base_url,
        "dataset": dataset,
        "config": config,
        "split": split,
        "domain_field": domain_field,
        "samples_per_domain": samples_per_domain,
        "seed": seed,
        "max_domains": max_domains,
        "max_requests": max_requests,
        "partial_statistics": partial,
        "domains": [],
    }

    for d in domains:
        safe = _sanitize_filename(d)
        out_path = outdir / f"domain={safe}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for _, row in sorted(collected[d].items()):
                row = dict(row)
                row["__domain"] = d
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote: {out_path}")
        manifest["domains"].append(
            {
                "domain": d,
                "count_estimate": int(freqs.get(d, 0)),
                "saved": int(len(collected[d])),
                "file": f"domain={safe}.jsonl",
            }
        )

    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {outdir / 'manifest.json'}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Stream OpenThoughts3-1.2M (or any HF dataset) and reservoir-sample "
            "K examples per domain into JSONL files (one pass, no full download)."
        )
    )
    ap.add_argument(
        "--backend",
        choices=["stream", "viewer"],
        default="stream",
        help=(
            "Data backend: 'stream' uses datasets.load_dataset(streaming=True) and "
            "scans the split once (exact reservoir sampling). 'viewer' uses the "
            "Hugging Face dataset-viewer API for faster approximate sampling."
        ),
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
        "--viewer-base-url",
        default="https://datasets-server.huggingface.co",
        help="Base URL for dataset-viewer API when --backend=viewer.",
    )
    ap.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help="Optional HF token for gated datasets (env: HF_TOKEN).",
    )
    ap.add_argument(
        "--viewer-max-requests",
        type=int,
        default=0,
        help=(
            "Maximum number of API requests when --backend=viewer (0 = no limit). "
            "Each request returns up to 100 rows."
        ),
    )
    ap.add_argument(
        "--sleep-on-429",
        type=float,
        default=2.0,
        help="Seconds to sleep and retry when dataset-viewer API rate-limits (HTTP 429).",
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

    if args.backend == "viewer":
        # Fast, approximate sampling via dataset-viewer API (no full split scan on client).
        args.outdir.mkdir(parents=True, exist_ok=True)
        base_url = str(args.viewer_base_url).rstrip("/")

        config = args.config
        if not config:
            config = _viewer_resolve_config(
                base_url, args.dataset, args.split, token=args.hf_token
            )
            print(f"[viewer] resolved config={config}", file=sys.stderr, flush=True)

        # If domain field not provided, try to guess it from /first-rows features.
        domain_field = args.domain_field
        if not domain_field:
            try:
                first = _viewer_get(
                    f"{base_url}/first-rows?dataset={args.dataset}&config={config}&split={args.split}",
                    token=args.hf_token,
                )
                feats = first.get("features")
                if isinstance(feats, list):
                    guessed = _viewer_guess_domain_field([x for x in feats if isinstance(x, dict)])
                    domain_field = guessed
            except Exception:
                domain_field = None

        if not domain_field:
            raise SystemExit(
                "error: could not infer domain field via dataset-viewer. "
                "Please pass --domain-field explicitly."
            )

        _viewer_sample_by_domain(
            base_url=base_url,
            dataset=args.dataset,
            config=str(config),
            split=args.split,
            domain_field=str(domain_field),
            samples_per_domain=args.samples_per_domain,
            seed=args.seed,
            outdir=args.outdir,
            max_domains=args.max_domains,
            max_requests=args.viewer_max_requests,
            token=args.hf_token,
            sleep_on_429=float(args.sleep_on_429),
        )
        return 0

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
