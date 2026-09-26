#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import gc
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from alignlock_localization import (
    Localizer,
    load_expressions,
    load_prompts,
    normalize,
    scaled_interval,
    save_json,
    sha256,
    prepare_output,
)


def parse_checkpoint(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Checkpoint must be LABEL=PATH, e.g. 'Vanilla Align.=/models/qwen-align'."
        )
    label, path = value.split("=", 1)
    label, path = label.strip(), path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("Checkpoint label and path must be non-empty.")
    return label, path


def build_refusal_patterns(expressions):
    patterns = []
    for expression in expressions:
        text = normalize(expression)
        patterns.append(
            re.compile(r"(?<!\w)" + re.escape(text) + r"(?!\w)")
        )
    return patterns


def is_refusal(text, patterns):
    text = normalize(text)
    return any(pattern.search(text) for pattern in patterns)


def make_localizer_args(args, model_path):
    return SimpleNamespace(
        model_path=model_path,
        tokenizer_path=args.tokenizer_path,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
        device=args.device,
        window_size=args.max_new_tokens,
        phrase_batch_size=1,
        prompt_format=args.prompt_format,
        system_prompt=args.system_prompt,
        max_input_tokens=args.max_input_tokens,
        log_every=args.log_every,
        backup_device=args.backup_device,
    )


def refusal_rate(localizer, encoded, patterns):
    """Keyword-based behavioral rate, not the localization probability proxy."""
    if not encoded:
        raise ValueError("The evaluation dataset must not be empty.")
    refused = 0
    at_limit = 0
    cfg = localizer.generation_config
    eos = cfg.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos]))
    for i, ids in enumerate(encoded):
        inputs = torch.tensor([ids], dtype=torch.long, device=localizer.device)
        with torch.inference_mode():
            output = localizer.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=cfg,
                return_dict_in_generate=False,
                output_logits=False,
                output_scores=False,
            )
        generated = output[0, len(ids):].tolist()
        ended = any(token in eos_ids for token in generated)
        if not ended and len(generated) >= cfg.max_new_tokens:
            at_limit += 1
        for j, token in enumerate(generated):
            if token in eos_ids:
                generated = generated[:j]
                break
        response = localizer.tokenizer.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        refused += int(is_refusal(response, patterns))
        if (i + 1) % localizer.args.log_every == 0:
            print(f"  generated {i + 1}/{len(encoded)}")
    return {"rate": refused / len(encoded), "refused": refused,
            "num_prompts": len(encoded), "responses_at_length_limit": at_limit}


def relative_sensitivity(baseline, perturbed):
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (baseline, perturbed)):
        raise ValueError("Refusal rates must be finite and in [0, 1].")
    return (baseline - perturbed) / baseline if baseline > 0 else math.nan


def object_hash(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def comparison_signature(localizer, encoded):
    """Check compatible architecture and identical evaluation inputs/decoding."""
    keys = ("model_type", "num_hidden_layers", "hidden_size", "intermediate_size",
            "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
            "max_position_embeddings", "rope_theta", "rope_scaling", "sliding_window",
            "num_experts", "num_local_experts", "num_experts_per_tok")
    return {
        "architecture": {key: getattr(localizer.model.config, key, None) for key in keys},
        "vocabulary_sha256": object_hash(localizer.tokenizer.get_vocab()),
        "special_tokens": localizer.tokenizer.special_tokens_map,
        "chat_template": (localizer.tokenizer.chat_template
                          if localizer.args.prompt_format == "chat" else None),
        "encoded_prompts_sha256": object_hash(encoded),
        "generation_config": localizer.generation_config.to_dict(),
    }


def check_comparable(reference, current):
    differences = [key for key in reference if reference[key] != current[key]]
    if differences:
        raise ValueError("Checkpoints are not comparable; differing fields: "
                         + ", ".join(differences))


def save_matrix_csv(path, labels, matrix):
    n_layers = matrix.shape[1]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage"] + list(range(n_layers)))
        for label, row in zip(labels, matrix):
            writer.writerow(
                [label] + ["" if np.isnan(x) else f"{x:.8f}" for x in row]
            )


def plot_heatmap(path, labels, matrix, vlim, block, title=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FormatStrFormatter

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or 0 in matrix.shape or matrix.shape[0] != len(labels):
        raise ValueError("Heatmap must be a nonempty stage-by-layer matrix.")
    if np.isinf(matrix).any():
        raise ValueError("Heatmap contains infinite values.")
    if not math.isfinite(vlim) or vlim <= 0:
        raise ValueError("Color limit must be finite and positive.")

    n_rows, n_layers = matrix.shape
    if block is not None and not 0 <= block[0] <= block[1] < n_layers:
        raise ValueError("Invalid safety-critical block.")

    finite = matrix[np.isfinite(matrix)]
    low = bool(np.any(finite < -vlim))
    high = bool(np.any(finite > vlim))
    extend = "both" if low and high else "min" if low else "max" if high else "neither"

    fig_w = max(12.0, n_layers * 0.48)
    fig_h = max(4.0, n_rows * 0.72 + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = LinearSegmentedColormap.from_list(
        "alignlock_sensitivity",
        ["#B7C6CD", "#F2F2F2", "#B7ACCF"],
        N=256,
    )
    cmap.set_bad("#D4D4D4")

    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-vlim,
        vmax=vlim,
    )

    ax.set_xticks(np.arange(n_layers))
    ax.set_xticklabels(np.arange(n_layers), fontsize=8)
    ax.set_yticks(np.arange(n_rows))
    display_labels = [
        label + ("  [N/A]" if np.isnan(matrix[i]).all() else "")
        for i, label in enumerate(labels)
    ]
    ax.set_yticklabels(display_labels, fontsize=10)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Layer index", fontsize=11, labelpad=8)

    ax.set_xticks(np.arange(-0.5, n_layers, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(image, ax=ax, pad=0.018, extend=extend)
    ticks = np.linspace(-vlim, vlim, 7)
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar.set_label("Relative safety sensitivity", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=12)

    if block is not None:
        start, end = block
        transform = ax.get_xaxis_transform()
        y = -0.10
        ax.plot(
            [start - 0.5, end + 0.5],
            [y, y],
            color="#333333",
            transform=transform,
            clip_on=False,
            linewidth=1.0,
        )
        for x in (start - 0.5, end + 0.5):
            ax.plot(
                [x, x],
                [y, y + 0.04],
                color="#333333",
                transform=transform,
                clip_on=False,
                linewidth=1.0,
            )
        ax.text(
            (start + end) / 2,
            y - 0.04,
            f"Safety-critical block [{start}, {end}]",
            ha="center",
            va="top",
            fontsize=8,
            transform=transform,
        )

    bottom = 0.18 if block is not None else 0.08
    fig.tight_layout(rect=(0, bottom, 1, 0.95 if title else 1))

    for suffix in (".png", ".pdf"):
        fig.savefig(Path(path).with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "vlim": vlim,
        "colorbar_extend": extend,
        "clipped_cells": int(np.sum(np.abs(finite) > vlim)),
        "undefined_cells": int(np.isnan(matrix).sum()),
        "display_units": "fraction",
        "csv_units": "fraction",
        "colormap": {
            "negative": "#B7C6CD",
            "zero": "#F2F2F2",
            "positive": "#B7ACCF",
        },
    }

def build_parser():
    p = argparse.ArgumentParser(
        description="Single-layer scaling analysis for the AlignLock Figure-2 heatmap."
    )
    p.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint,
        required=True,
        help="Repeat LABEL=PATH in heatmap row order.",
    )
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--text-column")
    p.add_argument("--refusal-expressions")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--prompt-format", choices=["chat", "raw"], default="chat")
    p.add_argument("--system-prompt", default="")
    p.add_argument("--max-input-tokens", type=int, default=2048)

    p.add_argument("--tokenizer-path")
    p.add_argument("--device", default="auto")
    p.add_argument("--device-map", choices=["none", "auto"], default="none")
    p.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    p.add_argument("--backup-device", choices=["cpu", "same"], default="cpu")
    p.add_argument("--trust-remote-code", action="store_true")

    p.add_argument("--sample-size", type=int)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--vlim", type=float, default=0.3,
                   help="Symmetric color limit as a fraction; default: 0.3.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--block-start", type=int)
    p.add_argument("--block-end", type=int)
    return p


def validate_args(args):
    if not math.isfinite(args.alpha) or args.alpha <= 0 or args.alpha == 1:
        raise ValueError("--alpha must be finite, positive and != 1.")
    if min(args.max_new_tokens, args.max_input_tokens, args.log_every) < 1:
        raise ValueError("Token and logging parameters must be positive.")
    if args.sample_size is not None and args.sample_size < 1:
        raise ValueError("--sample-size must be positive.")
    if not math.isfinite(args.vlim) or args.vlim <= 0:
        raise ValueError("--vlim must be finite and positive.")
    labels = [label for label, _ in args.checkpoint]
    if len(labels) != len(set(labels)):
        raise ValueError("Checkpoint labels must be unique.")
    if (args.block_start is None) != (args.block_end is None):
        raise ValueError("Provide both --block-start and --block-end.")
    if args.block_start is not None and not 0 <= args.block_start <= args.block_end:
        raise ValueError("Invalid safety-critical block.")


def run_analysis(args, output_dir, metadata):
    prompts = load_prompts(args.dataset_path, args.text_column)
    if args.sample_size is not None:
        prompts = prompts[:args.sample_size]

    expressions = load_expressions(args.refusal_expressions)
    patterns = build_refusal_patterns(expressions)
    block = None if args.block_start is None else (args.block_start, args.block_end)
    metadata.update({"num_prompts": len(prompts), "prompts_sha256": object_hash(prompts),
                     "refusal_expressions": expressions, "safety_critical_block": block})
    save_json(output_dir / "prompts.json", prompts)
    save_json(output_dir / "metadata.json", metadata)

    labels = []
    sensitivities = []
    scaled_rates_all = []
    baseline_rates = []
    reference = None
    metadata["stages"] = []

    for stage_index, (label, model_path) in enumerate(args.checkpoint):
        print(f"\n=== {label} ===")
        localizer = Localizer(make_localizer_args(args, model_path), expressions)
        encoded = [localizer.encode(prompt) for prompt in prompts]
        n_layers = len(localizer.groups)

        if block is not None and block[1] >= n_layers:
            raise ValueError("Safety-critical block exceeds checkpoint layer count.")
        signature = comparison_signature(localizer, encoded)
        if reference is None:
            reference = signature
            metadata["comparison_signature"] = reference
        else:
            check_comparable(reference, signature)

        baseline = refusal_rate(localizer, encoded, patterns)
        base_rate = baseline["rate"]
        print(f"  baseline refusal rate: {base_rate:.6f}")
        if base_rate == 0:
            print("  baseline is zero: relative sensitivity is undefined (N/A).")
        stage = {"label": label, "path": model_path, "baseline": baseline,
                 "layers": [], "status": "running",
                 "dtype": str(localizer.dtype),
                 "transformers_version": localizer.transformers_version}
        stage_path = output_dir / f"stage_{stage_index + 1:02d}.json"
        save_json(stage_path, stage)

        layer_rates = []
        layer_sensitivity = []

        for layer in range(n_layers):
            print(f"  layer {layer}/{n_layers - 1}")
            with scaled_interval(
                localizer.groups,
                layer,
                layer,
                args.alpha,
                args.backup_device,
            ):
                perturbed = refusal_rate(localizer, encoded, patterns)

            perturbed_rate = perturbed["rate"]
            layer_rates.append(perturbed_rate)
            score = relative_sensitivity(base_rate, perturbed_rate)
            layer_sensitivity.append(score)
            stage["layers"].append({"layer": layer, **perturbed,
                                    "sensitivity": score if math.isfinite(score) else None})
            save_json(stage_path, stage)

        labels.append(label)
        baseline_rates.append(base_rate)
        scaled_rates_all.append(layer_rates)
        sensitivities.append(layer_sensitivity)
        stage["status"] = "complete"
        save_json(stage_path, stage)
        metadata["stages"].append({"label": label, "file": stage_path.name})
        save_json(output_dir / "metadata.json", metadata)

        del localizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    matrix = np.asarray(sensitivities, dtype=float)
    rate_matrix = np.asarray(scaled_rates_all, dtype=float)

    save_matrix_csv(
        output_dir / "relative_sensitivity.csv",
        labels,
        matrix,
    )
    save_matrix_csv(
        output_dir / "scaled_refusal_rates.csv",
        labels,
        rate_matrix,
    )

    with (output_dir / "baseline_refusal_rates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "baseline_refusal_rate"])
        writer.writerows(zip(labels, baseline_rates))

    if block is not None:
        with (output_dir / "block_mean_sensitivity.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["stage", "start_layer", "end_layer", "mean_relative_sensitivity"])
            for label, row in zip(labels, matrix):
                values = row[block[0]:block[1] + 1]
                value = float(values.mean()) if np.isfinite(values).all() else None
                writer.writerow([label, *block, "" if value is None else f"{value:.8f}"])

    metadata["plot"] = plot_heatmap(
        output_dir / "layer_sensitivity_heatmap.png",
        labels,
        matrix,
        args.vlim,
        block,
    )

    print(f"\nSaved to: {output_dir}")
    print("  relative_sensitivity.csv")
    print("  scaled_refusal_rates.csv")
    print("  baseline_refusal_rates.csv")
    print("  layer_sensitivity_heatmap.png")
    print("  layer_sensitivity_heatmap.pdf")
    if block is not None:
        print("  block_mean_sensitivity.csv")
    print("  metadata.json")


def main():
    args = build_parser().parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = prepare_output(args.output_dir)
    metadata = {"status": "running", "arguments": vars(args),
                "formula": "S_t(l) = (R_t - R_t^(l)) / R_t",
                "layer_indexing": "zero_based_inclusive",
                "refusal_rate": "fraction of bounded generated responses matching a refusal expression",
                "matching": "NFKC, casefold, normalized apostrophes/whitespace, word boundaries",
                "undefined": "baseline refusal rate = 0; JSON null / empty CSV / gray heatmap",
                "comparability_note": "Architecture/tokenization checks do not establish checkpoint ancestry.",
                "torch_version": torch.__version__}
    try:
        metadata["dataset_sha256"] = sha256(args.dataset_path)
        metadata["script_sha256"] = sha256(__file__)
        metadata["localization_script_sha256"] = sha256(
            Path(__file__).with_name("alignlock_localization.py"))
        save_json(output_dir / "metadata.json", metadata)
        run_analysis(args, output_dir, metadata)
        metadata["status"] = "complete"
        save_json(output_dir / "metadata.json", metadata)
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        save_json(output_dir / "metadata.json", metadata)
        raise


if __name__ == "__main__":
    main()
