#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_DATASET = (
    Path(__file__).resolve().parent.parent
    / "code"
    / "code_alpaca_20k_test.json"
)


def load_dataset(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must contain a list.")
    return data


def make_question(item, index):
    instruction = str(item.get("instruction", "")).strip()
    input_text = str(item.get("input", "")).strip()
    reference = str(item.get("output", "")).strip()

    if not instruction:
        raise ValueError(f"Item {index}: missing 'instruction'.")
    if not reference:
        raise ValueError(f"Item {index}: missing reference 'output'.")

    question = (
        f"{instruction}\n\nInput:\n{input_text}"
        if input_text else instruction
    )
    return question, reference


def make_prompt(tokenizer, question, prompt_format):
    if prompt_format == "chat":
        if not tokenizer.chat_template:
            raise ValueError(
                "Tokenizer has no chat template; use --prompt-format alpaca."
            )
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )

    if "\n\nInput:\n" in question:
        instruction, input_text = question.split("\n\nInput:\n", 1)
        return (
            f"### Instruction:\n{instruction}\n"
            f"### Input:\n{input_text}\n"
            f"### Response:\n"
        )

    return f"### Instruction:\n{question}\n### Response:\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate downstream-task responses and evaluate ROUGE-L."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-file", default="rougel_results.json")
    parser.add_argument(
        "--prompt-format",
        choices=["chat", "alpaca"],
        default="chat",
    )
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--sample-size", type=int)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Default expects: {DEFAULT_DATASET}"
        )

    if args.dtype == "auto":
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16 if torch.cuda.is_available()
            else torch.float32
        )
    else:
        dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    data = load_dataset(dataset_path)
    if args.sample_size is not None:
        data = data[: args.sample_size]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = []
    total = 0.0

    for index, item in enumerate(data):
        question, reference = make_question(item, index)
        prompt = make_prompt(tokenizer, question, args.prompt_format)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(input_device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        rouge_l = scorer.score(reference, response)["rougeL"].fmeasure
        total += rouge_l

        results.append({
            "index": index,
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "question": question,
            "reference": reference,
            "response": response,
            "rougeL": rouge_l,
        })

        print(f"[{index + 1}/{len(data)}] ROUGE-L={rouge_l:.4f}")

    average = total / len(results) if results else 0.0
    output = {
        "summary": {
            "model_path": args.model_path,
            "dataset_path": str(dataset_path),
            "num_examples": len(results),
            "average_rougeL": average,
            "prompt_format": args.prompt_format,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
        },
        "results": results,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nAverage ROUGE-L: {average:.4f}")
    print(f"Saved to: {args.output_file}")


if __name__ == "__main__":
    main()
