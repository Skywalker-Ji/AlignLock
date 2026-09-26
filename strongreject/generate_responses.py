"""Generate StrongREJECT responses from a local Hugging Face causal LM."""

import argparse
import json
from pathlib import Path

from strongreject.common import DEFAULT_DATASET, load_dataset, sha256, write_json


def format_input(tokenizer, question, prompt_format, system_prompt):
    if prompt_format == "chat":
        if not tokenizer.chat_template:
            raise ValueError("Tokenizer has no chat template; select --prompt-format raw/alpaca explicitly.")
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        messages.append({"role": "user", "content": question})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text, False  # Template already supplies special tokens.
    if system_prompt:
        raise ValueError("--system-prompt-file is supported only for chat format.")
    if prompt_format == "alpaca":
        text = ("Below is an instruction that describes a task. Write a response that appropriately completes the request."
                f"\n\n### Instruction:\n{question}\n\n### Response:\n")
        return text, True
    return question, True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="HF model ID or local full/merged checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=4096, help="Fail above this length; never silently truncate.")
    parser.add_argument("--prompt-format", choices=("chat", "raw", "alpaca"), default="chat")
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int, help="Use the first N prompts for a subset run; not the full paper protocol.")
    parser.add_argument("--dry-run", action="store_true", help="Validate settings/data without loading model weights.")
    args = parser.parse_args(argv)
    if (args.runs < 1 or args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0
            or args.max_new_tokens < 1 or args.max_input_tokens < 1 or (args.limit is not None and args.limit < 1)):
        parser.error("Invalid generation parameters.")
    dataset = load_dataset(args.dataset)
    rows = dataset[:args.limit] if args.limit is not None else dataset
    system_prompt = args.system_prompt_file.read_text(encoding="utf-8") if args.system_prompt_file else ""
    if system_prompt and args.prompt_format != "chat":
        parser.error("System prompts require --prompt-format chat.")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be new or empty; generation does not overwrite old responses.")
    manifest = {"model_path": args.model_path, "dataset_sha256": sha256(args.dataset),
                "dataset_size": len(dataset), "prompts_per_run": len(rows), "runs": args.runs,
                "seeds": [args.seed + i for i in range(args.runs)],
                "do_sample": True, "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                "num_beams": 1, "max_new_tokens": args.max_new_tokens, "max_input_tokens": args.max_input_tokens,
                "prompt_format": args.prompt_format, "system_prompt": system_prompt,
                "dtype": args.dtype, "device_map": args.device_map, "trust_remote_code": args.trust_remote_code,
                "completed_responses": 0, "complete": False}
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype,
                                              device_map=args.device_map, trust_remote_code=args.trust_remote_code)
    model.eval()
    eos_id = model.generation_config.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id[0] if isinstance(eos_id, list) else eos_id
    # Fresh configuration prevents inherited beams/penalties/sampling settings.
    config = GenerationConfig(do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                              num_beams=1, max_new_tokens=args.max_new_tokens,
                              eos_token_id=eos_id, pad_token_id=pad_id, bos_token_id=tokenizer.bos_token_id)
    input_device = model.get_input_embeddings().weight.device
    context_limit = getattr(model.config, "max_position_embeddings", None)
    encoded = []
    for row in rows:
        text, add_special = format_input(tokenizer, row["forbidden_prompt"], args.prompt_format, system_prompt)
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=add_special, truncation=False)
        length = tokens["input_ids"].shape[-1]
        if length > args.max_input_tokens or (context_limit and length + args.max_new_tokens > context_limit):
            raise ValueError(f"Prompt {row['prompt_id']} exceeds configured/context token budget; adjust limits explicitly.")
        encoded.append(tokens)
    manifest.update(torch_version=torch.__version__, transformers_version=transformers.__version__,
                    model_commit=getattr(model.config, "_commit_hash", None),
                    chat_template=tokenizer.chat_template if args.prompt_format == "chat" else None,
                    generation_config=config.to_dict())
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "generation_config.json", manifest)
    with (output / "responses.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for run_index in range(args.runs):
            seed = args.seed + run_index
            set_seed(seed)
            for row, tokens in zip(rows, encoded):
                tokens = {name: tensor.to(input_device) for name, tensor in tokens.items()}
                input_length = tokens["input_ids"].shape[-1]
                with torch.inference_mode():
                    generated = model.generate(**tokens, generation_config=config)
                new_tokens = generated[0, input_length:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                result = {"run_id": run_index + 1, "seed": seed, **row, "response": response,
                          "input_tokens": input_length, "output_tokens": len(new_tokens),
                          "reached_max_new_tokens": len(new_tokens) >= args.max_new_tokens}
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                manifest["completed_responses"] += 1
                write_json(output / "generation_config.json", manifest)
                print(f"[{manifest['completed_responses']}/{len(rows) * args.runs}] run={run_index + 1} prompt={row['prompt_id']}", flush=True)
    manifest["complete"] = True
    write_json(output / "generation_config.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
