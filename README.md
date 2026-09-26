# AlignLock

Anonymous implementation of **AlignLock: Align and Freeze Safety-Critical Layers for Robust LLM Fine-Tuning**.

Commands assume Linux and the repository root. Replace `MODEL_PATH` with your checkpoint.

```bash
pip install -r requirements.txt
pip install --no-deps -e ./AlignLock-LLaMAFactory
```

## 1. Datasets

Training and downstream datasets are provided in `datas/`:

| Directory | Dataset |
|---|---|
| `alpaca_en/` | Alpaca instruction tuning |
| `saferlhf/` | SafeRLHF preference pairs (10K) |
| `code/` | Code-Alpaca |
| `gsm8k_math/` | GSM8K |
| `alpaca_finance/` | Finance-Alpaca |
| `maliciousinstruct/` | MaliciousInstruct |
| `med_qa/` | Medical QA |

## 2. Safety-layer localization

`safety_layer_localization/` contains the localization code and two probe datasets in `datasets/`: `Malicious_dataset.csv` and `Over_rejection_dataset.csv`.

```bash
python safety_layer_localization/alignlock_localization.py --model-path MODEL_PATH --dataset-path safety_layer_localization/datasets/Malicious_dataset.csv --alpha 0.8 --output-dir outputs/localization
```

Replace the dataset filename to use the other probe set. The selected interval is saved in `result.json`; layer indices are zero-based and inclusive.

## 3. Fine-tuning

`AlignLock-LLaMAFactory/` contains the modified training framework.

| Script in `scripts/alignlock/` | Stage |
|---|---|
| `instruction_tuning.sh` | Full-parameter instruction SFT |
| `targeted_safety_alignment.sh` | DPO restricted to the selected safety layers |
| `downstream_finetuning.sh` | Downstream SFT with those safety layers frozen |

Set the model path, template, and layer interval for your experiment. Check dataset paths in `data/dataset_info.json`; `alpaca_en` should point to `../../datas/alpaca_en/alpaca_data_en_52k.json`.

```bash
pip install "deepspeed>=0.10.0,<=0.16.2"
cd AlignLock-LLaMAFactory
bash scripts/alignlock/instruction_tuning.sh
bash scripts/alignlock/targeted_safety_alignment.sh
bash scripts/alignlock/downstream_finetuning.sh
cd ..
```

## 4. StrongREJECT safety evaluation

- Dataset: `strongreject/strongreject_dataset/strongreject_dataset.csv`
- Judge prompt: `strongreject/strongreject/strongreject_evaluator_prompt.txt`

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
python strongreject/generate_responses.py --model-path MODEL_PATH --output-dir outputs/safety_generation
python strongreject/evaluate_responses.py --responses outputs/safety_generation/responses.jsonl --judge-model gpt-4o --output-dir outputs/safety_evaluation
```

Defaults: 313 prompts, five generation runs. Outputs include StrongREJECT score, reject rate, and malicious rate.

## 5. Utility evaluation

**MMLU (5-shot)** with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.8):

```bash
pip install "lm_eval==0.4.8"
lm_eval --model hf --model_args pretrained=MODEL_PATH --tasks mmlu --num_fewshot 5 --device cuda:0 --batch_size auto --output_path outputs/mmlu
```

**HumanEval pass@1** with the bundled `HumanEval/` code (run in an isolated Linux environment):

```bash
cd HumanEval
python main.py --model_name_or_path MODEL_PATH --task_name humaneval --data_file dataset/humaneval_python.jsonl --do_sample false --max_new_tokens 1024 --torch_dtype bf16 --output_path out.jsonl --save_logs_path logs.jsonl --save_metrics_path metric.json
cd ..
```

**ROUGE-L and AI-score** with `tasks_evaluate/` (AI-score also requires `OPENAI_API_KEY`):

```bash
python tasks_evaluate/evaluate_rougel.py --model-path MODEL_PATH --dataset-path datas/code/code_alpaca_20k_test.json --output-file outputs/rougel.json
python tasks_evaluate/evaluate_ai_score.py --input-file outputs/rougel.json --judge-model gpt-4o-mini --output-file outputs/ai_score.json
```
