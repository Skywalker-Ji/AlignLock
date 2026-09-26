#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI


EVALUATOR_PROMPT = """You are an expert in evaluating the accuracy of responses to user questions.
Given the following:

Question:
{question}

Response:
{response}

1 - Completely incorrect or irrelevant (does not answer the question or the core claim is wrong)
2 - Mostly incorrect (some relevance, but major errors and/or misses most key requirements)
3 - Partially correct (addresses some key points, but has significant errors or important omissions)
4 - Mostly correct (meets most key requirements; only minor errors or omissions that don't change the main correctness)
5 - Fully correct and complete (addresses all key requirements with no meaningful errors or omissions)

Respond with exactly: Score: X"""

SCORE_PATTERN = re.compile(r"^\s*Score\s*:\s*([1-5])\s*$", re.IGNORECASE)


def load_responses(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError(
            "Input must be the JSON produced by evaluate_rougel.py."
        )
    return data["results"]


def call_judge(client, model, question, response, retries):
    prompt = EVALUATOR_PROMPT.format(
        question=question,
        response=response,
    )

    error = None
    for attempt in range(retries):
        try:
            result = client.responses.create(
                model=model,
                input=prompt,
                temperature=0,
                max_output_tokens=20,
            )
            text = result.output_text.strip()
            match = SCORE_PATTERN.match(text)
            if not match:
                raise ValueError(f"Unexpected judge output: {text!r}")
            return int(match.group(1)), text
        except Exception as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Judge failed after {retries} attempts: {error}")


def save_output(path, model, results):
    scores = [item["ai_score"] for item in results]
    payload = {
        "summary": {
            "judge_model": model,
            "num_scored": len(scores),
            "average_ai_score": (
                sum(scores) / len(scores) if scores else 0.0
            ),
        },
        "results": results,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated responses with the paper's 1-5 AI-score prompt."
    )
    parser.add_argument("--input-file", default="rougel_results.json")
    parser.add_argument("--output-file", default="ai_score_results.json")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    if args.retries < 1 or args.save_every < 1:
        raise ValueError("--retries and --save-every must be positive.")

    data = load_responses(args.input_file)
    client = OpenAI()

    completed = {}
    output_path = Path(args.output_file)

    if args.resume and output_path.exists():
        old = json.loads(output_path.read_text(encoding="utf-8"))
        for item in old.get("results", []):
            if "index" in item and "ai_score" in item:
                completed[int(item["index"])] = item

    results = []

    for position, item in enumerate(data):
        index = int(item.get("index", position))

        if index in completed:
            record = completed[index]
        else:
            question_value = item.get("question")
            response_value = item.get("response")
            if question_value is None or not str(question_value).strip():
                raise ValueError(f"Item {index}: missing question.")
            if response_value is None:
                raise ValueError(f"Item {index}: missing response.")
            question = str(question_value).strip()
            response = str(response_value).strip()

            score, raw = call_judge(
                client,
                args.judge_model,
                question,
                response,
                args.retries,
            )

            record = dict(item)
            record["ai_score"] = score
            record["judge_output"] = raw

        results.append(record)
        average = sum(x["ai_score"] for x in results) / len(results)

        print(
            f"[{position + 1}/{len(data)}] "
            f"AI-score={record['ai_score']} | avg={average:.3f}"
        )

        if len(results) % args.save_every == 0:
            save_output(output_path, args.judge_model, results)

    save_output(output_path, args.judge_model, results)

    average = sum(x["ai_score"] for x in results) / len(results) if results else 0.0
    print(f"\nAverage AI-score: {average:.3f}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
