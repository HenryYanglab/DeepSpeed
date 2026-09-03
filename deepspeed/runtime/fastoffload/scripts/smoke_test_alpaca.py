#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""CPU-only smoke tests for the Alpaca data and configuration helpers."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset, DatasetDict

from finetune_alpaca import SupervisedDataCollator, load_alpaca_dataset, load_deepspeed_config, tokenize_dataset


class FakeTokenizer:
    eos_token_id = 2

    def __call__(self, text, add_special_tokens, truncation):
        prefix = [1] if add_special_tokens else []
        return {"input_ids": prefix + [3 + ord(character) % 17 for character in text]}


class AlpacaScriptSmokeTest(unittest.TestCase):

    def test_collator_masks_padding_labels(self):
        collator = SupervisedDataCollator(pad_token_id=0, pad_to_multiple_of=4)
        batch = collator([{
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [-100, 2, 3],
        }, {
            "input_ids": [4, 5],
            "attention_mask": [1, 1],
            "labels": [-100, 5],
        }])

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 4))
        self.assertEqual(batch["input_ids"][1].tolist(), [4, 5, 0, 0])
        self.assertEqual(batch["labels"][1].tolist(), [-100, 5, -100, -100])

    def test_tokenization_masks_prompt_and_keeps_response(self):
        dataset = Dataset.from_list([{
            "instruction": "Summarize",
            "input": "A short input",
            "output": "A short answer",
        }])
        args = SimpleNamespace(max_length=512, preprocessing_num_workers=1)

        tokenized = tokenize_dataset(dataset, FakeTokenizer(), args)

        self.assertEqual(len(tokenized), 1)
        labels = tokenized[0]["labels"]
        self.assertIn(FakeTokenizer.eos_token_id, labels)
        self.assertTrue(any(label == -100 for label in labels))
        self.assertTrue(any(label != -100 for label in labels))

    def test_loads_dataset_saved_to_directory(self):
        source = Dataset.from_list([{
            "instruction": "Summarize",
            "input": "An input",
            "output": "An answer",
        }])
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "alpaca"
            DatasetDict({"train": source}).save_to_disk(dataset_path)
            args = SimpleNamespace(dataset_path=dataset_path,
                                   dataset_name="unused",
                                   dataset_split="train",
                                   max_samples=None,
                                   seed=42)
            loaded = load_alpaca_dataset(args)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["instruction"], "Summarize")

    def test_runtime_batch_and_precision_overrides(self):
        source_config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "train_batch_size": 1,
            "bf16": {
                "enabled": True
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-3
                }
            },
            "zero_optimization": {
                "stage": 2
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "ds.json"
            config_path.write_text(json.dumps(source_config), encoding="utf-8")
            args = SimpleNamespace(deepspeed_config=config_path,
                                   micro_batch_size=2,
                                   gradient_accumulation_steps=4,
                                   log_interval=7,
                                   learning_rate=2e-5,
                                   precision="fp16")
            old_world_size = os.environ.get("WORLD_SIZE")
            os.environ["WORLD_SIZE"] = "2"
            try:
                config = load_deepspeed_config(args)
            finally:
                if old_world_size is None:
                    os.environ.pop("WORLD_SIZE", None)
                else:
                    os.environ["WORLD_SIZE"] = old_world_size

        self.assertEqual(config["train_batch_size"], 16)
        self.assertEqual(config["steps_per_print"], 7)
        self.assertEqual(config["optimizer"]["params"]["lr"], 2e-5)
        self.assertEqual(config["fp16"], {"enabled": True})
        self.assertNotIn("bf16", config)


if __name__ == "__main__":
    unittest.main()
