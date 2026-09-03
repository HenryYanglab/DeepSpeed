#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Create one run-specific FastOffload config with structured telemetry paths."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as input_file:
        config = json.load(input_file)
    telemetry = config.setdefault("telemetry", {})
    telemetry["log_interval"] = args.log_interval
    telemetry["memory_metrics"] = True
    telemetry["jsonl_path"] = str((args.run_dir / "telemetry.jsonl").resolve())
    telemetry["csv_path"] = str((args.run_dir / "telemetry.csv").resolve())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(config, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
