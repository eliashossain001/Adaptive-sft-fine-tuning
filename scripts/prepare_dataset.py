"""Merge / clean / validate JSONL datasets.

For demo purposes, this simply copies dummy_instructions.jsonl → processed.jsonl
and ensures required keys exist.
"""
import json, argparse, os, pathlib

def main(args):
    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            ex = json.loads(line)
            assert all(k in ex for k in ("instruction", "input", "output")), "Missing keys"
            fout.write(json.dumps(ex) + "\n")
    print(f"Saved processed dataset → {dst}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/dummy_instructions.jsonl")
    p.add_argument("--dst", default="data/processed.jsonl")
    args = p.parse_args()
    main(args)