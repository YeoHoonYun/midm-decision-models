import json
import sys

for p in sys.argv[1:]:
    print(p)
    for k, v in json.load(open(p))["eval"].items():
        print(f"  {k:12s} n={v['n']:5d} acc={v['acc']:.4f} brier={v['brier']:.4f} ece={v['ece']:.4f}")
