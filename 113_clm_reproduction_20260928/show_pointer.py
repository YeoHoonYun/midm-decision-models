import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
for p in sorted(glob.glob(os.path.join(HERE, "results", "pointer", "FINAL_*.json"))
                + glob.glob(os.path.join(HERE, "results", "seeds", "FINAL_*.json"))):
    d = json.load(open(p))["eval"]
    jb = sum(d[k]["acc"] * d[k]["n"] for k in ("jb_original", "jb_easy", "jb_hard")) / 231
    print(os.path.basename(p)[6:-5], " ".join(f"{k}={v['acc']:.3f}" for k, v in d.items() if not k.startswith("jb_")),
          f"jb_public={jb:.3f}")
