#!/usr/bin/env python3
"""Final public release: GitHub (public + tagged release), Zenodo DOI, Hugging Face (public), DOI back-fill.

    python publish_release.py --version 0.1.0 --dry-run     # builds the archive and prints what would happen
    python publish_release.py --version 0.1.0               # does it (irreversible: public repos, permanent DOI)

Steps
 1. GitHub: tag v<version> on main, push the tag, make the repo public, create a GitHub release.
 2. Zenodo: new deposition from .zenodo.json (+ version, related identifiers), upload the git-archive zip of the tag,
    publish -> DOI. Token: %USERPROFILE%/.config/zenodo/token (never written to any repo).
 3. DOI back-fill: CITATION.cff (doi + identifiers), README badge/citation; commit + push.
 4. Hugging Face: add the DOI and GitHub link to the model card(s), upload, set public.
 5. Write a machine-readable record to <research>/paper/release/release_<version>.json.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.abspath(os.path.join(HERE, "..", "..", "research_topics", "clm_decision_heads_novelty_20260928"))
REPO = os.path.join(RESEARCH, "github", "midm-decision-models")
GH = r"C:\Program Files\GitHub CLI\gh.exe"
GH_REPO = "YeoHoonYun/midm-decision-models"
HF_REPOS = ["yunicro/MiDM-4B-q35-e1-bx"]
ZENODO = "https://zenodo.org/api"


def run(cmd, cwd=REPO, check=True):
    print("  $", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def zenodo_token():
    return open(os.path.join(os.path.expanduser("~"), ".config", "zenodo", "token"), encoding="ascii").read().strip()


def metadata(version):
    z = json.load(open(os.path.join(REPO, ".zenodo.json"), encoding="utf-8"))
    z["version"] = version
    z["publication_date"] = datetime.date.today().isoformat()
    z["related_identifiers"] = [
        {"identifier": f"https://github.com/{GH_REPO}/tree/v{version}", "relation": "isSupplementTo", "resource_type": "software"},
    ] + [{"identifier": f"https://huggingface.co/{r}", "relation": "isSupplementedBy", "resource_type": "other"} for r in HF_REPOS]
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tag, out_dir = f"v{a.version}", os.path.join(RESEARCH, "paper", "release")
    record = {"version": a.version, "tag": tag, "date": datetime.datetime.now().isoformat(timespec="seconds")}

    if run(["git", "status", "--porcelain"]):
        sys.exit("repo has uncommitted changes; commit them first")
    head = run(["git", "rev-parse", "HEAD"])
    if run(["git", "ls-remote", "origin", "main"]).split()[0] != head:
        sys.exit("local main is not pushed")
    archive = os.path.join(out_dir, f"midm-decision-models-{tag}.zip")
    run(["git", "archive", "--format=zip", f"--prefix=midm-decision-models-{a.version}/", "-o", archive, "HEAD"])
    record.update(commit=head, archive=os.path.relpath(archive, RESEARCH), archive_bytes=os.path.getsize(archive))
    meta = metadata(a.version)
    print(json.dumps(meta, indent=1))
    if a.dry_run:
        print(f"[dry-run] archive {archive} ({record['archive_bytes']:,} bytes); nothing published")
        return

    # 1. GitHub: tag, public, release
    run(["git", "tag", "-a", tag, "-m", f"MiDM {a.version}"])
    run(["git", "push", "origin", tag])
    run([GH, "repo", "edit", GH_REPO, "--visibility", "public", "--accept-visibility-change-consequences"])
    notes = os.path.join(out_dir, f"release_notes_{tag}.md")
    run([GH, "release", "create", tag, "--repo", GH_REPO, "--title", f"MiDM {a.version}",
         "--notes-file", notes if os.path.exists(notes) else os.path.join(REPO, "README.md")])
    record["github"] = {"repo": f"https://github.com/{GH_REPO}", "release": f"https://github.com/{GH_REPO}/releases/tag/{tag}"}

    # 2. Zenodo
    h = {"Authorization": f"Bearer {zenodo_token()}"}
    r = requests.post(f"{ZENODO}/deposit/depositions", json={}, headers=h, timeout=60); r.raise_for_status()
    dep = r.json()
    with open(archive, "rb") as f:
        requests.put(f"{dep['links']['bucket']}/{os.path.basename(archive)}", data=f, headers=h, timeout=600).raise_for_status()
    requests.put(f"{ZENODO}/deposit/depositions/{dep['id']}", json={"metadata": meta}, headers=h, timeout=60).raise_for_status()
    r = requests.post(f"{ZENODO}/deposit/depositions/{dep['id']}/actions/publish", headers=h, timeout=120); r.raise_for_status()
    pub = r.json()
    doi, url = pub["doi"], pub["links"].get("record_html") or pub["links"].get("html")
    concept = pub.get("conceptdoi")
    record["zenodo"] = {"id": pub["id"], "doi": doi, "concept_doi": concept, "url": url}
    print(f"[zenodo] DOI {doi} (concept {concept}) {url}", flush=True)

    # 3. DOI back-fill in the repo
    cff = os.path.join(REPO, "CITATION.cff")
    t = open(cff, encoding="utf-8").read()
    t = re.sub(r'(?m)^doi:.*\n', "", t)
    t = t.replace('version: "', f'doi: "{doi}"\nversion: "', 1)
    t = re.sub(r'(?m)^date-released:.*$', f'date-released: "{datetime.date.today().isoformat()}"', t)
    open(cff, "w", encoding="utf-8").write(t)
    rd = os.path.join(REPO, "README.md")
    t = open(rd, encoding="utf-8").read()
    badge = f"[![DOI](https://zenodo.org/badge/DOI/{doi}.svg)](https://doi.org/{doi})"
    if "zenodo.org/badge" not in t:
        t = t.replace("\n", f"\n\n{badge}\n", 1)
    t = t.replace("See `CITATION.cff`. A DOI will be added after the Zenodo release.",
                  f"See `CITATION.cff`. Archived on Zenodo: https://doi.org/{doi}" + (f" (all versions: https://doi.org/{concept})" if concept else ""))
    open(rd, "w", encoding="utf-8").write(t)
    run(["git", "add", "-A"]); run(["git", "commit", "-m", f"DOI {doi} for {tag}"]); run(["git", "push", "origin", "main"])
    record["github"]["doi_commit"] = run(["git", "rev-parse", "HEAD"])

    # 4. Hugging Face: DOI in the card, public
    from huggingface_hub import HfApi
    api = HfApi()
    for repo in HF_REPOS:
        local = os.path.join(RESEARCH, "huggingface", repo.split("/")[1], "README.md")
        card = open(local, encoding="utf-8").read()
        if "## Citation" not in card:
            card += (f"\n## Citation\nCode, results and the audit: https://github.com/{GH_REPO} "
                     f"(Zenodo DOI [{doi}](https://doi.org/{doi})).\n")
            open(local, "w", encoding="utf-8").write(card)
        api.upload_file(path_or_fileobj=local, path_in_repo="README.md", repo_id=repo, commit_message=f"Card: DOI {doi}")
        api.update_repo_settings(repo, private=False)
        record.setdefault("huggingface", []).append({"repo": f"https://huggingface.co/{repo}", "public": True,
                                                     "sha": api.model_info(repo).sha})

    # 5. record
    json.dump(record, open(os.path.join(out_dir, f"release_{tag}.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
