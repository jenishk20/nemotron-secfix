"""Upload hf_export/ to Hugging Face as PRIVATE dataset repos (run scripts/export_hf.py first).

Needs HF_TOKEN (a write token) in .env. Repos go under --namespace: your username, or an
organization so its members (e.g. a teammate) can see them; private repos under a personal
account are visible only to you.
Usage: .venv/bin/python scripts/upload_hf.py --namespace <user-or-org>
"""
import argparse, os

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
load_dotenv(os.path.join(ROOT, ".env"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", required=True)
    a = ap.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])
    print("logged in as", api.whoami()["name"])
    for name in ("cweval-tasks", "secure-code-synth"):
        repo = f"{a.namespace}/{name}"
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=os.path.join(ROOT, "hf_export", name),
                          commit_message="Upload from nemotron-secfix")
        info = api.repo_info(repo, repo_type="dataset")
        print(f"{repo}: private={info.private} -> https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
