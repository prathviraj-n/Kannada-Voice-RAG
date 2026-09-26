"""
download_models.py

Downloads both embedding models into the exact local folder layout that
src/config.py's find_local_model() expects — WITHOUT importing anything
from the src/ package, since src/config.py forces HF_HUB_OFFLINE=1 the
moment it's imported, which would make any download attempt fail.

Run this ONCE from the project root, on a machine/box that has internet
access.

USAGE:
  python download_models.py
"""
import os

# Make sure nothing forces offline mode for this process.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from huggingface_hub import snapshot_download

ROOT = os.path.abspath(os.path.dirname(__file__))

TARGETS = [
    ("l3cube-pune/kannada-sentence-bert-nli",
     os.path.join(ROOT, "models", "l3cube-pune", "kannada-sentence-bert-nli")),

    ("intfloat/multilingual-e5-small",
     os.path.join(ROOT, "backend", "models", "intfloat", "multilingual-e5-small")),
]

for repo_id, local_dir in TARGETS:
    print(f"\nDownloading {repo_id} -> {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    cfg = os.path.join(local_dir, "config.json")
    print(f"  {'OK' if os.path.exists(cfg) else 'MISSING'} config.json present: {os.path.exists(cfg)}")

print("\nDone. Verify with:")
print("  find models backend/models -iname 'config.json'")
