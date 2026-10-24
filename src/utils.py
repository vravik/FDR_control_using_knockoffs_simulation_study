from pathlib import Path
import numpy as np
import json
import hashlib

def rng_for(seed, scenario_id, rep, task):
    """Stable stream for one task, independent of loop order or worker.
    
    Hash the UTF-8 JSON list [scenario_id, rep, task] with SHA-256. Use its
    first 16 bytes as four little-endian uint32 words after the master seed
    in a SeedSequence. Unlike Python's hash(), this mapping is stable.
    """
    label = json.dumps([scenario_id, rep, task], separators=(",", ":"))
    digest = hashlib.sha256(label.encode()).digest()[:16]
    words = np.frombuffer(digest, dtype="<u4").tolist()
    return np.random.default_rng(np.random.SeedSequence([seed, *words]))


def save_csv(frame, path):
    """Publish a complete table by renaming a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
