"""CAS-deploy the tested HTML without discarding pre-existing user edits."""
import hashlib
import os
from pathlib import Path
import shutil

root = Path("/opt/agillm43_netserve")
stage = root / "quality_20260904T2257Z"
target = root / "agillm_completion.html"
candidate = stage / "agillm_completion.html"
backup = stage / "ui.original.html"
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert digest(target) == "88ea481be9c440913468264c40865fd3abad12419ea063fe329a3e52a17f7e8b"
assert digest(candidate) == "24f46073e8c208aba1d29d6c4d45f39a407a5c821607d2e8efc9e5963222dd6e"
assert not backup.exists()
shutil.copy2(target, backup)
temp = root / "agillm_completion.quality-tmp.html"
assert not temp.exists()
shutil.copy2(candidate, temp)
assert digest(target) == digest(backup), "Concurrent UI edit detected"
os.replace(temp, target)
print("UI deployed SHA256 " + digest(target) + "; original retained")
