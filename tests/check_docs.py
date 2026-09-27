"""Documentation integrity check: broken links + structure audit.

Verifies every relative markdown link/image in the repo resolves to a
real file, and prints a structure summary (which docs exist, which are
referenced from README).
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
md_files = sorted(REPO.rglob("*.md"))
md_files = [f for f in md_files if ".git" not in f.parts]

broken, checked = [], 0
for f in md_files:
    text = f.read_text(encoding="utf-8")
    for match in re.finditer(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
        target = match.group(1)
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        checked += 1
        resolved = (f.parent / target).resolve()
        if not resolved.exists():
            broken.append((f.relative_to(REPO), target))

print(f"markdown files : {len(md_files)}")
print(f"local links    : {checked} checked")
if broken:
    print("\nBROKEN LINKS:")
    for f, t in broken:
        print(f"  {f} -> {t}")
else:
    print("broken links   : none")

# structure audit: a doc is "linked" if any other markdown resolves to it
referenced = set()
for f in md_files:
    text = f.read_text(encoding="utf-8")
    for match in re.finditer(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
        t = match.group(1)
        if not t.startswith(("http://", "https://", "mailto:")):
            referenced.add((f.parent / t).resolve())

print("\ndocs present (docs/ + machine/ + deploy/):")
for d in sorted(REPO.glob("docs/*.md")) + sorted(REPO.glob("machine/**/*.md")) \
        + sorted(REPO.glob("deploy/**/*.md")):
    name = d.relative_to(REPO).as_posix()
    linked = d.resolve() in referenced
    print(f"  {'[linked]' if linked else '[ORPHAN]'} {name}")

top = [p.name for p in REPO.iterdir()]
print("\nrepo root:", ", ".join(sorted(top)))
sys.exit(1 if broken else 0)
