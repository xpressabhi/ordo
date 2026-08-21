# capture_wizard.py — interactive tool to build eval/eval.json from photos.
#
# Scans eval/photos/ for images not yet in the manifest, and for each prompts
# you for the question, ground-truth answer, category, difficulty and capture
# conditions. Appends to eval/eval.json incrementally (safe to re-run).
#
# Usage:
#   .venv/bin/python eval/scripts/capture_wizard.py            # interactive
#   .venv/bin/python eval/scripts/capture_wizard.py --import-json answers.json
#
# --import-json: bulk-import from a JSON list:
#   [{"file":"img_001.jpg","question":"...","ground_truth":"...",
#     "category":"restaurant_menu","difficulty":"medium",
#     "conditions":["glare"]}]
#   Any item already present (same file) is skipped.

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHOTOS = os.path.join(REPO, "eval", "photos")
MANIFEST = os.path.join(REPO, "eval", "eval.json")

CATEGORIES = [
    "restaurant_menu", "product_label", "street_sign", "handwriting",
    "receipt", "medicine", "book_spine", "whiteboard", "appliance_display",
]
DIFFICULTIES = ["easy", "medium", "hard"]
CONDITIONS = ["clean", "low_light", "sharp_angle", "glare", "small_text", "partial_occlusion"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}


def load_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            return json.load(f)
    return {"meta": {}, "items": []}


def save_manifest(m):
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
        f.write("\n")


def unassigned_images(items):
    assigned = {os.path.basename(i["path"]) for i in items}
    out = []
    for fn in sorted(os.listdir(PHOTOS)):
        if os.path.splitext(fn)[1].lower() in IMG_EXTS and fn not in assigned:
            out.append(fn)
    return out


def ask(prompt, choices=None, default=None):
    suffix = f" [{default}]" if default else ""
    while True:
        val = input(f"{prompt}{suffix}: ").strip() or (default or "")
        if choices and val not in choices:
            print(f"  choose from: {', '.join(choices)}")
            continue
        return val


def run_interactive(m):
    pending = unassigned_images(m["items"])
    if not pending:
        print("No unassigned photos in eval/photos/. Add more or run with --import-json.")
        return
    print(f"Found {len(pending)} photo(s) to annotate. (Ctrl-C to stop; progress saved per photo.)")
    for fn in pending:
        print(f"\n--- {fn} ---")
        try:
            question = ask("Question you'd ask out loud")
            if not question:
                continue
            ground_truth = ask("Ground-truth answer")
            category = ask("Category", CATEGORIES, default="restaurant_menu")
            difficulty = ask("Difficulty", DIFFICULTIES, default="medium")
            conds = ask("Conditions (comma-sep)", None, default="clean")
            conditions = [c.strip() for c in conds.split(",") if c.strip()] or ["clean"]
            notes = ask("Notes (optional)", None, default="")
        except (KeyboardInterrupt, EOFError):
            print("\nStopped.")
            break
        item = {
            "id": f"img_{len(m['items']) + 1:03d}",
            "path": os.path.join("photos", fn),
            "category": category,
            "question": question,
            "ground_truth": ground_truth,
            "difficulty": difficulty,
            "conditions": conditions,
            "notes": notes,
        }
        m["items"].append(item)
        save_manifest(m)
        print(f"  saved {item['id']}")
    print(f"\nManifest now has {len(m['items'])} items -> {MANIFEST}")


def run_import(m, path):
    with open(path) as f:
        rows = json.load(f)
    existing = {os.path.basename(i["path"]) for i in m["items"]}
    added = 0
    for r in rows:
        fn = os.path.basename(r["file"])
        if fn in existing:
            print(f"skip {fn} (already present)")
            continue
        item = {
            "id": f"img_{len(m['items']) + 1:03d}",
            "path": os.path.join("photos", fn),
            "category": r.get("category", "restaurant_menu"),
            "question": r["question"],
            "ground_truth": r["ground_truth"],
            "difficulty": r.get("difficulty", "medium"),
            "conditions": r.get("conditions", ["clean"]),
            "notes": r.get("notes", ""),
        }
        m["items"].append(item)
        existing.add(fn)
        added += 1
    save_manifest(m)
    print(f"Imported {added} item(s). Total: {len(m['items'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--import-json", help="bulk import from a JSON file")
    args = ap.parse_args()
    m = load_manifest()
    if args.import_json:
        run_import(m, args.import_json)
    else:
        run_interactive(m)


if __name__ == "__main__":
    main()
