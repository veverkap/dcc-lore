"""
Merge duplicate character files into one canonical file per cluster.

Reads duplicates.tsv (produced by find_duplicates.py). For each row:

  1. Reads the canonical file and every alias file.
  2. Concatenates body bullets, dedupes (lower+strip+strip-wikilink-brackets),
     and rewrites the canonical body.
  3. Adds every alias title to the canonical file's `aliases:` frontmatter
     field. (Quartz's alias-redirects plugin will redirect old wikilinks
     and old URLs to the canonical page automatically.)
  4. Deletes the alias source files.

Usage:
    uv run python merge_duplicates.py            # dry run, prints plan
    uv run python merge_duplicates.py --apply    # actually merges + deletes

Safety:
  - Refuses to run if duplicates.tsv has a row whose canonical_stem appears
    as an alias in another row (would delete the canonical mid-merge).
  - Refuses to run if any listed file is missing.
  - Pass --backup-dir DIR to copy originals before deletion.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

CHAR_DIR = Path("content/characters")
TSV_PATH = Path("duplicates.tsv")


@dataclass
class Row:
    canonical_stem: str
    alias_stems: list[str]
    titles: list[str]  # canonical first
    confidence: str
    reason: str


@dataclass
class Parsed:
    pre_fm: str         # everything before first ---
    fm: str             # frontmatter body (between --- markers)
    post_fm_sep: str    # closing "---\n"
    body: str           # rest of file


def parse(path: Path) -> Parsed:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"\A(---\s*\n)(.*?)(\n---\s*\n)", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"{path} has no frontmatter")
    return Parsed(
        pre_fm=m.group(1),
        fm=m.group(2),
        post_fm_sep=m.group(3),
        body=text[m.end():],
    )


def normalize_for_dedup(line: str) -> str:
    s = line.strip().lstrip("*-•").strip()
    s = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", r"\1", s)  # strip wikilink brackets
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def dedupe_bullets(bodies: list[str]) -> list[str]:
    """Flatten all bullets from given bodies, dedupe by normalized content."""
    seen: set[str] = set()
    out: list[str] = []
    for body in bodies:
        for raw in body.splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):  # skip headers
                continue
            if stripped.lower().startswith(("here is the consolidated",
                                            "here is the cleaned",
                                            "here are the consolidated")):
                continue
            key = normalize_for_dedup(stripped)
            if not key or key in seen:
                continue
            seen.add(key)
            # normalize bullet marker to "*"
            if stripped[0] in "•-*":
                stripped = "* " + stripped[1:].lstrip()
            else:
                stripped = "* " + stripped
            out.append(stripped)
    return out


# --- frontmatter aliases handling ---------------------------------------------
# We support YAML inline list (aliases: [a, b]) and block list (one per line).
# We rewrite to inline form because it's a single deterministic line we control.

_ALIASES_LINE_RE = re.compile(
    r"^aliases:[ \t]*(?P<rest>.*)$", flags=re.MULTILINE
)


def parse_existing_aliases(fm: str) -> list[str]:
    """Best-effort extraction of an existing aliases list from frontmatter."""
    m = _ALIASES_LINE_RE.search(fm)
    if not m:
        return []
    rest = m.group("rest").strip()
    if rest.startswith("[") and rest.endswith("]"):
        inner = rest[1:-1]
        return [_unquote(s).strip() for s in inner.split(",") if s.strip()]
    if rest == "":
        # block form: collect following lines starting with "- "
        block: list[str] = []
        tail = fm[m.end():]
        for line in tail.splitlines():
            if re.match(r"^\s*-\s+", line):
                block.append(_unquote(line.strip()[2:].strip()))
            elif line.strip() == "":
                continue
            else:
                break
        return block
    # single scalar
    return [_unquote(rest)]


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def _quote(s: str) -> str:
    if any(c in s for c in [":", "#", ",", "[", "]", "{", "}", "&", "*",
                            "!", "|", ">", "'", '"', "%", "@", "`"]) or " " in s:
        escaped = s.replace('"', '\\"')
        return f'"{escaped}"'
    return s


def set_aliases(fm: str, aliases: list[str]) -> str:
    # dedupe case-insensitively while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for a in aliases:
        k = a.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        uniq.append(a.strip())

    new_line = "aliases: [" + ", ".join(_quote(a) for a in uniq) + "]"

    m = _ALIASES_LINE_RE.search(fm)
    if not m:
        # append before end
        if fm.endswith("\n"):
            return fm + new_line
        return fm + "\n" + new_line

    # Determine how many lines to replace (handle block form)
    start = m.start()
    end = m.end()
    if m.group("rest").strip() == "":
        tail = fm[end:]
        consumed = 0
        for line in tail.splitlines(keepends=True):
            if re.match(r"^\s*-\s+", line) or line.strip() == "":
                consumed += len(line)
            else:
                break
        end += consumed
    return fm[:start] + new_line + fm[end:]


# --- TSV ---------------------------------------------------------------------


def load_tsv(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            canon = r["canonical_stem"].strip()
            aliases = [s.strip() for s in r["alias_stems"].split(",") if s.strip()]
            titles = [t.strip() for t in r["titles"].split("|") if t.strip()]
            rows.append(Row(
                canonical_stem=canon,
                alias_stems=aliases,
                titles=titles,
                confidence=r.get("confidence", "").strip(),
                reason=r.get("reason", "").strip(),
            ))
    return rows


def validate(rows: list[Row]) -> list[str]:
    errors: list[str] = []
    alias_to_canon: dict[str, str] = {}
    seen_canons: set[str] = set()
    for r in rows:
        if not r.canonical_stem:
            errors.append(f"row missing canonical_stem: {r}")
            continue
        if not r.alias_stems:
            errors.append(f"row {r.canonical_stem!r} has no aliases")
            continue
        if r.canonical_stem in seen_canons:
            errors.append(f"canonical stem repeated: {r.canonical_stem!r}")
        seen_canons.add(r.canonical_stem)
        for a in r.alias_stems:
            if a == r.canonical_stem:
                errors.append(f"alias equals canonical in row {r.canonical_stem!r}")
            if a in alias_to_canon:
                errors.append(f"alias {a!r} listed in two rows "
                              f"({alias_to_canon[a]!r} and {r.canonical_stem!r})")
            alias_to_canon[a] = r.canonical_stem

    # canonical of one row must not be alias of another
    for r in rows:
        if r.canonical_stem in alias_to_canon and alias_to_canon[r.canonical_stem] != r.canonical_stem:
            errors.append(
                f"canonical {r.canonical_stem!r} is also an alias of "
                f"{alias_to_canon[r.canonical_stem]!r} (would delete mid-merge)"
            )

    for r in rows:
        canon_path = CHAR_DIR / f"{r.canonical_stem}.md"
        if not canon_path.exists():
            errors.append(f"missing canonical file: {canon_path}")
        for a in r.alias_stems:
            ap = CHAR_DIR / f"{a}.md"
            if not ap.exists():
                errors.append(f"missing alias file: {ap}")
    return errors


# --- main --------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually write changes and delete alias files")
    ap.add_argument("--tsv", default=str(TSV_PATH))
    ap.add_argument("--backup-dir", default=None,
                    help="copy each alias file here before deletion")
    args = ap.parse_args()

    tsv = Path(args.tsv)
    if not tsv.exists():
        print(f"error: {tsv} not found", file=sys.stderr)
        return 2

    rows = load_tsv(tsv)
    print(f"loaded {len(rows)} clusters from {tsv}")

    errors = validate(rows)
    if errors:
        for e in errors:
            print(f"  ! {e}", file=sys.stderr)
        return 2

    backup_dir = Path(args.backup_dir) if args.backup_dir else None
    if backup_dir and args.apply:
        backup_dir.mkdir(parents=True, exist_ok=True)

    total_deleted = 0
    for r in rows:
        canon_path = CHAR_DIR / f"{r.canonical_stem}.md"
        alias_paths = [CHAR_DIR / f"{a}.md" for a in r.alias_stems]

        canon = parse(canon_path)
        parsed_aliases = [parse(p) for p in alias_paths]

        merged_bullets = dedupe_bullets([canon.body] + [a.body for a in parsed_aliases])

        # Canonical title is first in r.titles; subsequent are alias display names.
        canonical_title = r.titles[0] if r.titles else r.canonical_stem
        alias_titles = r.titles[1:] if len(r.titles) > 1 else []
        existing = parse_existing_aliases(canon.fm)
        new_aliases = existing + alias_titles
        # also include the alias stems themselves so URL redirects work
        new_aliases += [a.replace("_", " ") for a in r.alias_stems]
        new_fm = set_aliases(canon.fm, new_aliases)

        # body: title header + bullets
        header = f"# {canon_path.stem}"
        new_body = "\n" + header + "\n\n" + "\n".join(merged_bullets) + "\n"
        new_text = canon.pre_fm + new_fm + canon.post_fm_sep + new_body

        print(f"\n[{r.confidence}] {canon_path.name}  <- "
              f"{', '.join(p.name for p in alias_paths)}")
        print(f"    title:   {canonical_title}")
        print(f"    aliases: {new_aliases}")
        print(f"    bullets: {len(merged_bullets)} unique")

        if args.apply:
            if backup_dir:
                for p in [canon_path] + alias_paths:
                    shutil.copy2(p, backup_dir / p.name)
            canon_path.write_text(new_text, encoding="utf-8")
            for p in alias_paths:
                p.unlink()
                total_deleted += 1

    if args.apply:
        print(f"\nmerged {len(rows)} clusters, deleted {total_deleted} files")
    else:
        print(f"\ndry run: re-run with --apply to write changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
