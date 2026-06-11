"""
Find probable duplicate character files and emit a reviewable TSV.

A "duplicate cluster" is a set of files in content/characters/ that all
refer to the same entity but use different name variants
(e.g. `bea.md`, `beatrice.md`, `miss_beatrice.md`).

Heuristics (in order of confidence):

  high   - same core stem after stripping honorifics ("the X", "of Y",
           "miss_", "warlord_", etc.) AND mutual mention by `[[Title]]`
  high   - one stem is a token-prefix of another at a `_` boundary
           (e.g. `agatha` is a token-prefix of `agatha_the_residual`)
  medium - mutual `[[Title]]` cross-reference in both bodies, plus one
           title appears as a substring (word-boundary) of the other
  medium - one stem is a strict character-prefix of another (>=3 chars)
           AND there is a unidirectional or bidirectional cross-link

Output: duplicates.tsv (tab-separated) with columns
  canonical_stem    alias_stems    titles    confidence    reason
where:
  canonical_stem = suggested keep-file stem (longest title wins by default)
  alias_stems    = comma-separated stems to merge into canonical
  titles         = pipe-separated list of titles (canonical first)
  confidence     = high | medium
  reason         = short human-readable note

Review the TSV, delete rows you disagree with, then run merge_duplicates.py.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CHAR_DIR = Path("content/characters")
OUT_TSV = Path("duplicates.tsv")

# Honorifics/titles/role-prefixes that aren't part of an entity's identity.
# Used by `normalize_stem` to collapse `miss_beatrice` -> `beatrice`,
# `princess_donut` -> `donut`, `warlord_princess_donut_the_oak_fell` -> `donut`.
HONORIFICS = {
    "mr", "mrs", "ms", "miss", "mistress", "master", "sir", "dame",
    "lady", "lord", "lordess",
    "prince", "princess", "queen", "king", "tsar", "tsarina", "emperor",
    "empress", "sultan", "sultana", "viceroy", "baron", "baroness",
    "count", "countess", "duke", "duchess", "earl", "viscount",
    "marquess", "marchioness",
    "doctor", "dr", "professor", "prof",
    "captain", "admiral", "colonel", "sergeant", "general", "commander",
    "commandant", "magistrate", "judge", "warlord", "war-chief",
    "chieftain", "chief", "high",
    "saint", "st", "father", "mother", "abbot", "abbess", "pope",
    "bishop", "archbishop", "rabbi", "imam", "deacon", "presbyter",
    "demon", "ghost", "spirit", "the",
    "grand", "champion", "best", "supreme", "ultimate", "great",
    "crawler", "warlord",
}

# Connector words that introduce an epithet ("the wise", "of the prism").
# `normalize_stem` cuts at the first occurrence.
CONNECTORS = {"the", "of", "and"}

# Minimum length for a stem to be considered as a prefix-cluster anchor;
# avoids false positives like "an" or "ai" matching everything.
MIN_STEM_LEN = 3


@dataclass
class CharFile:
    path: Path
    stem: str
    title: str
    body: str
    # set of lowercased wikilink targets present in body (without brackets)
    links: set[str] = field(default_factory=set)

    @property
    def title_norm(self) -> str:
        return self.title.strip().lower()


def parse_file(path: Path) -> CharFile | None:
    text = path.read_text(encoding="utf-8")
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.DOTALL)
    if not fm_match:
        return None
    fm = fm_match.group(1)
    body = text[fm_match.end():]
    title_match = re.search(r'^title:\s*"((?:[^"\\\n]|\\.)*)"', fm, flags=re.MULTILINE)
    title = title_match.group(1) if title_match else path.stem
    title = title.replace('\\"', '"').strip()

    links: set[str] = set()
    for m in re.finditer(r"\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]", body):
        links.add(m.group(1).strip().lower())

    return CharFile(path=path, stem=path.stem, title=title, body=body, links=links)


def normalize_stem(stem: str) -> str:
    """Strip honorifics anywhere and cut at first connector ('the' / 'of')."""
    parts = stem.lower().split("_")
    # strip honorifics anywhere
    parts = [p for p in parts if p not in HONORIFICS]
    # cut at first connector
    for i, p in enumerate(parts):
        if p in CONNECTORS:
            parts = parts[:i]
            break
    # collapse repeats ("donut_donut" -> "donut")
    deduped: list[str] = []
    for p in parts:
        if not deduped or deduped[-1] != p:
            deduped.append(p)
    return "_".join(deduped) or stem.lower()


def title_to_link_key(title: str) -> str:
    """How a wikilink to this title would render in lowercased form."""
    return title.strip().lower()


def cluster_high_confidence(files: list[CharFile]) -> list[tuple[str, list[CharFile], str]]:
    """
    Group files by normalized stem. Each group with >1 member is a cluster.
    Returns list of (reason, members, kind).
    """
    groups: dict[str, list[CharFile]] = defaultdict(list)
    for f in files:
        groups[normalize_stem(f.stem)].append(f)

    clusters: list[tuple[str, list[CharFile], str]] = []
    for key, members in groups.items():
        if len(members) > 1 and len(key) >= MIN_STEM_LEN:
            clusters.append((f"same-core-stem:{key}", members, "high"))
    return clusters


def cluster_token_prefix(files: list[CharFile]) -> list[tuple[str, list[CharFile], str]]:
    """
    Cluster files where one stem is a token-boundary prefix of another:
    `agatha` <- `agatha_the_residual`, `architect_houston` <- `architect_houston_of_the_madness`.
    """
    stems = sorted({f.stem for f in files}, key=len)
    by_stem = {f.stem: f for f in files}
    # union-find by stem
    parent: dict[str, str] = {s: s for s in stems}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for short in stems:
        if len(short) < MIN_STEM_LEN:
            continue
        prefix = short + "_"
        for long in stems:
            if long != short and long.startswith(prefix):
                union(short, long)

    groups: dict[str, list[CharFile]] = defaultdict(list)
    for s in stems:
        groups[find(s)].append(by_stem[s])

    clusters: list[tuple[str, list[CharFile], str]] = []
    for key, members in groups.items():
        if len(members) > 1:
            clusters.append((f"token-prefix:{key}", members, "high"))
    return clusters


def cluster_cross_reference(
    files: list[CharFile], already_clustered: set[str]
) -> list[tuple[str, list[CharFile], str]]:
    """
    For files NOT already in a high-confidence cluster, find pairs that
    mutually wikilink each other AND where one title is a word-boundary
    substring of the other.
    """
    eligible = [f for f in files if f.stem not in already_clustered]
    by_title = {f.title_norm: f for f in eligible}
    clusters: list[tuple[str, list[CharFile], str]] = []
    seen: set[tuple[str, str]] = set()

    for a in eligible:
        for b_title in a.links:
            b = by_title.get(b_title)
            if b is None or b.stem == a.stem:
                continue
            # mutual reference?
            if a.title_norm not in b.links:
                continue
            # title containment (whole word)?
            t1, t2 = a.title_norm, b.title_norm
            short_t, long_t = (t1, t2) if len(t1) <= len(t2) else (t2, t1)
            pattern = rf"\b{re.escape(short_t)}\b"
            if not re.search(pattern, long_t):
                continue
            pair = tuple(sorted([a.stem, b.stem]))
            if pair in seen:
                continue
            seen.add(pair)
            clusters.append((f"mutual-ref+title-contains:{short_t}", [a, b], "medium"))
    return clusters


def cluster_char_prefix(
    files: list[CharFile], already_clustered: set[str]
) -> list[tuple[str, list[CharFile], str]]:
    """
    For files NOT already clustered, link short_stem -> long_stem when
    short_stem is a strict character prefix of long_stem (>=3 chars)
    AND one mentions the other's title in a wikilink.

    Catches `bea` -> `beatrice`.
    """
    eligible = [f for f in files if f.stem not in already_clustered]
    by_stem = {f.stem: f for f in eligible}
    by_title = {f.title_norm: f for f in eligible}
    stems = sorted(by_stem.keys(), key=len)

    pairs: dict[str, set[str]] = defaultdict(set)
    for short in stems:
        if len(short) < MIN_STEM_LEN or "_" in short:
            continue
        for long in stems:
            if long == short or "_" in long:
                continue
            if not long.startswith(short):
                continue
            a, b = by_stem[short], by_stem[long]
            if a.title_norm in b.links or b.title_norm in a.links:
                pairs[short].add(long)

    clusters: list[tuple[str, list[CharFile], str]] = []
    used: set[str] = set()
    for short, longs in pairs.items():
        members = [by_stem[short]] + [by_stem[l] for l in sorted(longs)]
        stems_in = {m.stem for m in members}
        if stems_in & used:
            continue
        used |= stems_in
        clusters.append((f"char-prefix+ref:{short}", members, "medium"))
    return clusters


def merge_overlapping(
    clusters: list[tuple[str, list[CharFile], str]]
) -> list[tuple[str, list[CharFile], str]]:
    """Merge any clusters that share a member."""
    parent: dict[str, str] = {}
    info: dict[str, tuple[str, str]] = {}  # stem -> (reason, confidence)
    members_by_stem: dict[str, CharFile] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for reason, members, conf in clusters:
        for m in members:
            if m.stem not in parent:
                parent[m.stem] = m.stem
                members_by_stem[m.stem] = m
            info.setdefault(m.stem, (reason, conf))
        first = members[0].stem
        for m in members[1:]:
            union(first, m.stem)

    groups: dict[str, list[str]] = defaultdict(list)
    for s in parent:
        groups[find(s)].append(s)

    result: list[tuple[str, list[CharFile], str]] = []
    for root, stems in groups.items():
        if len(stems) < 2:
            continue
        members = [members_by_stem[s] for s in stems]
        reasons = sorted({info[s][0] for s in stems})
        confs = {info[s][1] for s in stems}
        conf = "high" if confs == {"high"} else "medium"
        result.append((" | ".join(reasons), members, conf))
    return result


def pick_canonical(members: list[CharFile]) -> CharFile:
    """Longest title wins; tiebreak on body length (more content = canonical)."""
    return max(members, key=lambda f: (len(f.title), len(f.body)))


def main() -> int:
    if not CHAR_DIR.is_dir():
        print(f"error: {CHAR_DIR} not found (run from repo root)", file=sys.stderr)
        return 2

    files: list[CharFile] = []
    for p in sorted(CHAR_DIR.glob("*.md")):
        cf = parse_file(p)
        if cf is not None:
            files.append(cf)
    print(f"parsed {len(files)} character files")

    high = cluster_high_confidence(files) + cluster_token_prefix(files)
    high_stems = {m.stem for _, ms, _ in high for m in ms}
    medium = cluster_cross_reference(files, high_stems) + cluster_char_prefix(files, high_stems)

    merged = merge_overlapping(high + medium)
    merged.sort(key=lambda c: (c[2] != "high", pick_canonical(c[1]).stem))
    print(f"found {len(merged)} duplicate clusters "
          f"({sum(1 for c in merged if c[2] == 'high')} high, "
          f"{sum(1 for c in merged if c[2] == 'medium')} medium)")

    with OUT_TSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["canonical_stem", "alias_stems", "titles", "confidence", "reason"])
        for reason, members, conf in merged:
            canon = pick_canonical(members)
            aliases = [m for m in members if m.stem != canon.stem]
            titles = "|".join([canon.title] + [a.title for a in aliases])
            w.writerow([
                canon.stem,
                ",".join(a.stem for a in aliases),
                titles,
                conf,
                reason,
            ])
    print(f"wrote {OUT_TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
