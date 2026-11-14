#!/usr/bin/env python3
"""Long passages for the five -gemma4 datasets (bt_gemma4.py): stretches of whole consecutive paragraphs of the same
Dutch sources with more than MIN_TOKENS and at most MAX_TOKENS Gemma 4 tokens, each translated whole into English by
Gemma 4 31B-it (the prompt and checks of bt_gemma4.py) and back into Dutch (the round trip), as extra rows of the
existing splits with the columns of the existing rows. Decided on 14 Nov 2026: extra rows in the existing splits (no
column or config of their own), no paragraph shared with an existing passage of the same document (a document where
that is impossible gives none), the split per document as in the existing rows, FineWeb-2 only from the crawls up to
and including CC-MAIN-2023-23; MAX_TOKENS = 2,048 from the length trial (a whole passage up to 2,048 tokens translated
as well as one in pieces of ~512).

    python bt_long.py candidates --data DIR --tokenizer TOK --wordlist W --idioms I --pool-reports R ... \\
        --testsets DIR --out DIR [--sources fineweb,...] [--want 7500] [--scan-all dpc,handelingen] [--processes 32]
    python bt_long.py judge-shards --candidates DIR --workers 4 --out DIR
    python bt_long.py choose --candidates DIR --scores DIR --out DIR [--per-source 5000 | --budget budget.json]
    python bt_long.py smokeplan --candidates DIR --out DIR
    python bt_long.py budget --smoke DIR --selection DIR --out budget.json [--worker-hours 96]
    python bt_long.py english --selection DIR --shards DIR --out DIR
    python bt_long.py build --selection DIR --existing DIR --en-shards DIR --rt-shards DIR --out DIR

candidates (CPU): --data holds the -gemma4 datasets (bt_gemma4.py fetch --roundtrip layout) at the revisions the new
rows will join; their passages and context turns mark the paragraphs that are taken. Per source, the documents of the
original source (select_passages.SOURCES, read straight from Hugging Face, row group by row group in an order fixed
by SEED) that pass the selection's document filters (and for FineWeb-2 the crawl bound); per document its blocks as
the selection reads them (the turns of a speaker in the Handelingen, the document otherwise; in council documents a
speaker label stripped and a stretch never running over one); the paragraphs of a block that hold a piece of an
existing passage are taken (a whole context turn is taken; a document with a piece found in none of its paragraphs is
skipped); a paragraph is usable when it is free, not a table row or address, at most MAX_TOKENS long and passes the
selection's extraction rules on its own. In a run of usable paragraphs a stretch begins at one of up to TRIES starts
(random) and takes as many whole paragraphs as fit MAX_TOKENS; it must then have more than MIN_TOKENS tokens after the
two edits of the selection (marked headings on a line of their own, an empty line between paragraphs), a prose share
of at least MIN_PROSE_SHARE of its tokens, pass every filter of the selection on the whole text
(select_passages.Builder.checks; a failure that the heading repair cures passes as repaired) and the border rules
(open_before, continues_after). At most MAX_PER_DOC per document. In the order read, a candidate is kept unless it
shares a 13-gram of words with a test set, an existing passage (any source) or a candidate kept before (exact
duplicates included), its FineWeb-2 host has HOST_CAP of the target already, or its length bin (BINS) has --want
already; a source is read until every bin is full (--scan-all: to its end, counting what is there). Written:
<source>.parquet, report.json (counts per source, bin and filter, what is available, the checks of the
reconstruction against the existing rows).

judge-shards: the candidates as judge.py's input (task source_topic_origin, as the existing passages): <out>/w<k>/.
choose: the candidates with a source score >= 0.5 (FineWeb-2: origin score >= 0.5 as well), in their order, up to
--per-source (or the numbers of --budget) per source, the same number from every length bin (a bin short of it leaves
its remainder to the others, the shorter first); FineWeb-2's topics capped at TOPIC_CAP while enough are left;
<out>/<source>/data/{train,validation}-00000-of-00001.parquet for bt_gemma4.py plan, and selection.json.
smokeplan: two plans of a few units per length class (nl->en, and the round trip's classes with Dutch text standing in
for the English: the shapes, the memory and the speed are what the smoke test measures), from the candidates.
budget: from the smoke test's logs the steady output tokens per second and the start per class, the estimate of the
whole run (nl->en and the round trip) for a number of passages per source; the largest number up to the target
whose estimate stays within --worker-hours.
english: the English of the nl->en shards as the round trip's input (bt_gemma4.py plan --text en).
build: the new rows of every source and split with the schema and the column order of the existing files, and
build.json with their numbers.
"""

import argparse
import collections
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent, HERE):
    sys.path.insert(0, str(_path))

import bt_gemma4 as bg  # noqa: E402

ge = bg.ge

MIN_TOKENS = 512                # a long passage has more than this many Gemma 4 tokens ...
MAX_TOKENS = 2048               # ... and at most this many (nl, after the edits)
EVAL_PERMILLE = 20              # select_passages.py --eval-permille of the existing selection
# FineWeb-2 (Yeb, 14 Nov 2026): only the crawls up to and including CC-MAIN-2023-23
FINEWEB_LAST_DUMP = (2023, 23)
SEED = "bt-long/0"
TARGET = 5000                   # passages per source aimed at (the budget may lower it)
OVERSAMPLE = 1.5                # candidates per passage aimed at: the judge's threshold drops some
MIN_PROSE_SHARE = 0.6           # of a stretch's tokens in prose paragraphs, as in the length trial
TRIES = 8                       # starting paragraphs tried per run of usable paragraphs
# Length bins (upper bounds, Gemma 4 tokens of nl): the passages of a source are spread evenly over them, because the
# documents' own lengths would put most stretches just above MIN_TOKENS (FineWeb-2, a trial of 3,000 documents on 14
# Nov 2026: median 616, 95th percentile 1,197), and the long rows are there to train on long inputs up to MAX_TOKENS
BINS = (896, 1280, 1664, 2048)
MAX_PER_DOC = 1
HOST_CAP = 0.005                # FineWeb-2: no host above this share of the target (mainset.HOST_CAP)
TOPIC_CAP = 0.15                # FineWeb-2: no topic above this share while enough are left (mainset.TOPIC_CAP)
SOURCE_MIN = ORIGIN_MIN = 0.5   # mainset.SOURCE_MIN, ORIGIN_MIN; the origin score only for FineWeb-2
NGRAM = 13
SOURCE_KEYS = ("fineweb", "handelingen", "raad", "dpc", "rechtspraak")
# yhavinga/dutch-handelingen-modern: the selection's revision abdba14 is gone from the Hub; its history was squashed
# into ef8e7e5, against which the published passages were checked (README)
HANDELINGEN_REV = "ef8e7e53159cb631f6c24c40b8b8288c68a487d9"
# The columns each source's documents are read with: those of the selection, and what the datasets' columns need
READ_COLUMNS = {
    "fineweb": ["id", "text", "url", "dump", "date"],
    "handelingen": ["id", "chamber", "session_year", "meeting_number", "date", "title", "turns"],
    "gptnl": ["doc_id", "source_id", "text", "content_quality", "content_integrity", "content_type", "content_ratio",
              "pii_presence"],
}
MODEL = "gemma4-31b-bt-long"
RT_MODEL = "gemma4-31b-rt-long"
DUMP = re.compile(r"^CC-MAIN-(\d{4})-(\d{2})$")


def stable_hash(text):
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


def ngram_hashes(text, n=NGRAM):
    """Stable 64-bit hashes of the 13-grams of words of a text (select_passages.ngrams with a hash that does not change
    between processes)"""
    import select_passages as sp

    toks = [t.lower() for t in sp.NGRAM_WORD.findall(text)]
    return {stable_hash(" ".join(toks[i:i + n])) for i in range(len(toks) - n + 1)}


def norm(text):
    return " ".join(text.split())


def dump_ok(dump):
    """Whether a FineWeb-2 dump is a crawl up to FINEWEB_LAST_DUMP"""
    m = DUMP.match(dump or "")
    if not m:
        raise SystemExit(f"a FineWeb-2 dump that is no CC-MAIN-YYYY-WW: {dump!r}")
    return (int(m.group(1)), int(m.group(2))) <= FINEWEB_LAST_DUMP


def split_name(source, doc_id):
    """The split of a new document by the selection's rule (select_passages.split_of): eval is validation"""
    import select_passages as sp

    return "validation" if sp.split_of(source, doc_id, EVAL_PERMILLE) == "eval" else "train"


def parse_id(pid):
    """(source, doc_id, block, passage) of an id source:doc:block:k, or (source, doc_id, block, None) of a context
    turn source:doc:t<j> (the doc id may hold colons)"""
    source, rest = pid.split(":", 1)
    head, last = rest.rsplit(":", 1)
    if re.fullmatch(r"t\d+", last):
        return source, head, last, None
    doc_id, block = head.rsplit(":", 1)
    return source, doc_id, block, last


# ---------------------------------------------------------------------------------------------------- existing

# The datasets the long rows join: key -> Hugging Face dataset (bt_gemma4.ROUNDTRIP_SOURCES's, at their heads)
DATASETS = {k: repo for k, (repo, _) in bg.ROUNDTRIP_SOURCES.items()}


def fetch(args):
    """The -gemma4 datasets at their current heads (or those of --heads, heads.json of an earlier fetch), in the
    layout of bt_gemma4.py fetch --roundtrip; heads.json records the revisions"""
    from huggingface_hub import HfApi, snapshot_download

    out = Path(args.out)
    pinned = json.loads(Path(args.heads).read_text()) if args.heads else {}
    heads = {}
    for key, repo in DATASETS.items():
        revision = pinned.get(key) or HfApi().dataset_info(repo).sha
        patterns = ["data/*.parquet"] + (["context_turns/*.parquet"] if key == "handelingen" else [])
        snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=str(out / key), allow_patterns=patterns)
        heads[key] = revision
        print(f"FETCH {json.dumps({'source': key, 'repo': repo, 'revision': revision})}", flush=True)
    (out / "heads.json").write_text(json.dumps(heads, indent=1))

def existing_rows(data, source):
    """[(id, nl, split, folder)] of one source's -gemma4 dataset (data/ and, for the Handelingen, context_turns/)"""
    import pyarrow.parquet as pq

    out = []
    folders = ["data"] + (["context_turns"] if source == "handelingen" else [])
    for folder in folders:
        for split in bg.SPLITS:
            for path in bg.source_files(data, source, folder, split):
                t = pq.read_table(path, columns=["id", "nl", "tokens_nl"]).to_pydict()
                for i, nl, n in zip(t["id"], t["nl"], t["tokens_nl"]):
                    if n > MIN_TOKENS:
                        raise SystemExit(f"{i}: {n} tokens, an existing row above {MIN_TOKENS}: tokens_nl would not "
                                         "mark the long rows")
                    out.append((i, nl, split, folder))
    return out


def taken_pieces(rows):
    """{(doc_id, block): {"pieces": [normalized piece], "whole": bool}} and {doc_id: split}: the pieces (paragraphs or
    runs of sentences, split on the empty lines of the edit) of every existing passage, a context turn as a whole"""
    taken = collections.defaultdict(lambda: {"pieces": [], "whole": False})
    splits = {}
    for pid, nl, split, folder in rows:
        _, doc_id, block, k = parse_id(pid)
        if splits.setdefault(doc_id, split) != split:
            raise SystemExit(f"document {doc_id} stands in both splits")
        entry = taken[(doc_id, block)]
        if folder == "context_turns" or k is None:
            entry["whole"] = True
        entry["pieces"] += [norm(p) for p in nl.split("\n\n") if p.strip()]
    return dict(taken), splits


def locate(pieces, paragraphs):
    """The indices of the paragraphs that hold a piece (every paragraph that holds it), or None when a piece is in no
    paragraph"""
    normed = [norm(p) for p in paragraphs]
    out = set()
    for piece in pieces:
        found = [i for i, p in enumerate(normed) if piece in p or (p and p in piece)]
        if not found:
            return None
        out.update(found)
    return out


# ---------------------------------------------------------------------------------------------------- stretches

class Checker:
    """The selection's filters (select_passages.Builder.checks) and idioms, without the TranslateGemma tokenizer that
    Builder counts with"""

    def __init__(self, wordlist, idioms, frequent):
        import langid

        import select_passages as sp

        self.builder = sp.Builder.__new__(sp.Builder)
        self.builder.langid = langid
        langid.classify("init")
        with open(wordlist, encoding="utf-8") as f:
            self.builder.wordset = {line.strip().lower() for line in f if line.strip()}
        with open(idioms, encoding="utf-8") as f:
            self.matcher = sp.IdiomMatcher(json.load(f))
        self.frequent = frequent

    @property
    def wordset(self):
        return self.builder.wordset

    def checks(self, source, text):
        return self.builder.checks(source, text)[0]

    def idioms(self, source, text):
        """(idioms, idiom_score) as the selection's: matches per 100 words, without the idioms its pool found too
        frequent (drop_frequent_idioms)"""
        import select_passages as sp

        hits = [i for i in self.matcher.find(text) if i not in self.frequent.get(source, ())]
        return hits, round(100 * len(hits) / max(len(sp.words_of(text)), 1), 4)


def usable_paragraphs(paragraphs, counts, taken, checker):
    """Per paragraph whether a stretch may hold it: free, not a table row or address, at most MAX_TOKENS, and through
    the extraction rules on its own"""
    import select_passages as sp

    out = []
    for i, (p, n) in enumerate(zip(paragraphs, counts)):
        ok = i not in taken and n <= MAX_TOKENS and "|" not in p and not sp.URL.search(p)
        out.append(ok and not sp.extraction_errors(p, checker.wordset))
    return out


def runs(usable, labels):
    """[(first, end)] of the runs of usable paragraphs; a speaker label starts a new run"""
    out, start = [], None
    for i, ok in enumerate(list(usable) + [False]):
        if start is not None and (not ok or (i < len(labels) and labels[i])):
            out.append((start, i))
            start = None
        if ok and start is None:
            start = i
    return out


def border_fails(paragraphs, labels, first, end):
    """The selection's rules on the paragraphs around a stretch (Builder.context_fails); a label is a clean border"""
    import select_passages as sp

    fails = []
    if first > 0 and not labels[first] and sp.open_before(paragraphs[first - 1]):
        fails.append("open_before")
    if end < len(paragraphs) and not labels[end] and sp.LOWER_START.match(paragraphs[end]):
        fails.append("continues_after")
    return fails


def edited(text):
    """The two edits of the selection before translation (mainset.transformed): (nl, headings moved)"""
    import select_passages as sp

    fixed, n = sp.repair_headings(text)
    return sp.paragraph_breaks(fixed), n


def bin_of(tokens):
    for b in BINS:
        if tokens <= b:
            return b
    raise ValueError(f"{tokens} tokens: above the largest bin")


def find_stretch(source, paragraphs, labels, counts, usable, count_text, checker, rng, counters, target=None,
                 tries=TRIES):
    """One stretch of a block or None: {first, end, text, nl, tokens, headings_repaired, repaired}"""
    import select_passages as sp

    target = target or MAX_TOKENS
    prose = [sp.is_prose(p) for p in paragraphs]
    for a, b in runs(usable, labels):
        if sum(counts[a:b]) + (b - a - 1) <= MIN_TOKENS:
            continue
        starts = list(range(a, b))
        rng.shuffle(starts)
        for s in starts[:tries]:
            j, total = s, 0
            while j < b and total + counts[j] + (j > s) <= target:
                total += counts[j] + (j > s)
                j += 1
            if total <= MIN_TOKENS:
                counters["start_too_short"] += 1
                continue
            while j > s:
                text = "\n".join(paragraphs[s:j])
                nl, moved = edited(text)
                n = count_text(nl)
                if n <= min(target, MAX_TOKENS):
                    break
                j -= 1
            if j == s or n <= MIN_TOKENS:
                counters["start_too_short"] += 1
                continue
            if sum(c for c, p in zip(counts[s:j], prose[s:j]) if p) < MIN_PROSE_SHARE * sum(counts[s:j]):
                counters["prose_share"] += 1
                continue
            fails = checker.checks(source, text) + border_fails(paragraphs, labels, s, j)
            repaired = False
            if fails:
                fixed, k = sp.repair_headings(text)
                if k and not checker.checks(source, fixed) and not border_fails(paragraphs, labels, s, j):
                    repaired, fails = True, []
            if fails:
                for f in fails:
                    counters[f"fail_{f}"] += 1
                continue
            return {"first": s, "end": j, "text": text, "nl": nl, "tokens": n, "headings_repaired": moved,
                    "repaired": repaired}
    return None


def blocks(source, doc, wordset):
    """[(block id, paragraphs, labels, meta)] as the selection reads a document (select_passages.blocks_of); in council
    documents a speaker label is stripped and marks a border"""
    import select_passages as sp

    out = []
    for block_id, paragraphs, meta in sp.blocks_of(source, doc, wordset):
        labels = [False] * len(paragraphs)
        if source == "raad":
            for k, p in enumerate(paragraphs):
                m = sp.SPEAKER_LABEL.match(p)
                if m:
                    labels[k] = True
                    paragraphs[k] = p[m.end():]
        out.append((block_id, paragraphs, labels, meta))
    return out


def doc_id_of(source, doc):
    return str(doc["id"] if source in ("handelingen", "fineweb") else doc["doc_id"])


def meta_of(source, doc, block_id, meta):
    """The source's columns of the datasets for a passage of this document and block"""
    import mainset

    if source == "fineweb":
        return {"fineweb2_id": str(doc["id"]), "url": doc["url"], "host": mainset.host_of(doc["url"]),
                "dump": doc["dump"], "date": doc["date"]}
    if source == "handelingen":
        return {"handelingen_id": str(doc["id"]), "url": f"https://zoek.officielebekendmakingen.nl/{doc['id']}.html",
                "session_year": doc["session_year"], "meeting_number": doc["meeting_number"], "date": doc["date"],
                "title": doc["title"], "turn": int(block_id[1:]), "speaker": meta.get("speaker"),
                "party": meta.get("party")}
    return {"gptnl_doc_id": str(doc["doc_id"]), "gptnl_source_id": doc["source_id"],
            "content_type": list(doc["content_type"] or []), "content_quality": doc["content_quality"],
            "content_integrity": doc["content_integrity"], "pii_presence": doc["pii_presence"]}


def document_candidates(source, doc, taken, splits, checker, count, count_text, counters):
    """The stretches of one document (at most MAX_PER_DOC), each a row with its metadata"""
    doc_id = doc_id_of(source, doc)
    rng = random.Random(f"{SEED}/{source}/{doc_id}")
    out = []
    for block_id, paragraphs, labels, meta in blocks(source, doc, checker.wordset):
        if len(out) >= MAX_PER_DOC:
            break
        if not paragraphs or sum(map(len, paragraphs)) < 2 * MIN_TOKENS:  # ~4-5 characters a token
            continue
        entry = taken.get((doc_id, block_id))
        if entry and entry["whole"]:
            counters["blocks_taken_whole"] += 1
            continue
        held = set()
        if entry:
            held = locate(entry["pieces"], paragraphs)
            if held is None:
                counters["docs_piece_not_found"] += 1
                return []
        counts = count(paragraphs)
        if sum(c for i, c in enumerate(counts) if i not in held) <= MIN_TOKENS:
            counters["blocks_too_short"] += 1
            continue
        counters["blocks_tried"] += 1
        usable = usable_paragraphs(paragraphs, counts, held, checker)
        found = find_stretch(source, paragraphs, labels, counts, usable, count_text, checker, rng, counters)
        if found is None:
            counters["blocks_no_stretch"] += 1
            continue
        hits, score = checker.idioms(source, found["text"])
        row = {"id": f"{source}:{doc_id}:{block_id}:p{found['first']}-{found['end'] - 1}", "source": source,
               "doc_id": doc_id, "block": block_id, "first": found["first"], "last": found["end"] - 1,
               "block_paragraphs": len(paragraphs), "split": splits.get(doc_id) or split_name(source, doc_id),
               "doc_known": doc_id in splits, "nl": found["nl"], "text": found["text"], "tokens_nl": found["tokens"],
               "headings_repaired": found["headings_repaired"], "repaired": found["repaired"], "idioms": hits,
               "idiom_score": score, "bin": bin_of(found["tokens"]), **meta_of(source, doc, block_id, meta)}
        out.append(row)
        counters["stretches"] += 1
    return out


# ---------------------------------------------------------------------------------------------------- candidates

_STATE = {}


def _init_worker(tokenizer, wordlist, idioms, frequent):
    from tokenizers import Tokenizer

    _STATE["tok"] = Tokenizer.from_file(str(Path(tokenizer) / "tokenizer.json"))
    _STATE["checker"] = Checker(wordlist, idioms, frequent)


def _count(texts):
    return [len(e.ids) for e in _STATE["tok"].encode_batch(list(texts), add_special_tokens=False)] if texts else []


def _count_text(text):
    return _count([text])[0]


def _group_task(task):
    """The candidates of one row group (runs in a worker process; the source's taken pieces and splits are the
    module's, inherited through fork)"""
    from huggingface_hub import HfFileSystem

    import select_passages as sp

    source, group = task
    counters = collections.Counter()
    for attempt in range(5):
        try:
            docs, c = sp.read_docs(source, [group], HfFileSystem())
            break
        except Exception as error:  # a read from Hugging Face that broke off: the same row group again
            if attempt == 4:
                raise
            print(f"retry {source} {group[:2]}: {error!r}", file=sys.stderr, flush=True)
            time.sleep(30 * (attempt + 1))
    counters.update(c)
    out = []
    for doc in docs:
        if source == "fineweb" and not dump_ok(doc["dump"]):
            counters["docs_dump_after_bound"] += 1
            continue
        counters["docs_searched"] += 1
        found = document_candidates(source, doc, _STATE["taken"], _STATE["splits"], _STATE["checker"], _count,
                                    _count_text, counters)
        if found:
            counters["docs_with_stretch"] += 1
        out += found
    return group, out, counters


def configure_sources():
    """select_passages.SOURCES with the revision and columns this module reads"""
    import select_passages as sp

    sp.SOURCES["handelingen"] = dict(sp.SOURCES["handelingen"], repo=(sp.HANDELINGEN[0], HANDELINGEN_REV),
                                     columns=READ_COLUMNS["handelingen"])
    sp.SOURCES["fineweb"] = dict(sp.SOURCES["fineweb"], columns=READ_COLUMNS["fineweb"])
    for s in ("raad", "dpc", "rechtspraak"):
        sp.SOURCES[s] = dict(sp.SOURCES[s], columns=READ_COLUMNS["gptnl"])
    return sp.SOURCES


def frequent_idioms(reports):
    """{source: set of idioms the selection's pools left out as too frequent} from their report.json files; a source
    takes the list of the pool that holds it"""
    out = {}
    for path in reports:
        rep = json.loads(Path(path).read_text())
        left_out = {i for i, _ in rep["idioms"]["left_out_list"]}
        # the pool's sources: those it was asked passages of (select_passages.py --counts source=n,...)
        sources = [k for k, v in (part.split("=") for part in rep["args"]["counts"].split(",")) if int(v) > 0]
        for s in sources:
            if s in out:
                raise SystemExit(f"two pool reports for {s}")
            out[s] = left_out
    return out


def blocked_ngrams(testsets):
    out = set()
    for path in sorted(Path(testsets).glob("*.txt")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                out |= ngram_hashes(line)
    return out


def existing_ngrams(rows):
    import numpy as np

    grams = set()
    for _, nl, _, _ in rows:
        grams |= ngram_hashes(nl)
    return np.array(sorted(grams), dtype=np.uint64)


def back_ratio(data, above=384):
    """The round trip's output tokens per Gemma 4 token of the original Dutch, over the existing passages of more than
    `above` tokens (the longest there): the budget's estimate of the round trip's output"""
    import pyarrow.parquet as pq

    out = n = 0
    for source in SOURCE_KEYS:
        for split in bg.SPLITS:
            for path in bg.source_files(data, source, "data", split):
                t = pq.read_table(path, columns=["tokens_nl", "back_tokens_out"]).to_pydict()
                for a, b in zip(t["tokens_nl"], t["back_tokens_out"]):
                    if a > above:
                        n += a
                        out += b
    return round(out / n, 4)


def reconstruction_checks(data, source, rows, checker, n=400):
    """How well this module reproduces the existing rows: the split rule on their documents, and the idiom score and
    sentence count of a sample (both measured on nl, the text after the edits)"""
    import pyarrow.parquet as pq

    import pairs as pm
    import select_passages as sp

    docs = {}
    for pid, _, split, folder in rows:
        if folder == "data":
            docs[parse_id(pid)[1]] = split
    split_ok = sum(split_name(source, d) == s for d, s in docs.items())
    sample = []
    for split in bg.SPLITS:
        for path in bg.source_files(data, source, "data", split):
            sample += pq.read_table(path, columns=["id", "nl", "idiom_score", "sentences_nl"]).to_pylist()
    sample = sorted(sample, key=lambda r: stable_hash(f"{SEED}/check/{r['id']}"))[:n]
    idiom_ok = sum(abs(checker.idioms(source, r["nl"])[1] - r["idiom_score"]) < 1e-4 for r in sample)
    sentences_ok = sum(pm.sentences(r["nl"], sp.ABBREVIATIONS) == r["sentences_nl"] for r in sample)
    return {"documents": len(docs), "split_rule_agrees": split_ok, "sample": len(sample),
            "idiom_score_agrees": idiom_ok, "sentences_nl_agrees": sentences_ok}


def candidates(args):
    import multiprocessing

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    import select_passages as sp

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    configure_sources()
    frequent = frequent_idioms(args.pool_reports)
    checker = Checker(args.wordlist, args.idioms, frequent)
    blocked = blocked_ngrams(args.testsets)
    sources = [s for s in SOURCE_KEYS if s in args.sources.split(",")]
    scan_all = set(filter(None, args.scan_all.split(",")))
    all_rows = {s: existing_rows(args.data, s) for s in SOURCE_KEYS}
    taken_grams = existing_ngrams([r for rs in all_rows.values() for r in rs])
    print(f"EXISTING {json.dumps({s: len(r) for s, r in all_rows.items()})} rows, {len(taken_grams):,} 13-grams, "
          f"{len(blocked):,} 13-grams of the test sets", flush=True)
    report_path = out / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report.update({"seed": SEED, "min_tokens": MIN_TOKENS, "max_tokens": MAX_TOKENS, "max_per_doc": MAX_PER_DOC,
                   "tries": TRIES, "min_prose_share": MIN_PROSE_SHARE, "fineweb_last_dump": FINEWEB_LAST_DUMP,
                   "want": args.want, "handelingen_revision": HANDELINGEN_REV,
                   "existing_13grams": len(taken_grams), "test_set_13grams": len(blocked),
                   "heads": json.loads((Path(args.data) / "heads.json").read_text()),
                   "back_ratio": back_ratio(args.data)})
    report.setdefault("sources", {})
    fs = HfFileSystem()
    host_cap = max(1, math.floor(HOST_CAP * TARGET))
    for source in sources:
        started = time.time()
        taken, splits = taken_pieces(all_rows[source])
        checks = reconstruction_checks(args.data, source, all_rows[source], checker)
        print(f"CHECKS {source} {json.dumps(checks)}", flush=True)
        _STATE.update(taken=taken, splits=splits)
        groups = sp.row_groups(source, 10 ** 9, f"{SEED}/groups", fs)
        random.Random(f"{SEED}/{source}/order").shuffle(groups)
        if args.max_groups:
            groups = groups[:args.max_groups]
        kept, seen_grams, per_host = [], set(), collections.Counter()
        counters, skipped = collections.Counter(), collections.Counter()
        valid, per_bin, read = collections.Counter(), collections.Counter(), 0
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(args.processes, initializer=_init_worker,
                      initargs=(args.tokenizer, args.wordlist, args.idioms, frequent)) as pool:
            for group, rows, c in pool.imap(_group_task, [(source, g) for g in groups], chunksize=1):
                read += 1
                counters.update(c)
                for r in rows:
                    grams = ngram_hashes(r["nl"])
                    arr = np.fromiter(grams, dtype=np.uint64, count=len(grams))
                    if grams & blocked:
                        skipped["test_set_13gram"] += 1
                        continue
                    if np.isin(arr, taken_grams, assume_unique=True).any():
                        skipped["existing_13gram"] += 1
                        continue
                    if grams & seen_grams:
                        skipped["candidate_13gram"] += 1
                        continue
                    valid[r["bin"]] += 1
                    if per_bin[r["bin"]] >= args.want:
                        continue
                    if source == "fineweb" and per_host[r["host"]] >= host_cap:
                        skipped["host_cap"] += 1
                        continue
                    seen_grams |= grams
                    per_host[r.get("host")] += 1
                    per_bin[r["bin"]] += 1
                    kept.append(dict(r, order=len(kept), row_group=f"{group[0]}#{group[1]}"))
                if read % 20 == 0:
                    print(f"GROUPS {source} {read}/{len(groups)} kept {dict(sorted(per_bin.items()))} valid "
                          f"{dict(sorted(valid.items()))} {round(time.time() - started)} s", flush=True)
                if all(per_bin[b] >= args.want for b in BINS) and source not in scan_all:
                    pool.terminate()
                    break
        table = pa.Table.from_pylist(kept)
        pq.write_table(table, out / f"{source}.parquet", compression="zstd")
        tokens = [r["tokens_nl"] for r in kept]
        scale = 1 if read == len(groups) else len(groups) / max(read, 1)
        info = {"row_groups": len(groups), "row_groups_read": read, "kept": len(kept),
                "kept_per_bin": {str(b): per_bin[b] for b in BINS}, "valid_per_bin": {str(b): valid[b] for b in BINS},
                "scanned_all": read == len(groups), "counters": dict(counters), "skipped": dict(skipped),
                "available_estimate": round(sum(valid.values()) * scale),
                "available_estimate_per_bin": {str(b): round(valid[b] * scale) for b in BINS},
                "splits": dict(collections.Counter(r["split"] for r in kept)),
                "docs_known": sum(r["doc_known"] for r in kept), "repaired": sum(r["repaired"] for r in kept),
                "tokens": {f"p{q}": bg.quantile(tokens, q / 100) for q in (5, 25, 50, 75, 95)} if tokens else {},
                "tokens_sum": sum(tokens), "reconstruction": checks, "seconds": round(time.time() - started)}
        if source == "fineweb":
            info["hosts"] = len(per_host)
            info["top_hosts"] = per_host.most_common(5)
        report["sources"][source] = info
        report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
        print(f"SOURCE {source} {json.dumps({k: v for k, v in info.items() if k != 'counters'})}", flush=True)
    print(f"CANDIDATES OK {json.dumps({s: report['sources'][s]['kept'] for s in report['sources']})}", flush=True)


# ---------------------------------------------------------------------------------------------------- judge, choose

def read_candidates(directory, sources=SOURCE_KEYS):
    import pyarrow.parquet as pq

    out = {}
    for source in sources:
        path = Path(directory) / f"{source}.parquet"
        if not path.exists():
            raise SystemExit(f"no candidates of {source}: {path}")
        out[source] = sorted(pq.read_table(path).to_pylist(), key=lambda r: r["order"])
    return out


def judge_shards(args):
    """The candidates as judge.py's input: <out>/w<k>/shard-NNNNN.parquet with id, source, split, doc_id, tokens, nl
    and an empty en, every worker a similar mix of lengths"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    import select_passages as sp

    rows = [{"id": r["id"], "source": r["source"], "split": r["split"], "doc_id": r["doc_id"], "tokens": r["tokens_nl"],
             "nl": r["nl"], "en": ""} for rs in read_candidates(args.candidates).values() for r in rs]
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("an id twice among the candidates")
    counts = {}
    for k, part in enumerate(sp.balanced_workers(rows, args.workers)):
        part.sort(key=lambda r: r["id"])
        d = Path(args.out) / f"w{k}"
        d.mkdir(parents=True, exist_ok=True)
        for n, start in enumerate(range(0, len(part), args.rows)):
            pq.write_table(pa.Table.from_pylist(part[start:start + args.rows]), d / f"shard-{n:05d}.parquet",
                           compression="zstd")
        counts[f"w{k}"] = len(part)
    print(f"JUDGE-SHARDS {json.dumps(counts)}", flush=True)


def topic_names():
    sys.path.insert(0, str(HERE.parent / "gemma_judge"))
    import judge

    return dict(judge.TOPICS)


def eligible(source, score):
    """None when a candidate's scores pass (mainset.final_selection's rule), else why not"""
    if score is None or score.get("skipped") is not None:
        return "no_score"
    if not score["score"] >= SOURCE_MIN:
        return "source_below"
    if source == "fineweb" and not (score.get("origin_score") or 0) >= ORIGIN_MIN:
        return "origin_below"
    return None


def bin_quotas(available, n):
    """Passages per bin for n in all: an equal share of n, a bin with fewer available leaving its remainder to the
    others, the shorter bins first (available: {bin: candidates that pass})"""
    quota = {b: 0 for b in BINS}
    left = n
    while left > 0:
        open_bins = [b for b in BINS if quota[b] < available.get(b, 0)]
        if not open_bins:
            break
        share = max(1, left // len(open_bins))
        for b in open_bins:
            take = min(share, available.get(b, 0) - quota[b], left)
            quota[b] += take
            left -= take
            if left == 0:
                break
    return quota


def pick(rows, n, topic_cap=None):
    """Up to n of rows in their order, spread over the bins (bin_quotas); with topic_cap no topic above that share
    of n while others are left (a second pass fills up from the capped ones, bin by bin)"""
    quota = bin_quotas(collections.Counter(r["bin"] for r in rows), n)
    cap = max(1, math.floor(topic_cap * n)) if topic_cap else None
    chosen, later, per, in_bin = [], [], collections.Counter(), collections.Counter()
    for r in rows:
        if in_bin[r["bin"]] >= quota[r["bin"]]:
            continue
        if cap is not None and per[r["topic"]] >= cap:
            later.append(r)
            continue
        per[r["topic"]] += 1
        in_bin[r["bin"]] += 1
        chosen.append(r)
    for r in later:
        if in_bin[r["bin"]] < quota[r["bin"]]:
            in_bin[r["bin"]] += 1
            chosen.append(r)
    return sorted(chosen, key=lambda r: r["order"])


def choose(args):
    """The passages of every source: the candidates whose scores pass, in their order, up to the number per source;
    <out>/<source>/data/{split}-00000-of-00001.parquet and selection.json"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    import mainset

    names = topic_names()
    scores = mainset.read_scores(args.scores)
    want = json.loads(Path(args.budget).read_text())["per_source"] if args.budget else None
    result = {"rule": {"source_min": SOURCE_MIN, "origin_min_fineweb": ORIGIN_MIN, "topic_cap_fineweb": TOPIC_CAP},
              "sources": {}}
    for source, rows in read_candidates(args.candidates).items():
        reasons = collections.Counter()
        live = []
        for r in rows:
            s = scores.get(r["id"])
            why = eligible(source, s)
            reasons[why or "eligible"] += 1
            if why is None:
                live.append(dict(r, source_score=s["score"], origin_score=s.get("origin_score"),
                                 topic=names[s["topic"]], topic_letter=s["topic"]))
        n = want[source] if want else args.per_source
        chosen = pick(live, n, TOPIC_CAP if source == "fineweb" else None)
        for split in bg.SPLITS:
            part = [r for r in chosen if r["split"] == split]
            path = Path(args.out) / source / "data" / f"{split}-00000-of-00001.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            schema = pa.Table.from_pylist(chosen).schema if chosen else None
            pq.write_table(pa.Table.from_pylist(part, schema=schema), path, compression="zstd")
        tokens = [r["tokens_nl"] for r in chosen]
        result["sources"][source] = {
            "candidates": len(rows), "reasons": dict(reasons), "eligible": len(live), "wanted": n,
            "chosen": len(chosen), "splits": dict(collections.Counter(r["split"] for r in chosen)),
            "eligible_per_bin": {str(b): sum(r["bin"] == b for r in live) for b in BINS},
            "chosen_per_bin": {str(b): sum(r["bin"] == b for r in chosen) for b in BINS},
            "tokens_sum": sum(tokens), "tokens": {f"p{q}": bg.quantile(tokens, q / 100) for q in (5, 25, 50, 75, 95)}
            if tokens else {}, "topics": dict(collections.Counter(r["topic"] for r in chosen).most_common())}
        print(f"CHOSEN {source} {json.dumps({k: v for k, v in result['sources'][source].items() if k != 'topics'})}",
              flush=True)
    (Path(args.out) / "selection.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------------------------------------------- smoke, budget

SMOKE_UNITS = 2       # per class: the first holds XLA's compilation, the second is the steady speed
SMOKE_PER_SLOT = 2    # passages per slot in a smoke unit: every slot is used again


def smokeplan(args):
    """The smoke test's two plans (smoke-nl, smoke-rt): per class of MODEL and RT_MODEL SMOKE_UNITS units of
    SMOKE_PER_SLOT x its slots passages of the candidates, the longest that the class holds first; the round trip's
    classes take the Dutch as its stand-in for English (what is measured is the shapes, the memory and the speed)"""
    pool = sorted((r for rs in read_candidates(args.candidates).values() for r in rs),
                  key=lambda r: (-r["tokens_nl"], r["id"]))
    for name, model, langs in (("smoke-nl", MODEL, {}), ("smoke-rt", RT_MODEL, {"source": "en", "target": "nl"})):
        m = ge.MODELS[model]
        classes = m["length_classes"]
        out = Path(args.out) / name
        units, lower = [], 0
        for cls, c in classes.items():
            k = SMOKE_PER_SLOT * m["slots"][cls]
            fits = [r for r in pool if r["tokens_nl"] <= c["max_source"]]
            inside = [r for r in fits if r["tokens_nl"] > lower]
            rows = (inside + [r for r in fits if r["tokens_nl"] <= lower])[:k * SMOKE_UNITS]
            lower = c["max_source"]
            if len(rows) < k * SMOKE_UNITS:
                raise SystemExit(f"{name} {cls}: {len(rows)} candidates, {k * SMOKE_UNITS} needed")
            for u in range(SMOKE_UNITS):
                chunk = sorted(rows[u * k:(u + 1) * k], key=lambda r: r["id"])
                rel = f"segments/smoke/train-{cls}-{u:04d}.parquet"
                bg.write_segments([{"id": r["id"], "text": r["nl"], "tokens": r["tokens_nl"]} for r in chunk], out / rel)
                units.append({"name": f"smoke/train-{cls}-{u:04d}", "part": "smoke", "split": "train", "class": cls,
                              "prompt": None, "segments": rel, "start": 0, "end": k, "n": k, "first": chunk[0]["id"],
                              "last": chunk[-1]["id"]})
        info = {"model": model, "classes": classes, "shard_size": None, "passages": sum(u["n"] for u in units),
                "units": units, "hash": bg.plan_hash(units), **langs}
        (out / "units.json").write_text(json.dumps(info, indent=0))
        print(f"SMOKEPLAN {name} {json.dumps({u['name']: u['n'] for u in units})}", flush=True)


def smoke_measures(log_lines):
    """Per unit of a smoke test's logs (bt_gemma4.py work): {unit: {tokens_out, seconds, out_per_s, not_ended, slots,
    peak_gib}} and per class the engine's start (s)"""
    units, starts, current = {}, collections.defaultdict(list), None
    for line in log_lines:
        if line.startswith("MAXTEXT "):
            current = json.loads(line[len("MAXTEXT "):])["class"]
        elif line.startswith("engine ready in ") and current:
            starts[current].append(float(line.split()[3]))
        elif line.startswith("SHARD "):
            r = json.loads(line[len("SHARD "):])
            units[r["unit"]] = {k: r[k] for k in ("tokens_out", "seconds", "out_per_s", "not_ended", "segments")}
        elif line.startswith("MEM "):
            r = json.loads(line[len("MEM "):])
            units.setdefault(r["unit"], {}).update(slots=r["slots"], peak_gib=r["peak_gib_max"])
    return units, dict(starts)


def class_rates(units, starts):
    """Per class: the steady output tokens per second (the last smoke unit), the compilation (the first unit's time
    beyond its tokens at that speed) and the engine's start"""
    out = {}
    per = collections.defaultdict(dict)
    for name, u in units.items():
        cls, k = name.rsplit("-", 2)[-2:]
        per[cls][int(k)] = u
    for cls, us in per.items():
        if len(us) < SMOKE_UNITS or any("seconds" not in u for u in us.values()):
            raise SystemExit(f"smoke class {cls}: {len(us)} units measured, {SMOKE_UNITS} needed")
        steady, first = us[max(us)], us[min(us)]
        rate = steady["tokens_out"] / steady["seconds"]
        out[cls] = {"out_per_s": round(rate, 1), "compile_s": round(max(0.0, first["seconds"] -
                                                                           first["tokens_out"] / rate), 1),
                    "start_s": round(max(starts.get(cls, [0.0])), 1), "slots": steady.get("slots"),
                    "peak_gib": max(u.get("peak_gib", 0) for u in us.values()),
                    "not_ended": sum(u["not_ended"] for u in us.values()),
                    "segments": sum(u["segments"] for u in us.values())}
    return out


def estimate(rows, nl_rates, rt_rates, out_ratio, back_ratio, workers):
    """Worker-hours of nl->en and the round trip for rows (dicts with tokens_nl): every passage's output tokens at the
    steady speed of its class, plus each class's start and compilation once per worker. out_ratio: English tokens per
    Dutch token, per nl->en class; back_ratio: the round trip's Dutch tokens per Dutch token of the original."""
    nl_classes = ge.MODELS[MODEL]["length_classes"]
    rt_classes = ge.MODELS[RT_MODEL]["length_classes"]
    seconds = collections.Counter()
    used = {"nl": set(), "rt": set()}
    for r in rows:
        n = r["tokens_nl"]
        c = bg.class_of(n, nl_classes)
        en = out_ratio[c] * n
        seconds[("nl", c)] += en / nl_rates[c]["out_per_s"]
        rc = bg.class_holding(en, n * bg.DECODE_MARGIN, rt_classes)
        if rc is None:
            raise SystemExit(f"{r.get('id')}: no class of {RT_MODEL} holds {en:.0f} tokens of English and "
                             f"{n * bg.DECODE_MARGIN:.0f} of decoding")
        seconds[("rt", rc)] += back_ratio * n / rt_rates[rc]["out_per_s"]
        used["nl"].add(c)
        used["rt"].add(rc)
    fixed = sum((nl_rates if d == "nl" else rt_rates)[c]["start_s"] + (nl_rates if d == "nl" else rt_rates)[c]["compile_s"]
                for d in ("nl", "rt") for c in used[d]) * workers
    decode = sum(seconds.values())
    return {"decode_worker_hours": round(decode / 3600, 2), "fixed_worker_hours": round(fixed / 3600, 2),
            "worker_hours": round((decode + fixed) / 3600, 2),
            "per_class_worker_hours": {f"{d}/{c}": round(v / 3600, 2) for (d, c), v in sorted(seconds.items())}}


def smoke_measures_command(args):
    """smoke.json from the smoke test's logs and shards: per class of both models the speed, the start, the slots
    that ran and the peak memory; per nl->en class the English tokens per Dutch token"""
    lines = []
    for path in args.logs:
        lines += Path(path).read_text(errors="replace").splitlines()
    units, starts = smoke_measures(lines)
    tokens = {r["id"]: r["tokens_nl"] for rs in read_candidates(args.candidates).values() for r in rs}
    ratio = collections.defaultdict(lambda: [0, 0])
    for i, r in read_shards(args.shards, "text_en").items():
        cls = r["unit"].rsplit("-", 2)[-2]
        ratio[cls][0] += r["tokens_out"]
        ratio[cls][1] += tokens[i]
    by = {"nl": {}, "rt": {}}
    nl_names, rt_names = set(ge.MODELS[MODEL]["length_classes"]), set(ge.MODELS[RT_MODEL]["length_classes"])
    rates = class_rates(units, starts)
    for cls, v in rates.items():
        by["nl" if cls in nl_names else "rt"][cls] = v
    missing = (nl_names - set(by["nl"])) | (rt_names - set(by["rt"]))
    if missing:
        raise SystemExit(f"smoke test: no measure of the classes {sorted(missing)}")
    by["out_ratio"] = {c: round(o / n, 4) for c, (o, n) in ratio.items() if n}
    if set(by["out_ratio"]) != nl_names:
        raise SystemExit(f"smoke test: English per Dutch token of {sorted(by['out_ratio'])}, not of every nl->en class")
    Path(args.out).write_text(json.dumps(by, indent=1))
    print(f"SMOKE {json.dumps(by)}", flush=True)


def budget(args):
    """The number of passages per source: the target, lowered (the same for every source) until the estimate of the
    run fits --worker-hours; budget.json with the measures and the estimate"""
    import pyarrow.parquet as pq

    smoke = json.loads(Path(args.smoke).read_text())
    nl_rates, rt_rates = smoke["nl"], smoke["rt"]
    report = json.loads(Path(args.report).read_text())
    by_source = {}
    for source in SOURCE_KEYS:
        rows = []
        for split in bg.SPLITS:
            for path in sorted((Path(args.selection) / source / "data").glob(f"{split}-*.parquet")):
                rows += pq.read_table(path, columns=["id", "tokens_nl", "order"]).to_pylist()
        by_source[source] = sorted(rows, key=lambda r: r["order"])
    out_ratio = smoke["out_ratio"]
    back_ratio = report["back_ratio"]

    def rows_for(n):
        return [r for rows in by_source.values() for r in rows[:n]]

    target = max(len(rows) for rows in by_source.values())
    n = target
    while n > 0 and estimate(rows_for(n), nl_rates, rt_rates, out_ratio, back_ratio, args.workers)["worker_hours"] \
            > args.worker_hours:
        n -= args.step
    if n <= 0:
        raise SystemExit(f"not even {args.step} passages per source fit {args.worker_hours} worker-hours")
    est = estimate(rows_for(n), nl_rates, rt_rates, out_ratio, back_ratio, args.workers)
    result = {"per_source": {s: min(n, len(rows)) for s, rows in by_source.items()}, "n": n, "target": target,
              "worker_hours_allowed": args.worker_hours, "workers": args.workers, "estimate": est,
              "hours_on_the_workers": round(est["worker_hours"] / args.workers, 2),
              "estimate_at_target": estimate(rows_for(target), nl_rates, rt_rates, out_ratio, back_ratio,
                                             args.workers),
              "out_ratio": out_ratio, "back_ratio": back_ratio, "nl": nl_rates, "rt": rt_rates}
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"BUDGET {json.dumps({k: result[k] for k in ('per_source', 'n', 'target', 'hours_on_the_workers')})} "
          f"{json.dumps(est)}", flush=True)


# ---------------------------------------------------------------------------------------------------- english, build

def read_shards(root, column):
    """{id: shard row} of every shard parquet under root; an id twice is an error"""
    import pyarrow.parquet as pq

    out = {}
    for path in sorted(Path(root).rglob("*.parquet")):
        for r in pq.read_table(path).to_pylist():
            if r["id"] in out:
                raise SystemExit(f"{r['id']} twice in the shards under {root} ({path.name})")
            if column not in r:
                raise SystemExit(f"{path}: no column {column}")
            out[r["id"]] = r
    return out


def selection_rows(selection, source, split):
    import pyarrow.parquet as pq

    rows = []
    for path in sorted((Path(selection) / source / "data").glob(f"{split}-*.parquet")):
        rows += pq.read_table(path).to_pylist()
    return rows


def english(args):
    """The English of every chosen passage (the nl->en shards) as the round trip's input, in the layout of the
    selection: <out>/<source>/data/{split}-00000-of-00001.parquet with id, nl, en and tokens_nl"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    shards = read_shards(args.shards, "text_en")
    counts = {}
    for source in SOURCE_KEYS:
        for split in bg.SPLITS:
            rows = selection_rows(args.selection, source, split)
            missing = [r["id"] for r in rows if r["id"] not in shards]
            if missing:
                raise SystemExit(f"{source}/{split}: {len(missing)} passages without an English, e.g. {missing[:3]}")
            out = [{"id": r["id"], "nl": r["nl"], "en": shards.pop(r["id"])["text_en"], "tokens_nl": r["tokens_nl"]}
                   for r in rows]
            path = Path(args.out) / source / "data" / f"{split}-00000-of-00001.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            schema = pa.schema([("id", pa.string()), ("nl", pa.string()), ("en", pa.string()), ("tokens_nl", pa.int32())])
            pq.write_table(pa.Table.from_pylist(out, schema=schema), path, compression="zstd")
            counts[f"{source}/{split}"] = len(out)
    if shards:
        raise SystemExit(f"{len(shards)} translations belong to no chosen passage, e.g. {sorted(shards)[:3]}")
    print(f"ENGLISH {json.dumps(counts)}", flush=True)


# The columns of a long row that the source's metadata and the selection give, by dataset
SELECTION_COLUMNS = {
    "fineweb": ["fineweb2_id", "url", "host", "dump", "date"],
    "handelingen": ["handelingen_id", "url", "session_year", "meeting_number", "date", "title", "turn", "speaker",
                    "party"],
    "gptnl": ["gptnl_doc_id", "gptnl_source_id", "content_type", "content_quality", "content_integrity",
              "pii_presence"],
}


def long_row(source, sel, en, back):
    """One new row of a dataset: the selection's columns, Gemma 4's English and its checks (the names of the
    published datasets, bt_gemma4.standalone), the round trip's columns (bt_gemma4.back_columns)"""
    import pairs as pm
    import select_passages as sp

    meta = SELECTION_COLUMNS["gptnl" if source in ("raad", "dpc", "rechtspraak") else source]
    row = {"id": sel["id"], **{c: sel[c] for c in meta}, "nl": sel["nl"], "tokens_nl": sel["tokens_nl"],
           "sentences_nl": pm.sentences(sel["nl"], sp.ABBREVIATIONS), "source_score": sel["source_score"],
           "origin_score": sel["origin_score"], "topic": sel["topic"],
           # drawn at random from the candidates that passed; the idiom class and the sampling passes of the shorter
           # passages do not apply
           "half": "random", "idiom_class": None, "idiom_score": sel["idiom_score"],
           "headings_repaired": sel["headings_repaired"], "drawn_in_pass": None}
    if source == "raad":
        row["repair_pass"] = sel["repaired"]
    if source == "handelingen":
        row["context_ids"] = []
    g4 = bg.g4_columns(sel["nl"], en["text_en"], source, en["ended"])
    row.update({"en": en["text_en"], "ended": en["ended"], "tokens_out": en["tokens_out"], "prompt": en["prompt"],
                "prompt_text": bg.prompt_text(en["prompt"]), "model": bg.MODEL_ID})
    row.update({k[len("gemma4_"):]: v for k, v in g4.items() if k != "gemma4_r2_length_out"})
    row.update({"region": en["region"], "vm": en["owner"], "time": en["time"]})
    row.update({"nl_back": back["text_nl"], "back_tokens_out": back["tokens_out"]})
    row.update(bg.back_columns(sel["nl"], en["text_en"], back["text_nl"], back["ended"]))
    return row


def build(args):
    """The new rows of every source and split (<out>/<source>/data/{split}.parquet) with the schema and column
    order of the existing files (--existing: the -gemma4 datasets the rows join), and build.json"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    en_shards = read_shards(args.en_shards, "text_en")
    rt_shards = read_shards(args.rt_shards, "text_nl")
    prompts = collections.Counter(r["prompt"] for r in list(en_shards.values()) + list(rt_shards.values()))
    if len(prompts) != 1:
        raise SystemExit(f"the shards have {len(prompts)} prompts: {dict(prompts)}")
    result = {"rows": {}, "stats": {}, "prompt": next(iter(prompts)), "heads": json.loads(
        (Path(args.existing) / "heads.json").read_text())}
    for source in SOURCE_KEYS:
        schema = pq.read_schema(bg.source_files(args.existing, source, "data", "train")[0])
        for split in bg.SPLITS:
            sel = selection_rows(args.selection, source, split)
            rows = []
            for r in sel:
                if r["id"] not in en_shards or r["id"] not in rt_shards:
                    raise SystemExit(f"{r['id']}: no English or no round trip in the shards")
                rows.append(long_row(source, r, en_shards.pop(r["id"]), rt_shards.pop(r["id"])))
            for r in rows:
                if set(r) != set(schema.names):
                    raise SystemExit(f"{source}: a new row's columns differ from the dataset's: "
                                     f"{sorted(set(r) ^ set(schema.names))}")
            table = pa.Table.from_pylist(rows, schema=schema)
            path = Path(args.out) / source / "data" / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, path, compression="zstd")
            if not pq.read_schema(path).equals(schema):
                raise SystemExit(f"{path}: the schema differs from the dataset's")
            result["rows"][f"{source}/{split}"] = len(rows)
            if rows:
                stats = bg.roundtrip_stats(rows)
                stats["hard"] = sum(bool(r["hard_fails"]) for r in rows)
                stats["checks"] = dict(collections.Counter(c for r in rows for c in r["hard_fails"]))
                stats["not_ended"] = sum(not r["ended"] for r in rows)
                stats["word_ratio"] = {f"p{q}": bg.quantile([r["word_ratio"] for r in rows], q / 100)
                                       for q in (5, 50, 95)}
                stats["tokens_nl"] = {f"p{q}": bg.quantile([r["tokens_nl"] for r in rows], q / 100)
                                      for q in (5, 50, 95)}
                result["stats"][f"{source}/{split}"] = stats
            print(f"BUILT {source}/{split} {len(rows)}", flush=True)
    if en_shards or rt_shards:
        raise SystemExit(f"{len(en_shards)} English and {len(rt_shards)} round trips belong to no chosen passage")
    (Path(args.out) / "build.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"BUILD OK {sum(result['rows'].values())} rows", flush=True)


# ---------------------------------------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--out", required=True)
    p.add_argument("--heads", default=None, help="heads.json of an earlier fetch: the same revisions")
    p = sub.add_parser("candidates")
    p.add_argument("--data", required=True, help="The -gemma4 datasets (bt_gemma4.py fetch --roundtrip layout)")
    p.add_argument("--tokenizer", required=True, help="Gemma 4's tokenizer directory")
    p.add_argument("--wordlist", required=True)
    p.add_argument("--idioms", required=True)
    p.add_argument("--pool-reports", nargs="+", required=True,
                   help="report.json of the selection's pools (their idioms left out as too frequent)")
    p.add_argument("--testsets", required=True, help="Directory of the test sets' texts (mainset.py testsets)")
    p.add_argument("--out", required=True)
    p.add_argument("--sources", default=",".join(SOURCE_KEYS))
    p.add_argument("--want", type=int, default=math.ceil(TARGET * OVERSAMPLE / len(BINS)),
                   help="Candidates kept per length bin and source")
    p.add_argument("--scan-all", default="", help="Sources whose every row group is read (exact counts)")
    p.add_argument("--max-groups", type=int, default=0, help="At most this many row groups per source (a trial)")
    p.add_argument("--processes", type=int, default=32)
    p = sub.add_parser("judge-shards")
    p.add_argument("--candidates", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rows", type=int, default=256)
    p.add_argument("--out", required=True)
    p = sub.add_parser("choose")
    p.add_argument("--candidates", required=True)
    p.add_argument("--scores", required=True, help="judge.py's score shards (w*/shard-*.parquet)")
    p.add_argument("--out", required=True)
    p.add_argument("--per-source", type=int, default=TARGET)
    p.add_argument("--budget", default=None, help="budget.json: its per_source instead of --per-source")
    p = sub.add_parser("smokeplan")
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("budget")
    p.add_argument("--smoke", required=True, help="smoke.json (smoke-measures)")
    p.add_argument("--report", required=True, help="report.json of candidates (back_ratio)")
    p.add_argument("--selection", required=True, help="choose's output at the target")
    p.add_argument("--worker-hours", type=float, default=96.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--step", type=int, default=100)
    p.add_argument("--out", required=True)
    p = sub.add_parser("smoke-measures")
    p.add_argument("--logs", nargs="+", required=True, help="the smoke test's work logs")
    p.add_argument("--shards", required=True, help="the smoke test's nl->en shards (for the English per Dutch token)")
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("english")
    p.add_argument("--selection", required=True)
    p.add_argument("--shards", required=True, help="A local copy of the nl->en shards")
    p.add_argument("--out", required=True)
    p = sub.add_parser("build")
    p.add_argument("--selection", required=True)
    p.add_argument("--existing", required=True, help="The -gemma4 datasets (fetch), for their schema")
    p.add_argument("--en-shards", required=True)
    p.add_argument("--rt-shards", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"fetch": fetch, "candidates": candidates, "judge-shards": judge_shards, "choose": choose, "smokeplan": smokeplan,
     "smoke-measures": smoke_measures_command, "budget": budget, "english": english,
     "build": build}[args.command](args)


if __name__ == "__main__":
    main()
