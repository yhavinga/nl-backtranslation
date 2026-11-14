#!/usr/bin/env python3
"""A back-translation (nl->en) of all 271,710 Dutch passages of the five back-translation datasets (SOURCES) with
Gemma 4 31B-it instead of TranslateGemma 27B. TranslateGemma's English is wordier than a human translation (median
0.956 Dutch words per English word, human ~1.06), and a translator trained on it learns to compress. An earlier pilot
found Gemma 4 with the prompt natural-v1 near the human length (1.019) but with added formatting (hard checks 4.9%).

    python bt_gemma4.py fetch --out DIR
    python bt_gemma4.py probeplan --data DIR [--pilot probe-ids.json] --tokenizer TOK --out DIR
    python bt_gemma4.py plan --data DIR --tokenizer TOK --out DIR [--model gemma4-31b-bt --max-source N]
    python bt_gemma4.py work --plan DIR --root gs://.../full --length-class small --prompt P --ckpt CKPT --owner VM/h0 ...
    python bt_gemma4.py status --plan DIR --root gs://.../full
    python bt_gemma4.py select --plan DIR --shards DIR [--qe qe.parquet] --out DIR
    python bt_gemma4.py publish --plan DIR --root gs://.../full --data DIR --repo REPO --region R --owner VM
    python bt_gemma4.py final --plan DIR --root gs://.../full --repo REPO --work-dir DIR
    python bt_gemma4.py card --prompt P --out README.md
    python bt_gemma4.py report --data DIR --out bt-gemma4.md
    python bt_gemma4.py datasets --data DIR --sources DIR --out DIR [--standalone]
    python bt_gemma4.py fetch --roundtrip --out DIR
    python bt_gemma4.py plan --data DIR --text en --tokenizer TOK --out DIR [--model gemma4-31b-rt]
    python bt_gemma4.py roundtrip --data DIR --shards DIR --out DIR

fetch: the five datasets of SOURCES (TranslateGemma 27B's English of each Dutch passage, with its scores and checks)
at fixed revisions, straight from Hugging Face.

Step 1, the prompt (probeplan, work on the probe plan, select): ~300 passages, per source 40 uniform (stratum random,
from the train rows outside the earlier pilot's sample), 7 long (385-511 TranslateGemma tokens), 7 from the lowest
decile of Dutch words per English word of TranslateGemma's English (ratio_low), and 6 of the pilot's natural-v1
translations that failed `structure` (stratum structure; --pilot names both id lists); plus 20 context turns of the
Handelingen (stratum context). Each prompt of PROMPTS translates all of them. THE RULE (fixed on 13 Nov 2026 before
any translation of the probe), on the stratum random only (200 passages: the other strata are chosen hard cases, not a
sample of the data): a prompt is eligible when its share of hard-check failures (pairs.hard_fails, or chatter) is at
most 1.5%; of the eligible ones the prompt whose median Dutch words per English word is closest to 1.06 (the median of
human translations: real English-Dutch pairs of Europarl and Global Voices); where the distances to 1.06 differ by
less than 0.01, the one with fewer hard-check failures. When no prompt is eligible: the one with the fewest failures,
reported as such. choose() is the rule.

Step 2, the full run (plan, work, publish): the passages of every source and split, and the Handelingen's context
turns, in units of SHARD_SIZE consecutive ids of one source, split and length class. Every process (4 chips, half a
v6e-8) takes units of one class at a time: it claims a unit in its region's bucket (an object created only if absent;
a claim older than the lease, or of an earlier run on the same VM, may be taken over), translates it and stores the
shard there. One region goes through the units forwards, the other backwards; other-done.txt in each region's bucket
(written by keeper.sh, which lists all buckets) names the units another region has done, so that no VM reads another
region's bucket. Where two regions should not even start the same unit (an on-demand TPU joining a run of spot VMs),
work --claims-root makes one region claim in the other's claims directory; its shards stay in its own bucket.
publish (one process a region, under a lease) puts each shard, with the source dataset's columns, the
new English and its checks, as one parquet file into a private dataset on Hugging Face: the dataset is the merge.
Nothing is filtered out. final makes the report (report) once every unit is there.

Step 3, one dataset per source (datasets): from a local copy of the merged dataset and of the source datasets (fetch),
per source the files of the source dataset, each with the same rows in the same order and the new columns. With
--standalone (the published datasets) without TranslateGemma's columns, and Gemma 4's under the plain names
(standalone()).

Step 4, the round trip (fetch --roundtrip, plan --text en, work, publish, roundtrip): Gemma 4 translates the English
`en` of the published -gemma4 datasets (ROUNDTRIP_SOURCES) back into Dutch, with the prompt of step 2 from English
into Dutch, greedy, in the length classes of RT_MODEL on the tokens of `en`; plan checks that every passage's original
Dutch fits the decode limit of its class with a margin (DECODE_MARGIN). The units run as in step 2, but publish puts
each shard as it is into the private dataset of the run. roundtrip then appends to every file of the -gemma4
datasets nl_back, chrF of nl_back against nl (pairs.back_measures, as chrf of the TranslateGemma datasets) and the
checks of nl_back (back_columns()), and counts them per source, config and split (roundtrip.json).
"""

import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import socket
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# pairs.py (and what it imports) one directory up in the repository, or beside this file on a VM
for _path in (HERE.parent, HERE):
    sys.path.insert(0, str(_path))

import gemma4_engine as ge  # noqa: E402

# The source datasets, at the revisions that were translated: key -> (Hugging Face dataset, revision)
SOURCES = {
    "handelingen": ("yhavinga/handelingen-tweede-kamer-nl-backtranslated-en", "b12d17adf134e06718182b2e81e95a5672782f05"),
    "raad": ("yhavinga/openraadsinformatie-nl-backtranslated-en", "16c10acce1d5336f420c11bb267d9b8c572cedd4"),
    "dpc": ("yhavinga/rijksoverheid-websites-nl-backtranslated-en", "881f04a20059aeaa51c616fa312ee387b7f6bf36"),
    "rechtspraak": ("yhavinga/rechtspraak-nl-backtranslated-en", "b187a9a22745876d7c74f1e49282f5ecb1b03402"),
    "fineweb": ("yhavinga/fineweb-2-nl-backtranslated-en", "763f2cbc76f326b221357aca7c0e9534f7a0d57d"),
}
# name: (key of SOURCES, directory of that dataset on Hugging Face); one config of the new dataset each
PARTS = {
    "fineweb": ("fineweb", "data"),
    "handelingen": ("handelingen", "data"),
    "handelingen_context": ("handelingen", "context_turns"),
    "raad": ("raad", "data"),
    "dpc": ("dpc", "data"),
    "rechtspraak": ("rechtspraak", "data"),
}
# The round trip's input: the published -gemma4 datasets (step 3 with --standalone), same keys, layout and rows
ROUNDTRIP_SOURCES = {
    "handelingen": ("yhavinga/handelingen-tweede-kamer-nl-en-gemma4", "6684cf1e46fa0f967d897fe99fe3cee14a717f23"),
    "raad": ("yhavinga/openraadsinformatie-nl-en-gemma4", "1e04772378dd007320e3253d982d25115dab4c84"),
    "dpc": ("yhavinga/rijksoverheid-websites-nl-en-gemma4", "8772802c64b1264d67b15cb3f8c4f71b94120e09"),
    "rechtspraak": ("yhavinga/rechtspraak-nl-en-gemma4", "6a8e1a82d8af0330a5d5b04113a06c1be3d336e2"),
    "fineweb": ("yhavinga/fineweb-2-nl-en-gemma4", "2d2e36d0dc0543201b09be0b20fbc30d45d5446a"),
}
RT_MODEL = "gemma4-31b-rt"
# The decode limit of a passage's class must hold its original Dutch (Gemma 4 tokens) this many times over
DECODE_MARGIN = 1.25
# The prompt's own words in a translation (the instruction translated instead of the text, as once in a pilot)
INSTRUCTION = re.compile(r"translate the following|vertaal de volgende|idiomatic (?:english|dutch)|"
                         r"idiomatisch (?:nederlands|engels)|output only the translation|geef alleen de vertaling",
                         re.IGNORECASE)
SPLITS = ("train", "validation")
EXPECTED_PASSAGES = 271_710
MODEL = "gemma4-31b-bt"
MODEL_ID = "google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475"
PROMPTS = ("natural-v1", "natural-v1-fmt", "natural-v1-fmt-len")
SHORT = {"natural-v1": "v1", "natural-v1-fmt": "fmt", "natural-v1-fmt-len": "fmtlen"}
SHARD_SIZE = 512
PROBE_SEED = "bt-gemma4/probe/0"
PROBE_STRATA = (("random", 40), ("long", 7), ("ratio_low", 7))
PROBE_STRUCTURE = 6
PROBE_CONTEXT = 20
RATIO_LOW_QUANTILE = 0.10
# The rule (module docstring)
TARGET_RATIO = 1.06
MAX_HARD = (3, 200)  # 1.5% as a fraction of integers: hard * 200 <= n * 3
TIE = 0.01
RULE_STRATUM = "random"
# Human translations (real English-Dutch pairs of Europarl and Global Voices), Dutch words per English word
HUMAN = {"median": 1.062, "p5": 0.889, "p95": 1.278}
# The length bound of the experiment's fine-tune data (gemma4_r2_length_out): characters Dutch / English within a
# factor 4/3 of the source's median over its train passages without hard_fails
R2_BOUNDS = {
    "dpc": {"median": 0.9836, "lo": 0.7377, "hi": 1.3114},
    "fineweb": {"median": 1.003, "lo": 0.7522, "hi": 1.3373},
    "handelingen": {"median": 1.0061, "lo": 0.7546, "hi": 1.3414},
    "raad": {"median": 0.993, "lo": 0.7448, "hi": 1.3239},
    "rechtspraak": {"median": 1.0287, "lo": 0.7715, "hi": 1.3716},
}
# The columns of a source dataset the probe reads
COLUMNS = ["id", "nl", "en", "qe_en_nl", "length_class", "hard_fails", "ended", "tokens_nl", "tokens_en"]
LEASE_SECONDS = 20 * 60
PUBLISH_LEASE_SECONDS = 15 * 60
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
# An introduction before the translation, or a note after it that the source has no counterpart of. "Of course," or
# "Certainly," alone opens many a faithful translation ("Natuurlijk, ..."), so an introduction must name the translation
INTRO = re.compile(r"^\s*(?:\*\*)?(?:(?:sure|certainly|of course)[,!.]?\s+)?(?:here(?:'s| is)\b[^\n]{0,60}\btranslation|"
                   r"(?:english |dutch )?translation\s*:|hier is\b[^\n]{0,60}\bvertaling|vertaling\s*:)", re.IGNORECASE)
NOTE = re.compile(r"\n\s*(?:\*\*)?(?:note|translator'?s note|opmerking|noot|n\.b\.)\s*:", re.IGNORECASE)


def stable(key):
    return hashlib.sha1(key.encode()).hexdigest()


def now():
    return time.time()


def utc(t=None):
    return datetime.datetime.fromtimestamp(now() if t is None else t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------------------------------- measures, io

def words(text):
    return len(text.split())


def word_ratio(nl, en):
    """Dutch words per English word (split on white space)"""
    return words(nl) / max(1, words(en))


def char_ratio(nl, en):
    """Dutch characters over English characters (stripped)"""
    return len(nl.strip()) / max(1, len(en.strip()))


def quantile(values, q):
    values = sorted(values)
    if not values:
        raise ValueError("quantile of nothing")
    return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]


def chatter(text, source=""):
    """An introduction to the translation, or a note the source does not have"""
    if not text:
        return False
    return bool(INTRO.search(text)) or (bool(NOTE.search(text)) and not NOTE.search(source or ""))


def measure(nl, en, source, ended):
    """The measures of one (Dutch original, English) pair: pairs.measure and pairs.hard_fails, word and character
    ratios, the length bound of R2_BOUNDS and chatter"""
    import pairs as pm

    m = pm.measure(nl, en, None)
    row = {"ended": ended, **{k: m[k] for k in ("decode_loop", "dutch_left", "len_ratio", "numbers_missing",
                                                 "structure", "sentences_nl", "sentences_en")}}
    row["hard_fails"] = pm.hard_fails(row)
    row["chatter"] = chatter(en, nl)
    row["word_ratio"] = round(word_ratio(nl, en), 4)
    row["char_ratio"] = round(char_ratio(nl, en), 4)
    b = R2_BOUNDS[source]
    row["r2_length_out"] = not b["lo"] <= row["char_ratio"] <= b["hi"]
    return row


def nl_num(x, digits=2, signed=False):
    """A number as Dutch writes it (the reports are in Dutch)"""
    if x is None:
        return "–"
    s = f"{x:+,.{digits}f}" if signed else f"{x:,.{digits}f}"
    return s.replace(",", " ").replace(".", ",").replace(" ", ".")


def pct(x, digits=1):
    return "–" if x is None else nl_num(100 * x, digits) + "%"


def write_parquet(rows, path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def read_rows(path, columns=None):
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns).to_pylist()


class Counter:
    """Gemma 4 token counts of texts"""

    def __init__(self, tokenizer_dir):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))

    def __call__(self, text):
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def many(self, texts):
        return [len(e.ids) for e in self.tok.encode_batch(list(texts), add_special_tokens=False)]


def class_of(tokens, classes):
    """The length class of a segment: the first whose max_source holds it"""
    for name, c in classes.items():
        if tokens <= c["max_source"]:
            return name
    raise SystemExit(f"a segment of {tokens} tokens is longer than the largest class")


def class_holding(tokens, decode, classes):
    """The first class whose max_source holds a segment of tokens and whose max_decode holds decode tokens, or None
    (a model with move_up: a round trip whose original Dutch is long for its English goes up a class)"""
    for name, c in classes.items():
        if tokens <= c["max_source"] and decode <= c["max_decode"]:
            return name
    return None


def shard_sizes(model, shard_size):
    """The passages of a unit per class: shard_slots x the class's slots where the model sets shard_slots (classes
    of very different slots), shard_size otherwise"""
    m = ge.MODELS[model]
    if "shard_slots" in m:
        return {c: m["shard_slots"] * m["slots"][c] for c in m["length_classes"]}
    return {c: shard_size for c in m["length_classes"]}


# ---------------------------------------------------------------------------------------------------- stores

class MemoryStore:
    """A bucket directory in memory, for the tests: key -> (bytes, generation)"""

    def __init__(self):
        self.objects, self.generation = {}, 0

    def _bump(self, key, data):
        self.generation += 1
        self.objects[key] = (data, self.generation)

    def list(self, prefix):
        return {k[len(prefix):] for k in self.objects if k.startswith(prefix)}

    def read(self, key):
        if key not in self.objects:
            return None
        data, gen = self.objects[key]
        return data.decode(), gen

    def create(self, key, text):
        if key in self.objects:
            return False
        self._bump(key, text.encode())
        return True

    def replace(self, key, text, generation):
        if key not in self.objects or self.objects[key][1] != generation:
            return False
        self._bump(key, text.encode())
        return True

    def exists(self, key):
        return key in self.objects

    def put(self, path, key):
        self._bump(key, Path(path).read_bytes())

    def get(self, key, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(self.objects[key][0])


class GcsStore:
    """A bucket directory (gs://bucket/prefix) through the Python client (gemma4_engine.gcs_client: gsutil is a snap on
    v6e and refuses to start under a process with no_new_privs); create and replace use generation preconditions"""

    def __init__(self, root):
        self.client = ge.gcs_client()
        name, prefix = ge.split_gs(root.rstrip("/"))
        self.bucket, self.prefix = self.client.bucket(name), prefix.rstrip("/")

    def _name(self, key):
        return f"{self.prefix}/{key}"

    def list(self, prefix):
        full = self._name(prefix)
        return {b.name[len(full):] for b in self.client.list_blobs(self.bucket, prefix=full)}

    def read(self, key):
        """(text, generation), or None when absent or changed while being read (the caller tries again later)"""
        from google.api_core.exceptions import NotFound, PreconditionFailed

        blob = self.bucket.blob(self._name(key))
        try:
            blob.reload()
            return blob.download_as_text(if_generation_match=blob.generation), blob.generation
        except (NotFound, PreconditionFailed):
            return None

    def _write(self, key, text, generation):
        from google.api_core.exceptions import NotFound, PreconditionFailed

        try:
            self.bucket.blob(self._name(key)).upload_from_string(text, if_generation_match=generation)
            return True
        except (PreconditionFailed, NotFound):
            return False

    def create(self, key, text):
        return self._write(key, text, 0)

    def replace(self, key, text, generation):
        return self._write(key, text, generation)

    def exists(self, key):
        return self.bucket.blob(self._name(key)).exists()

    def put(self, path, key):
        from google.cloud.storage.retry import DEFAULT_RETRY

        self.bucket.blob(self._name(key)).upload_from_filename(str(path), retry=DEFAULT_RETRY)

    def get(self, key, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.bucket.blob(self._name(key)).download_to_filename(str(path))


def claim(store, key, me, lease, t=None):
    """Claim key for me ({owner, run}): created if absent; taken over when the claim is older than lease seconds, or
    of another run of the same VM (a VM started again after a preemption keeps its name). Returns whether it is mine."""
    t = now() if t is None else t
    body = json.dumps({**me, "time": t})
    if store.create(key, body):
        return True
    found = store.read(key)
    if found is None:  # gone in between: the next pass tries again
        return False
    text, generation = found
    old = json.loads(text)
    mine = old.get("owner") == me["owner"] and old.get("run") == me["run"]  # a lease renewed by its holder
    same_vm = old.get("owner", "").split("/")[0] == me["owner"].split("/")[0] and old.get("run") != me["run"]
    if mine or same_vm or old.get("released") or t - old.get("time", 0) > lease:
        return store.replace(key, body, generation)
    return False


def release(store, key, me):
    """Give up a lease (claim treats it as free) without deleting the object"""
    found = store.read(key)
    if found and json.loads(found[0]).get("owner") == me["owner"]:
        store.replace(key, json.dumps({**me, "released": True, "time": now()}), found[1])


# ---------------------------------------------------------------------------------------------------- data

def fetch(args):
    from huggingface_hub import snapshot_download

    out = Path(args.out)
    for key, (repo, revision) in (ROUNDTRIP_SOURCES if args.roundtrip else SOURCES).items():
        patterns = ["data/*.parquet"] + (["context_turns/*.parquet"] if key == "handelingen" else [])
        snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=str(out / key),
                          allow_patterns=patterns)
        files = sorted(p.relative_to(out / key).as_posix() for p in (out / key).rglob("*.parquet"))
        print(f"FETCH {json.dumps({'source': key, 'repo': repo, 'revision': revision, 'files': files})}", flush=True)


def source_files(data, key, folder="data", split="train"):
    files = sorted((Path(data) / key / folder).glob(f"{split}-*.parquet"))
    if not files:
        raise SystemExit(f"no {folder}/{split}-*.parquet of {key} under {data}")
    return files


def part_files(data, part, split):
    key, folder = PARTS[part]
    return source_files(data, key, folder, split)


def read_part(data, part, split, columns=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pa.concat_tables([pq.read_table(p, columns=columns) for p in part_files(data, part, split)])


def classes_of(model):
    return ge.MODELS[model]["length_classes"]


def write_segments(rows, path):
    """translate.py's segments (id, text, tokens), sorted by id"""
    rows = sorted(rows, key=lambda r: r["id"])
    write_parquet([{"id": r["id"], "text": r["text"], "tokens": r["tokens"]} for r in rows], path)
    return rows


def plan_hash(units):
    h = hashlib.sha1()
    for u in units:
        h.update(f"{u['name']}|{u['first']}|{u['last']}|{u['n']}\n".encode())
    return h.hexdigest()


def make_units(segments_by_key, class_names, shard_size, part_order):
    """Units of shard_size consecutive segments of one (part, split, class), in the order class, part, split, k;
    shard_size an int, or one per class"""
    units = []
    for (part, split, cls), (path, rows) in segments_by_key.items():
        size = shard_size[cls] if isinstance(shard_size, dict) else shard_size
        for k, start in enumerate(range(0, len(rows), size)):
            chunk = rows[start:start + size]
            units.append({"name": f"{part}/{split}-{cls}-{k:04d}", "part": part, "split": split, "class": cls,
                          "prompt": None, "segments": path, "start": start, "end": start + len(chunk),
                          "n": len(chunk), "first": chunk[0]["id"], "last": chunk[-1]["id"]})
    units.sort(key=lambda u: (class_names.index(u["class"]), part_order.index(u["part"]), SPLITS.index(u["split"]),
                              u["name"]))
    return units


def plan(args):
    """The full run's units: every passage of PARTS (train and validation), by the Gemma 4 tokens of its text (--text:
    nl, translated into English; en, the round trip, translated back into Dutch) in the model's classes; a passage
    above --max-source (default the largest class) stops the plan. For the round trip (--text en, the -gemma4
    datasets) every passage's original Dutch (tokens_nl, Gemma 4 tokens) times DECODE_MARGIN must fit the decode
    limit of its class; with a model that sets move_up such a passage goes to the first later class that holds both.
    --parts: only these parts (a set of other passages, as the long ones of bt_long.py). --check-prompt: every
    passage's prompt (that prompt in the model's turn, Gemma 4 tokens) fits the largest prefill length of its class."""
    model = args.model or (RT_MODEL if args.text == "en" else MODEL)
    classes = classes_of(model)
    names = list(classes)
    largest = classes[names[-1]]["max_source"]
    max_source = args.max_source or largest
    if max_source > largest:
        raise SystemExit(f"--max-source {max_source} is beyond the largest class of {model} ({largest})")
    parts = args.parts.split(",") if args.parts else list(PARTS)
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        raise SystemExit(f"--parts: {unknown} are no parts ({', '.join(PARTS)})")
    move_up = args.text == "en" and ge.MODELS[model].get("move_up", False)
    source, target = ("en", "nl") if args.text == "en" else ("nl", "en")
    codec = ge.TurnCodec(None, ge.MODELS[model], args.check_prompt, source, target) if args.check_prompt else None
    count = Counter(args.tokenizer)
    out = Path(args.out)
    segments_by_key, total, longest_prompt = {}, 0, {}
    for part in parts:
        for split in SPLITS:
            columns = ["id", args.text] + (["tokens_nl"] if args.text == "en" else [])
            table = read_part(args.data, part, split, columns)
            ids, texts = table.column("id").to_pylist(), table.column(args.text).to_pylist()
            tokens = count.many(texts)
            over = [i for i, n in zip(ids, tokens) if n > max_source]
            if over:
                raise SystemExit(f"{part}/{split}: {len(over)} passages above {max_source} tokens, e.g. {over[:3]}")
            cls_of = [class_of(n, classes) for n in tokens]
            if args.text == "en":
                dutch = table.column("tokens_nl").to_pylist()
                if move_up:
                    cls_of = [class_holding(n, n_nl * DECODE_MARGIN, classes) for n, n_nl in zip(tokens, dutch)]
                    short = [i for i, c in zip(ids, cls_of) if c is None]
                else:
                    short = [i for i, c, n_nl in zip(ids, cls_of, dutch) if n_nl * DECODE_MARGIN > classes[c]["max_decode"]]
                if short:
                    raise SystemExit(f"{part}/{split}: {len(short)} passages whose original Dutch times "
                                     f"{DECODE_MARGIN} exceeds the decode limit of their class, e.g. {short[:3]}")
            if codec is not None:
                for i, c, n in zip(ids, cls_of, count.many(codec.text(t) for t in texts)):
                    limit = classes[c]["prefill_lengths"][-1]
                    if n > limit:
                        raise SystemExit(f"{i}: its prompt {args.check_prompt} has {n} tokens, more than the prefill "
                                         f"length {limit} of its class {c}")
                    longest_prompt[c] = max(longest_prompt.get(c, 0), n)
            per = collections.defaultdict(list)
            for i, text, n, c in zip(ids, texts, tokens, cls_of):
                per[c].append({"id": i, "text": text, "tokens": n})
            for cls, rows in per.items():
                rel = f"segments/{part}/{split}-{cls}.parquet"
                segments_by_key[(part, split, cls)] = (rel, write_segments(rows, out / rel))
            total += len(ids)
            print(f"PLAN {json.dumps({'part': part, 'split': split, 'passages': len(ids), 'classes': {c: len(r) for c, r in sorted(per.items())}})}", flush=True)
    if args.expect is not None and total != args.expect:
        raise SystemExit(f"{total} passages, expected {args.expect}")
    sizes = shard_sizes(model, args.shard_size)
    units = make_units(segments_by_key, names, sizes, list(PARTS))
    info = {"model": model, "classes": classes, "max_source": max_source,
            "shard_size": sizes if "shard_slots" in ge.MODELS[model] else args.shard_size,
            "passages": total, "units": units, "hash": plan_hash(units),
            "revisions": {k: v[1] for k, v in (ROUNDTRIP_SOURCES if args.text == "en" else SOURCES).items()}}
    if args.parts:
        info["parts"] = parts
    if codec is not None:
        info["longest_prompt"] = longest_prompt
    if args.text == "en":  # the languages of a plan; a plan without them is nl->en
        info.update(source="en", target="nl")
    (out / "units.json").write_text(json.dumps(info, indent=0))
    print(f"UNITS {json.dumps({'units': len(units), 'passages': total, 'hash': info['hash'], 'per_class': dict(collections.Counter(u['class'] for u in units))})}", flush=True)


def draw(rows, source, strata=PROBE_STRATA, ratio_quantile=RATIO_LOW_QUANTILE, seed=PROBE_SEED):
    """The sample of one source: per stratum, in the order of a hash of the id, the rows that pass its test and were
    not drawn before. Returns (rows with source and stratum, {ratio_cut, short: {stratum: [taken, wanted]}})."""
    cut = quantile([word_ratio(r["nl"], r["en"]) for r in rows], ratio_quantile)
    tests = {"random": lambda r: True,
             "long": lambda r: r["length_class"] == "long",
             "ratio_low": lambda r: word_ratio(r["nl"], r["en"]) <= cut}
    order = sorted(rows, key=lambda r: stable(f"{seed}/{r['id']}"))
    chosen, out, short = set(), [], {}
    for name, n in strata:
        taken = [r for r in order if r["id"] not in chosen and tests[name](r)][:n]
        if len(taken) < n:
            short[name] = [len(taken), n]
        for r in taken:
            chosen.add(r["id"])
            out.append(dict(r, source=source, stratum=name))
    return out, {"ratio_cut": cut, "short": short}


def probe_draw(rows_by_source, context_rows, pilot, seed=PROBE_SEED):
    """The probe's passages: per source the strata of PROBE_STRATA from the train rows outside the pilot's sample, then
    PROBE_STRUCTURE of the pilot's structure failures of that source; PROBE_CONTEXT context turns"""
    exclude = set(pilot["sample_ids"])
    structure = set(pilot["structure_ids"])
    out, info = [], {}
    for source, rows in rows_by_source.items():
        fresh = [r for r in rows if r["id"] not in exclude]
        drawn, info[source] = draw(fresh, source, strata=PROBE_STRATA, seed=seed)
        cases = sorted((r for r in rows if r["id"] in structure), key=lambda r: stable(f"{seed}/{r['id']}"))
        taken = [dict(r, source=source, stratum="structure") for r in cases[:PROBE_STRUCTURE]]
        if len(taken) < PROBE_STRUCTURE:
            info[source]["short"]["structure"] = [len(taken), PROBE_STRUCTURE]
        out += drawn + taken
    turns = sorted(context_rows, key=lambda r: stable(f"{seed}/{r['id']}"))[:PROBE_CONTEXT]
    out += [dict(r, source="handelingen", stratum="context") for r in turns]
    ids = [r["id"] for r in out]
    if len(ids) != len(set(ids)):
        raise SystemExit("a probe passage is in two strata")
    return out, info


def probeplan(args):
    """The probe's passages (sample.parquet) and units: one per prompt and class, all passages of the class"""
    from transformers import AutoTokenizer

    pilot = json.loads(Path(args.pilot).read_text()) if args.pilot else {"sample_ids": [], "structure_ids": []}
    out = Path(args.out)
    rows_by_source = {}
    for part, (key, folder) in PARTS.items():
        if folder != "data":
            continue
        table = read_part(args.data, part, "train", COLUMNS)
        rows_by_source[key] = table.to_pylist()
    context = read_part(args.data, "handelingen_context", "train",
                        ["id", "nl", "en", "qe_en_nl", "hard_fails", "ended", "tokens_nl", "tokens_en"]).to_pylist()
    for r in context:
        r["length_class"] = None
    rows, info = probe_draw(rows_by_source, context, pilot)
    classes = classes_of(args.model)
    count = Counter(args.tokenizer)
    for r, n in zip(rows, count.many(r["nl"] for r in rows)):
        r["tokens_g4"], r["class_g4"] = n, class_of(n, classes)
    rows.sort(key=lambda r: r["id"])
    write_parquet(rows, out / "sample.parquet")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    segs = [{"id": r["id"], "text": r["nl"], "tokens": r["tokens_g4"]} for r in rows]
    overhead = {}
    for name in PROMPTS:
        codec = ge.TurnCodec(tokenizer, ge.MODELS[args.model], name, "nl", "en")
        codec.check([s["text"] for s in segs[:3]])
        worst = 0
        for s in segs:
            n = len(codec.encode(s["text"]))
            limit = classes[class_of(s["tokens"], classes)]["prefill_lengths"][-1]
            if n > limit:
                raise SystemExit(f"{s['id']}: prompt {name} has {n} tokens, more than its class's prefill {limit}")
            worst = max(worst, n - s["tokens"])
        overhead[name] = worst
    units = []
    for cls in classes:
        chosen = [s for s in segs if class_of(s["tokens"], classes) == cls]
        if not chosen:
            continue
        rel = f"segments/probe-{cls}.parquet"
        write_segments(chosen, out / rel)
        for name in PROMPTS:
            units.append({"name": f"{SHORT[name]}/{cls}", "part": "probe", "split": "train", "class": cls,
                          "prompt": name, "segments": rel, "start": 0, "end": len(chosen), "n": len(chosen),
                          "first": chosen[0]["id"], "last": chosen[-1]["id"]})
    summary = {"rows": len(rows), "strata": dict(collections.Counter(r["stratum"] for r in rows)),
               "classes": dict(collections.Counter(r["class_g4"] for r in rows)), "prompt_overhead": overhead,
               "draw": info}
    (out / "units.json").write_text(json.dumps({"model": args.model, "classes": classes, "shard_size": None,
                                                "passages": len(rows), "units": units, "hash": plan_hash(units),
                                                "probe": summary}, indent=0))
    print(f"PROBEPLAN {json.dumps(summary)}", flush=True)


# ---------------------------------------------------------------------------------------------------- work

def load_plan(path):
    return json.loads((Path(path) / "units.json").read_text())


def languages(plan_info):
    """(source, target) of a plan: nl->en unless the plan names them (the round trip: en->nl)"""
    return plan_info.get("source", "nl"), plan_info.get("target", "en")


class Progress:
    """The units done in this region (shards/) and in the other (other-done.txt), read again after `every` seconds"""

    def __init__(self, store, every=30):
        self.store, self.every, self.read_at = store, every, None
        self.done, self.other = set(), set()

    def refresh(self, force=False):
        if force or self.read_at is None or now() - self.read_at > self.every:
            self.done = {k[:-len(".parquet")] for k in self.store.list("shards/") if k.endswith(".parquet")}
            found = self.store.read("other-done.txt")
            self.other = set(found[0].split()) if found else set()
            self.read_at = now()

    def finished(self, name):
        self.refresh()
        return name in self.done or name in self.other


def ordered(units, direction):
    return list(units) if direction == "forward" else list(reversed(units))


def work_units(units, store, me, translate, lease=LEASE_SECONDS, progress=None, log=print, claims=None):
    """For each unit in order not done here or there: claim it, translate it (translate(unit) -> (rows, seconds),
    rows a dict of columns) and store its shard. Returns the names done. claims: the store of the claims (default
    store), when the processes of two regions share one claims directory so that neither starts a unit the other has
    started; shards and other-done.txt stay in store."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    progress = progress or Progress(store)
    claims = store if claims is None else claims
    done = []
    for u in units:
        if progress.finished(u["name"]):
            continue
        if not claim(claims, f"claims/{u['name']}", me, lease):
            continue
        if store.exists(f"shards/{u['name']}.parquet"):  # finished by another process since the last listing
            continue
        rows, seconds = translate(u)
        n = len(rows["id"])
        rows = dict(rows, unit=[u["name"]] * n, prompt=[u["prompt"]] * n, owner=[me["owner"]] * n,
                    region=[me.get("region")] * n, time=[utc()] * n)
        local = Path(me.get("work_dir", "/tmp")) / "shards" / f"{u['name']}.parquet"
        local.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(rows), local, compression="zstd")
        store.put(local, f"shards/{u['name']}.parquet")
        local.unlink()
        done.append(u["name"])
        n_in, n_out = sum(rows["tokens_in"]), sum(rows["tokens_out"])
        log(f"SHARD {json.dumps({'unit': u['name'], 'prompt': u['prompt'], 'segments': n, 'tokens_in': n_in, 'tokens_out': n_out, 'seconds': round(seconds, 1), 'out_per_s': round(n_out / max(seconds, 1e-9)), 'not_ended': rows['ended'].count(False)})}")
    return done


class Translator:
    """Gemma 4 on this process's 4 chips for one length class: the engine is made at the first unit"""

    def __init__(self, args, plan_info):
        self.args, self.plan = args, plan_info
        self.model = ge.MODELS[plan_info["model"]]
        slots = getattr(args, "slots", None) or self.model["slots"][args.length_class]
        self.shapes = dict(plan_info["classes"][args.length_class], slots=slots)
        self.device = self.tokenizer = None
        self.codecs, self.tables = {}, {}

    def _start(self):
        import jax
        from transformers import AutoTokenizer

        args = self.args
        if (jax.device_count(), jax.process_count()) != (self.model["chips"], 1):
            raise SystemExit(f"needs one process with {self.model['chips']} chips, has {jax.process_count()} "
                             f"process(es) with {jax.device_count()} devices")
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        print(f"MAXTEXT {json.dumps({'model': self.plan['model'], 'class': args.length_class, 'shapes': self.shapes, 'ckpt': args.ckpt, **ge.maxtext_source(), 'patches': ge.patch_maxtext()})}", flush=True)
        started = now()
        engine, params = ge.make_engine(self.plan["model"], args.ckpt, args.tokenizer, args.work_dir, self.shapes)
        self.device = ge.EngineSlots(engine, params, self.shapes["prefill_lengths"])
        print(f"engine ready in {now() - started:.0f} s", flush=True)

    def codec(self, prompt):
        if prompt not in self.codecs:
            self.codecs[prompt] = ge.TurnCodec(self.tokenizer, self.model, prompt, *languages(self.plan))
        return self.codecs[prompt]

    def segments(self, u):
        import pyarrow.parquet as pq

        if u["segments"] not in self.tables:
            self.tables[u["segments"]] = pq.read_table(Path(self.args.plan) / u["segments"]).to_pylist()
        rows = self.tables[u["segments"]][u["start"]:u["end"]]
        if not rows or rows[0]["id"] != u["first"] or rows[-1]["id"] != u["last"] or len(rows) != u["n"]:
            raise SystemExit(f"{u['name']}: the segments file does not hold the plan's unit")
        return rows

    def __call__(self, u):
        if self.device is None:
            self._start()
        rows = self.segments(u)
        prompt = u["prompt"] or self.args.prompt
        codec = self.codec(prompt)
        texts = [r["text"] for r in rows]
        codec.check(texts[:2] + texts[-1:])
        out, seconds = ge.translate_shard(self.device, codec, self.tokenizer, [r["id"] for r in rows], texts,
                                          self.shapes, self.args.sync_every, self.model["eos_ids"])
        print(f"MEM {json.dumps({'unit': u['name'], 'slots': self.shapes['slots'], **ge.peak_memory()})}", flush=True)
        u["prompt"] = prompt
        return out, seconds


def work(args):
    plan_info = load_plan(args.plan)
    if args.length_class not in plan_info["classes"]:
        raise SystemExit(f"--length-class {args.length_class} is not a class of the plan")
    units = ordered([dict(u) for u in plan_info["units"] if u["class"] == args.length_class], args.direction)
    if not args.prompt and any(u["prompt"] is None for u in units):
        raise SystemExit("--prompt is needed for the units of the full plan")
    store = GcsStore(args.root)
    claims = GcsStore(args.claims_root) if args.claims_root else store
    me = {"owner": args.owner, "run": args.run, "region": args.region, "work_dir": args.work_dir}
    translator = Translator(args, plan_info)
    started = now()
    done = work_units(units, store, me, translator, args.lease, log=lambda s: print(s, flush=True), claims=claims)
    print(f"CLASS {json.dumps({'class': args.length_class, 'units': len(units), 'done_here': len(done), 'engine': translator.device is not None, 'seconds': round(now() - started)})}", flush=True)


def region_status(plan_info, progress):
    progress.refresh(force=True)
    per = collections.defaultdict(lambda: {"units": 0, "here": 0, "other": 0, "pending": 0})
    for u in plan_info["units"]:
        c = per[u["class"]]
        c["units"] += 1
        if u["name"] in progress.done:
            c["here"] += 1
        elif u["name"] in progress.other:
            c["other"] += 1
        else:
            c["pending"] += 1
    pending = sum(c["pending"] for c in per.values())
    return {"classes": dict(per), "pending": pending, "complete": pending == 0}


def status(args):
    s = region_status(load_plan(args.plan), Progress(GcsStore(args.root)))
    print(json.dumps(s["classes"]))
    print("COMPLETE" if s["complete"] else f"PENDING {s['pending']}", flush=True)


# ---------------------------------------------------------------------------------------------------- measures

def g4_columns(nl, en, part, ended):
    """The new English's checks as columns (gemma4_*): pairs.measure and hard_fails, chatter, word and character
    ratios, the length bound of R2_BOUNDS of the source"""
    m = measure(nl, en, PARTS[part][0], ended)
    return {"gemma4_hard_fails": m["hard_fails"] + (["chatter"] if m["chatter"] else []),
            "gemma4_structure": m["structure"], "gemma4_chatter": m["chatter"],
            "gemma4_decode_loop": m["decode_loop"], "gemma4_dutch_left": m["dutch_left"],
            "gemma4_len_ratio": m["len_ratio"], "gemma4_numbers_missing": m["numbers_missing"],
            "gemma4_word_ratio": m["word_ratio"], "gemma4_char_ratio": m["char_ratio"],
            "gemma4_r2_length_out": m["r2_length_out"]}


def g4_schema():
    import pyarrow as pa

    strings = pa.list_(pa.string())
    return [("en_gemma4", pa.string()), ("gemma4_ended", pa.bool_()), ("gemma4_tokens_nl", pa.int32()),
            ("gemma4_tokens_out", pa.int32()), ("gemma4_prompt", pa.string()), ("gemma4_prompt_text", pa.string()),
            ("gemma4_model", pa.string()), ("gemma4_hard_fails", strings), ("gemma4_structure", strings),
            ("gemma4_chatter", pa.bool_()), ("gemma4_decode_loop", pa.bool_()), ("gemma4_dutch_left", pa.bool_()),
            ("gemma4_len_ratio", pa.float64()), ("gemma4_numbers_missing", strings),
            ("gemma4_word_ratio", pa.float64()), ("tg_word_ratio", pa.float64()), ("gemma4_char_ratio", pa.float64()),
            ("gemma4_r2_length_out", pa.bool_()), ("gemma4_region", pa.string()), ("gemma4_vm", pa.string()),
            ("gemma4_time", pa.string())]


def prompt_text(prompt, source="nl", target="en"):
    return ge.user_text(ge.PROMPTS[prompt], "{text}", source, target)


def published_rows(unit, shard, source_rows, segments_tokens):
    """The rows of one unit for the dataset: the published dataset's columns of each passage, then the new English and
    its checks; in the order of the shard"""
    out = []
    texts = {}
    for i, en, ended, n_out, prompt, owner, region, t in zip(
            shard["id"], shard["text_en"], shard["ended"], shard["tokens_out"], shard["prompt"], shard["owner"],
            shard["region"], shard["time"]):
        src = source_rows.get(i)
        if src is None:
            raise SystemExit(f"{unit['name']}: {i} is not a passage of {unit['part']}/{unit['split']}")
        if prompt not in texts:
            texts[prompt] = prompt_text(prompt)
        row = dict(src)
        row.update({"en_gemma4": en, "gemma4_ended": ended, "gemma4_tokens_nl": segments_tokens[i],
                    "gemma4_tokens_out": n_out, "gemma4_prompt": prompt, "gemma4_prompt_text": texts[prompt],
                    "gemma4_model": MODEL_ID})
        row.update(g4_columns(src["nl"], en, unit["part"], ended))
        row["tg_word_ratio"] = round(word_ratio(src["nl"], src["en"]), 4)
        row.update({"gemma4_region": region, "gemma4_vm": owner, "gemma4_time": t})
        out.append(row)
    return out


# ---------------------------------------------------------------------------------------------------- select

def choose(stats):
    """THE RULE (module docstring). stats: {prompt: {"n", "hard", "median"}} of the stratum random, in the order of
    PROMPTS. Returns (prompt, reason, table)."""
    table = []
    for prompt, s in stats.items():
        table.append({"prompt": prompt, "n": s["n"], "hard": s["hard"], "share": s["hard"] / s["n"],
                      "median": s["median"], "distance": abs(s["median"] - TARGET_RATIO),
                      "eligible": s["hard"] * MAX_HARD[1] <= s["n"] * MAX_HARD[0]})
    eligible = [t for t in table if t["eligible"]]
    if eligible:
        best = min(t["distance"] for t in eligible)
        close = [t for t in eligible if t["distance"] - best < TIE]
        pick = min(close, key=lambda t: (t["hard"] / t["n"], t["distance"]))
        reason = ("rule: eligible (hard <= 1.5%), closest to 1.06" if len(close) == 1 else
                  f"rule: eligible, within {TIE} of the closest to 1.06: {', '.join(t['prompt'] for t in close)}; "
                  "the fewest hard-check failures")
    else:
        pick = min(table, key=lambda t: (t["hard"] / t["n"], t["distance"]))
        reason = "no prompt has hard <= 1.5%: the fewest hard-check failures (reported as such)"
    return pick["prompt"], reason, table


def setup_stats(rows, key):
    ratios = [r[f"word_ratio_{key}"] for r in rows]
    hard = sum(bool(r[f"hard_{key}"]) for r in rows)
    cats = collections.Counter(c for r in rows for c in (r[f"structure_{key}"] or []))
    checks = collections.Counter(c for r in rows for c in r[f"hard_{key}"])
    return {"n": len(rows), "hard": hard, "median": statistics.median(ratios) if ratios else None,
            "quantiles": {f"p{round(q * 100)}": quantile(ratios, q) for q in QUANTILES} if ratios else {},
            "under_0_9": sum(x < 0.9 for x in ratios) / len(ratios) if ratios else None,
            "structure_categories": dict(cats), "checks": dict(checks)}


def select(args):
    """The probe's measures per prompt and the rule's choice: compare.parquet, selection.json, selection.md"""
    plan_info = load_plan(args.plan)
    sample = read_rows(Path(args.plan) / "sample.parquet")
    shards = {}
    for path in sorted(Path(args.shards).rglob("*.parquet")):
        for r in read_rows(path):
            key = SHORT[r["prompt"]]
            if r["id"] in shards.setdefault(key, {}):
                raise SystemExit(f"{r['id']} twice for {r['prompt']}")
            shards[key][r["id"]] = r
    qe = {}
    if args.qe:
        qe = {r["id"]: r for r in read_rows(args.qe)}
    keys = ["tg"] + [SHORT[p] for p in PROMPTS]
    rows = []
    for p in sample:
        part = "handelingen_context" if p["stratum"] == "context" else p["source"]
        row = {k: p[k] for k in ("id", "source", "stratum", "class_g4", "tokens_g4", "nl")}
        for key in keys:
            if key == "tg":
                en, ended = p["en"], p["ended"]
            else:
                t = shards.get(key, {}).get(p["id"])
                if t is None:
                    raise SystemExit(f"{key}: no translation of {p['id']}")
                en, ended = t["text_en"], t["ended"]
            m = g4_columns(p["nl"], en, part, ended)
            row[f"en_{key}"] = en
            row[f"hard_{key}"] = m["gemma4_hard_fails"]
            row[f"structure_{key}"] = m["gemma4_structure"]
            row[f"word_ratio_{key}"] = m["gemma4_word_ratio"]
            for name in (f"qe_nl_en_{key}", f"qe_en_{key}_nl"):
                if name in qe.get(p["id"], {}):
                    row[name] = qe[p["id"]][name]
        rows.append(row)
    out = Path(args.out)
    write_parquet(rows, out / "compare.parquet")
    by_stratum = collections.defaultdict(list)
    for r in rows:
        by_stratum[r["stratum"]].append(r)
    stats = {s: {k: setup_stats(rs, k) for k in keys} for s, rs in sorted(by_stratum.items())}
    rule_rows = by_stratum[RULE_STRATUM]
    prompt, reason, table = choose({p: {"n": stats[RULE_STRATUM][SHORT[p]]["n"],
                                        "hard": stats[RULE_STRATUM][SHORT[p]]["hard"],
                                        "median": stats[RULE_STRATUM][SHORT[p]]["median"]} for p in PROMPTS})
    per_source = {src: {k: setup_stats([r for r in rule_rows if r["source"] == src], k) for k in keys}
                  for src in sorted({r["source"] for r in rule_rows})}
    qe_stats = {}
    if qe:
        for s, rs in by_stratum.items():
            # metricx_qe.py leaves a pair with an empty translation unscored (None)
            qe_stats[s] = {k: {d: statistics.fmean(v) for d, c in
                               (("nl_en", f"qe_nl_en_{k}"), ("en_nl", f"qe_en_{k}_nl")) if all(c in r for r in rs)
                               for v in [[r[c] for r in rs if r[c] is not None]] if v}
                           for k in keys}
    result = {"created": utc(), "plan_hash": plan_info["hash"], "chosen": prompt, "reason": reason, "rule": table,
              "stats": stats, "per_source_random": per_source, "qe": qe_stats,
              "prompts": {p: prompt_text(p) for p in PROMPTS}}
    (out / "selection.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    (out / "selection.md").write_text(selection_md(result, rows), encoding="utf-8")
    print(f"CHOSEN {json.dumps({'prompt': prompt, 'reason': reason})}", flush=True)


def selection_md(result, rows):
    num = nl_num
    names = {"tg": "TranslateGemma 27B (gepubliceerd)", "v1": "Gemma 4 natural-v1", "fmt": "Gemma 4 natural-v1-fmt",
             "fmtlen": "Gemma 4 natural-v1-fmt-len"}
    keys = list(names)
    lines = [f"# Promptkeuze voor de nieuwe BT met Gemma 4 31B-it (gegenereerd {result['created']})", "",
             "## Keuzeregel (vastgelegd vóór de uitslag)", "",
             "Alleen op het stratum `random` (200 passages, 40 per bron, uit de trainrijen buiten de steekproef van de eerdere pilot). "
             "Een prompt komt in aanmerking als zijn aandeel harde-controlefouten (`pairs.hard_fails` of `chatter`) "
             "hoogstens 1,5% is (3 van 200). Daarvan de prompt met de mediaan NL/EN-woorden het dichtst bij 1,06; "
             "liggen de afstanden tot 1,06 minder dan 0,01 uit elkaar, dan de prompt met minder harde-controlefouten. "
             "Haalt geen prompt 1,5%: de prompt met de minste fouten.", "",
             "| prompt | n | hard | aandeel | mediaan NL/EN | afstand tot 1,06 | in aanmerking |", "|---|---:|---:|---:|---:|---:|---|"]
    for t in result["rule"]:
        lines.append(f"| `{t['prompt']}` | {t['n']} | {t['hard']} | {pct(t['share'])} | {num(t['median'], 3)} | "
                     f"{num(t['distance'], 3)} | {'ja' if t['eligible'] else 'nee'} |")
    lines += ["", f"**Gekozen: `{result['chosen']}`** ({result['reason']}).", "",
              "## Per stratum", "",
              "| stratum | opzet | n | NL/EN-woorden p5 / p25 / mediaan / p75 / p95 | < 0,9 | hard | structuur-categorieën |",
              "|---|---|---:|---|---:|---:|---|"]
    for s, per in result["stats"].items():
        for k in keys:
            st = per[k]
            q = st["quantiles"]
            lines.append(f"| `{s}` | {names[k]} | {st['n']} | {num(q['p5'], 3)} / {num(q['p25'], 3)} / "
                         f"{num(st['median'], 3)} / {num(q['p75'], 3)} / {num(q['p95'], 3)} | {pct(st['under_0_9'])} | "
                         f"{st['hard']} | {', '.join(f'{c} {n}' for c, n in sorted(st['structure_categories'].items(), key=lambda x: -x[1])) or '–'} |")
    lines += ["", "Menselijke doelverdeling (echte paren van Europarl en Global Voices): mediaan 1,062, p5 0,889, p95 1,278.",
              "", "## Stratum `random` per bron", "", "| bron | opzet | mediaan NL/EN | hard |", "|---|---|---:|---:|"]
    for src, per in result["per_source_random"].items():
        for k in keys:
            lines.append(f"| {src} | {names[k]} | {num(per[k]['median'], 3)} | {per[k]['hard']} |")
    if result["qe"]:
        lines += ["", "## MetricX-24-QE (alleen ter rapportage)", "",
                  "| stratum | opzet | QE nl→en gem. | QE en→nl gem. |", "|---|---|---:|---:|"]
        for s, per in result["qe"].items():
            for k in keys:
                lines.append(f"| `{s}` | {names[k]} | {num(per[k].get('nl_en'))} | {num(per[k].get('en_nl'))} |")
    lines += ["", "## Prompts", ""]
    for p, text in result["prompts"].items():
        lines += [f"`{p}`:", "", "```text", text, "```", ""]
    lines += ["## Gevallen met toegevoegde opmaak in de eerdere pilot (stratum `structure`)", ""]
    for r in [r for r in rows if r["stratum"] == "structure"][:6]:
        lines += [f"### `{r['id']}`", "", f"- natural-v1: {r['hard_v1'] or '–'}; fmt: {r['hard_fmt'] or '–'}; "
                  f"fmt-len: {r['hard_fmtlen'] or '–'}", "", "```text", "NL: " + r["nl"][:600], "",
                  "fmt: " + (r["en_fmt"] or "")[:600], "```", ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------- publish

def hf_path(unit_name):
    part, rest = unit_name.split("/", 1)
    return f"{part}/{rest}.parquet"


class SourceRows:
    """The published dataset's rows of a (part, split) by id, read once from the local copy of Hugging Face"""

    def __init__(self, data):
        self.data, self.cache = data, {}

    def __call__(self, part, split):
        if (part, split) not in self.cache:
            table = read_part(self.data, part, split)
            self.cache[(part, split)] = (table.schema, {r["id"]: r for r in table.to_pylist()})
        return self.cache[(part, split)]


def unit_file(unit, shard_path, sources, tokens_of, out_path):
    """The parquet file of one unit for the dataset, with the published schema plus g4_schema()"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema, by_id = sources(unit["part"], unit["split"])
    shard = pq.read_table(shard_path).to_pydict()
    rows = published_rows(unit, shard, by_id, tokens_of(unit))
    full = pa.schema(list(schema) + [pa.field(n, t) for n, t in g4_schema()])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=full), out_path, compression="zstd")
    return len(rows)


def hf_file(plan_info, unit, shard_path, sources, tokens_of, out_path):
    """The file of one unit for the run's Hugging Face dataset and its rows: unit_file() of the nl->en run, or for
    the round trip the shard as it is (roundtrip joins it with the -gemma4 datasets)"""
    import pyarrow.parquet as pq

    if languages(plan_info) == ("en", "nl"):
        return str(shard_path), pq.read_metadata(shard_path).num_rows
    return str(out_path), unit_file(unit, shard_path, sources, tokens_of, out_path)


def publish(args):
    """One process a region (a lease on publish.lock): each shard of this region not yet on Hugging Face, as one
    parquet file, in commits of at most --batch files every --interval seconds; ends once the region is complete and
    all its shards are on Hugging Face (writes region-complete-<region>.json)"""
    import pyarrow.parquet as pq
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    plan_info = load_plan(args.plan)
    units = {u["name"]: u for u in plan_info["units"]}
    store = GcsStore(args.root)
    progress = Progress(store)
    sources = SourceRows(args.data)
    seg_cache = {}

    def tokens_of(unit):
        if unit["segments"] not in seg_cache:
            t = pq.read_table(Path(args.plan) / unit["segments"], columns=["id", "tokens"]).to_pydict()
            seg_cache[unit["segments"]] = dict(zip(t["id"], t["tokens"]))
        return seg_cache[unit["segments"]]

    me = {"owner": args.owner, "run": args.run}
    work_dir = Path(args.work_dir)
    complete_key = f"region-complete-{args.region}.json"
    while True:
        if store.exists(complete_key) and region_status(plan_info, progress)["complete"]:
            print(f"{utc()} {complete_key} is there and the region is complete: nothing to publish", flush=True)
            return
        if not claim(store, "publish.lock", me, PUBLISH_LEASE_SECONDS):
            print(f"{utc()} publish.lock held by another process", flush=True)
            if args.once:
                return
            time.sleep(args.interval)
            continue
        state = region_status(plan_info, progress)
        on_hf = {f for f in api.list_repo_files(args.repo, repo_type="dataset") if f.endswith(".parquet")}
        new = [n for n in sorted(progress.done) if n in units and hf_path(n) not in on_hf]
        for k in range(0, len(new), args.batch):
            ops, rows = [], 0
            for name in new[k:k + args.batch]:
                shard = work_dir / "pub-shards" / f"{name}.parquet"
                store.get(f"shards/{name}.parquet", shard)
                out, n = hf_file(plan_info, units[name], shard, sources, tokens_of,
                                 work_dir / "pub-files" / hf_path(name))
                rows += n
                ops.append(CommitOperationAdd(path_in_repo=hf_path(name), path_or_fileobj=out))
            api.create_commit(args.repo, repo_type="dataset", operations=ops,
                              commit_message=f"{len(ops)} shards from {args.region} ({rows} rows)")
            size = sum(Path(op.path_or_fileobj).stat().st_size for op in ops)
            for op in ops:
                Path(op.path_or_fileobj).unlink()
            print(f"PUBLISHED {json.dumps({'files': len(ops), 'rows': rows, 'bytes': size, 'region': args.region, 'time': utc()})}", flush=True)
            if not claim(store, "publish.lock", me, PUBLISH_LEASE_SECONDS):
                print("publish.lock lost", flush=True)
                break
        if state["complete"] and not new:
            on_hf = {f for f in api.list_repo_files(args.repo, repo_type="dataset") if f.endswith(".parquet")}
            missing_here = [n for n in progress.done if n in units and hf_path(n) not in on_hf]
            if not missing_here:
                all_on_hf = sum(hf_path(n) in on_hf for n in units)
                body = json.dumps({"region": args.region, "time": utc(), "units_here": len(progress.done),
                                   "units_on_hf": all_on_hf, "units": len(units)})
                found = store.read(complete_key)
                if found is None:
                    store.create(complete_key, body)
                else:
                    store.replace(complete_key, body, found[1])
                release(store, "publish.lock", me)
                print(f"REGION COMPLETE {body}", flush=True)
                return
        if args.once:
            return
        time.sleep(args.interval)


# ---------------------------------------------------------------------------------------------------- card, report

def card(args):
    import yaml

    configs = [{"config_name": part, "data_files": [{"split": s, "path": f"{part}/{s}-*"} for s in SPLITS]}
               for part in PARTS]
    meta = {"license": "other", "language": ["nl", "en"], "pretty_name": "Dutch passages back-translated to English by "
            "Gemma 4 31B-it", "task_categories": ["translation"], "tags": ["back-translation", "synthetic"],
            "configs": configs}
    text = prompt_text(args.prompt)
    sources = ", ".join(f"[{repo}](https://huggingface.co/datasets/{repo})" for repo, _ in SOURCES.values())
    body = f"""# Dutch passages back-translated to English by Gemma 4 31B-it (private, work in progress)

The 271,710 Dutch passages and context turns of the five back-translation datasets {sources}
(revisions of 11 Nov 2026), translated again from Dutch into English with `{MODEL_ID}` instead of TranslateGemma 27B,
so that the synthetic English is about as long as a human translation. One config per source (`handelingen_context`:
the Handelingen's context turns), splits as in the source datasets. Every row keeps all columns of its source dataset
(TranslateGemma's English is `en`) and adds:

| column | meaning |
|---|---|
| `en_gemma4` | Gemma 4 31B-it's English translation of `nl` (greedy) |
| `gemma4_ended` | the translation stopped by itself |
| `gemma4_tokens_nl`, `gemma4_tokens_out` | Gemma 4 tokens of `nl` and of the output (with the stop token) |
| `gemma4_prompt`, `gemma4_prompt_text` | the prompt's name and its text (`{{text}}` stands for `nl`) |
| `gemma4_model` | model and revision |
| `gemma4_hard_fails` | the hard checks the pair (`nl`, `en_gemma4`) fails: `not_ended`, `decode_loop`, `dutch_left`, `len_ratio`, `structure`, `newline_doubled` (as `hard_fails` of the source datasets), and `chatter` (an introduction or a note around the translation) |
| `gemma4_structure`, `gemma4_chatter`, `gemma4_decode_loop`, `gemma4_dutch_left`, `gemma4_len_ratio`, `gemma4_numbers_missing` | the measures behind the checks, as in the source datasets |
| `gemma4_word_ratio`, `tg_word_ratio` | Dutch words per English word (split on whitespace) of `en_gemma4` and of `en` |
| `gemma4_char_ratio`, `gemma4_r2_length_out` | Dutch characters over English characters; outside the length bound of our fine-tune data (factor 4/3 around the source's median) |
| `gemma4_region`, `gemma4_vm`, `gemma4_time` | where and when the translation was made |

Nothing was filtered out. The prompt, in Gemma 4's chat template with an empty thought channel:

```text
{text}
```

Code: `gemma4_bt/bt_gemma4.py` of nl-backtranslation. The license of each config is that of its source dataset;
`en_gemma4` was made with Gemma 4 31B-it (Apache 2.0), TranslateGemma's columns fall under the Gemma Terms of Use.
Before this dataset is made public its card must be completed (license per config, selection, limitations).

## Acknowledgements

The translations and scores in this dataset were produced on Cloud TPUs provided by Google's
[TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).
"""
    Path(args.out).write_text("---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + "---\n\n" + body,
                              encoding="utf-8")


def ratio_stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"n": len(values), **{f"p{round(q * 100)}": quantile(values, q) for q in QUANTILES},
            "under_0_9": sum(v < 0.9 for v in values) / len(values),
            "within_human_p5_p95": sum(HUMAN["p5"] <= v <= HUMAN["p95"] for v in values) / len(values)}


def report_stats(rows):
    hard = collections.Counter(c for r in rows for c in r["gemma4_hard_fails"])
    cats = collections.Counter(c for r in rows for c in (r["gemma4_structure"] or []))
    tg_hard = collections.Counter(c for r in rows for c in (r["hard_fails"] or []))
    return {"n": len(rows), "g4": ratio_stats([r["gemma4_word_ratio"] for r in rows]),
            "tg": ratio_stats([r["tg_word_ratio"] for r in rows]),
            "g4_hard": sum(bool(r["gemma4_hard_fails"]) for r in rows) / len(rows),
            "tg_hard": sum(bool(r["hard_fails"]) for r in rows) / len(rows),
            "g4_checks": dict(hard), "g4_structure": dict(cats), "tg_checks": dict(tg_hard),
            "g4_not_ended": sum(not r["gemma4_ended"] for r in rows),
            "g4_r2_length_out": sum(r["gemma4_r2_length_out"] for r in rows) / len(rows)}


def report(args):
    """bt-gemma4.md and .json from the dataset's files (a local copy of the Hugging Face dataset)"""
    import pyarrow.parquet as pq

    cols = ["id", "nl", "en", "en_gemma4", "hard_fails", "gemma4_hard_fails", "gemma4_structure", "gemma4_ended",
            "gemma4_word_ratio", "tg_word_ratio", "gemma4_r2_length_out", "gemma4_prompt", "gemma4_region"]
    by_part, files, size = {}, 0, 0
    for part in PARTS:
        rows = []
        for path in sorted((Path(args.data) / part).glob("*.parquet")):
            rows += pq.read_table(path, columns=cols).to_pylist()
            files += 1
            size += path.stat().st_size
        by_part[part] = rows
    total = sum(len(r) for r in by_part.values())
    ids = [r["id"] for rs in by_part.values() for r in rs]
    stats = {part: report_stats(rows) for part, rows in by_part.items() if rows}
    stats["_all"] = report_stats([r for rs in by_part.values() for r in rs])
    result = {"created": utc(), "files": files, "bytes": size, "rows": total, "unique_ids": len(set(ids)),
              "expected": EXPECTED_PASSAGES, "human": HUMAN, "stats": stats,
              "prompts": dict(collections.Counter(r["gemma4_prompt"] for rs in by_part.values() for r in rs)),
              "regions": dict(collections.Counter(r["gemma4_region"] for rs in by_part.values() for r in rs))}
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    Path(args.out).write_text(report_md(result, by_part), encoding="utf-8")
    print(f"REPORT {json.dumps({'rows': total, 'unique': len(set(ids)), 'files': files})}", flush=True)


def report_md(result, by_part):
    num = nl_num
    lines = [f"# Nieuwe BT nl→en met Gemma 4 31B-it: uitkomst (gegenereerd {result['created']})", "",
             f"Rijen {result['rows']:,} in {result['files']} parquet-bestanden ({result['bytes'] / 1e6:,.1f} MB, de totale "
             f"upload naar Hugging Face), unieke id's {result['unique_ids']:,} (verwacht {result['expected']:,}). "
             f"Prompts: {result['prompts']}. Regio's: {result['regions']}.", "",
             "## NL-woorden per EN-woord per bron", "",
             "| bron | opzet | n | p5 | p10 | p25 | mediaan | p75 | p90 | p95 | < 0,9 | binnen p5–p95 van de mens |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for part, st in result["stats"].items():
        for key, name in (("tg", "TranslateGemma 27B"), ("g4", "Gemma 4 31B")):
            q = st[key]
            lines.append(f"| {part} | {name} | {q['n']:,} | " + " | ".join(num(q[p], 3) for p in
                         ("p5", "p10", "p25", "p50", "p75", "p90", "p95")) +
                         f" | {pct(q['under_0_9'])} | {pct(q['within_human_p5_p95'])} |")
    h = result["human"]
    lines += ["", f"Menselijke doelverdeling (echte paren van Europarl en Global Voices): mediaan {num(h['median'], 3)}, "
              f"p5 {num(h['p5'], 3)}, p95 {num(h['p95'], 3)}.", "", "## Harde controles per bron", "",
              "| bron | hard TG | hard Gemma 4 | Gemma 4 per controle | structuur-categorieën Gemma 4 | niet af | buiten lengtegrens |",
              "|---|---:|---:|---|---|---:|---:|"]
    for part, st in result["stats"].items():
        lines.append(f"| {part} | {pct(st['tg_hard'])} | {pct(st['g4_hard'])} | "
                     f"{', '.join(f'{c} {n}' for c, n in sorted(st['g4_checks'].items(), key=lambda x: -x[1])) or '–'} | "
                     f"{', '.join(f'{c} {n}' for c, n in sorted(st['g4_structure'].items(), key=lambda x: -x[1])) or '–'} | "
                     f"{st['g4_not_ended']} | {pct(st['g4_r2_length_out'])} |")
    lines += ["", "## Voorbeelden", ""]
    for part, rows in by_part.items():
        if not rows:
            continue
        pick = sorted(rows, key=lambda r: stable(f"report/{r['id']}"))
        shown = [r for r in pick if len(r["nl"].split()) <= 120][:1]
        shown += [r for r in pick if r["gemma4_hard_fails"] and len(r["nl"].split()) <= 120][:1]
        for r in shown:
            lines += [f"### {part} `{r['id']}` (NL/EN TG {num(r['tg_word_ratio'])}, Gemma 4 {num(r['gemma4_word_ratio'])}; "
                      f"hard Gemma 4: {r['gemma4_hard_fails'] or '–'})", "",
                      "| Nederlands | TranslateGemma 27B | Gemma 4 31B |", "|---|---|---|",
                      "| " + " | ".join(x.replace("\n", "<br>").replace("|", "\\|") for x in (r["nl"], r["en"], r["en_gemma4"])) + " |", ""]
    return "\n".join(lines) + "\n"


def final(args):
    """Once every unit of the plan is a file on Hugging Face (and no other file is): the report, once a region (a
    lease on report.lock), from a copy of the dataset on this VM, to <root>/report/"""
    from huggingface_hub import HfApi, snapshot_download

    plan_info = load_plan(args.plan)
    expected = {hf_path(u["name"]) for u in plan_info["units"]}
    on_hf = {f for f in HfApi().list_repo_files(args.repo, repo_type="dataset") if f.endswith(".parquet")}
    if on_hf - expected:
        raise SystemExit(f"{len(on_hf - expected)} files on Hugging Face that are no unit of the plan, e.g. "
                         f"{sorted(on_hf - expected)[:3]}")
    if expected - on_hf:
        print(f"NOT COMPLETE: {len(expected - on_hf)} of {len(expected)} units not yet on Hugging Face", flush=True)
        return
    if languages(plan_info) == ("en", "nl"):
        print("FINAL every unit of the round trip is on Hugging Face; roundtrip makes the datasets and their numbers",
              flush=True)
        return
    store = GcsStore(args.root)
    if store.exists("report/bt-gemma4.md"):
        print("the report is there", flush=True)
        return
    me = {"owner": args.owner, "run": args.run}
    if not claim(store, "report.lock", me, 3600):
        print("report.lock held by another VM", flush=True)
        return
    work_dir = Path(args.work_dir)
    local = snapshot_download(args.repo, repo_type="dataset", local_dir=str(work_dir / "dataset"),
                              allow_patterns=["*.parquet", "README.md"])
    out = work_dir / "report" / "bt-gemma4.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    report(argparse.Namespace(data=local, out=str(out)))
    for path in (out, out.with_suffix(".json")):
        store.put(path, f"report/{path.name}")
    print(f"FINAL report in {args.root}/report/", flush=True)


# The columns a standalone dataset leaves out: TranslateGemma's translation, back-translation, scores and checks, the
# token count and length class of its tokenizer, the word ratio of its English, and gemma4_r2_length_out, whose bound
# is a factor around the median of TranslateGemma's pairs. sentences_nl stays: it counts the sentences of nl alone.
TRANSLATEGEMMA_COLUMNS = ("en", "nl_back", "qe_en_nl", "chrf", "tokens_nl", "tokens_en", "length_class", "len_ratio",
                          "sentences_en", "numbers_missing", "structure", "ended", "decode_loop", "dutch_left",
                          "back_ended", "back_decode_loop", "back_structure", "hard_fails", "tg_word_ratio",
                          "gemma4_r2_length_out")
# Gemma 4's columns that take the places of TranslateGemma's (new name -> column of the merged dataset)
IN_PLACE = {"en": "en_gemma4", "tokens_nl": "gemma4_tokens_nl"}


def standalone(table):
    """A table of a dataset per source without TranslateGemma: the columns of TRANSLATEGEMMA_COLUMNS left out,
    en_gemma4 as en and gemma4_<x> as <x>. en and tokens_nl (Gemma 4 tokens) stand where TranslateGemma's stood, the
    other Gemma 4 columns follow in their order; every other column stays where it is."""
    names = table.column_names
    missing = [c for c in ("en", "tokens_nl", *IN_PLACE.values()) if c not in names]
    if missing:
        raise SystemExit(f"not a dataset per source with Gemma 4's columns: {missing} missing")
    renamed = {"en_gemma4": "en", **{c: c[len("gemma4_"):] for c in names
                                     if c.startswith("gemma4_") and c not in TRANSLATEGEMMA_COLUMNS}}
    order = []
    for c in names:
        if c in IN_PLACE:
            order.append(IN_PLACE[c])
        elif c not in TRANSLATEGEMMA_COLUMNS and c not in renamed:
            order.append(c)
    order += [c for c in names if c in renamed and c not in IN_PLACE.values()]
    new_names = [renamed.get(c, c) for c in order]
    twice = sorted({n for n in new_names if new_names.count(n) > 1})
    if twice:
        raise SystemExit(f"columns twice after dropping the prefix gemma4_: {twice}")
    return table.select(order).rename_columns(new_names)


def datasets(args):
    """One dataset per source of SOURCES, in --out/<key>/: the files of the source dataset (its local copy in
    --sources, as fetch wrote it), each with the same rows in the same order; every row has the source's columns,
    checked to be unchanged, and the new ones of the merged dataset (--data: a local copy, one directory per part).
    With --standalone each file is standalone() of that."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(args.out)
    counts = {}
    for part, (key, folder) in PARTS.items():
        files = sorted((Path(args.data) / part).glob("*.parquet"))
        if not files:
            raise SystemExit(f"no {part}/*.parquet under {args.data}")
        merged = pa.concat_tables([pq.read_table(f) for f in files])
        ids = merged.column("id").to_pylist()
        position = {i: k for k, i in enumerate(ids)}
        if len(position) != len(ids):
            raise SystemExit(f"{part}: {len(ids) - len(position)} ids more than once in the merged dataset")
        for split in SPLITS:
            for path in source_files(args.sources, key, folder, split):
                source = pq.read_table(path)
                take = []
                for i in source.column("id").to_pylist():
                    if i not in position:
                        raise SystemExit(f"{part}: {i} of {path.name} is not (or twice) in the merged dataset")
                    take.append(position.pop(i))
                rows = merged.take(take)
                if not rows.select(source.column_names).equals(source):
                    raise SystemExit(f"{part}/{path.name}: the merged rows differ from the source's in its columns")
                if args.standalone:
                    rows = standalone(rows)
                target = out / key / folder / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(rows, target, compression="zstd")
                counts[f"{key}/{folder}/{path.name}"] = rows.num_rows
        if position:
            raise SystemExit(f"{part}: {len(position)} rows of the merged dataset are in no file of the source, "
                             f"e.g. {sorted(position)[:3]}")
    print(f"DATASETS {json.dumps(counts)}", flush=True)


# ---------------------------------------------------------------------------------------------------- round trip

def back_columns(nl, en, back, ended):
    """The round trip's measures of one row: chrF of the back-translation against the original and its decode_loop
    and structure against en (pairs.back_measures, as in the TranslateGemma datasets), English left in it
    (publish.english_left, the mirror of dutch_left) and chatter"""
    import pairs as pm
    import publish

    m = pm.back_measures(nl, en, back)
    return {"chrf": m["chrf"], "back_ended": ended, "back_decode_loop": m["back_decode_loop"],
            "back_english_left": publish.english_left(back, en), "back_structure": m["back_structure"],
            "back_chatter": chatter(back, en)}


def _back_chunk(rows):
    return [back_columns(*r) for r in rows]


def back_schema():
    import pyarrow as pa

    return [("nl_back", pa.string()), ("back_tokens_out", pa.int32()), ("back_ended", pa.bool_()),
            ("chrf", pa.float64()), ("back_decode_loop", pa.bool_()), ("back_english_left", pa.bool_()),
            ("back_structure", pa.list_(pa.string())), ("back_chatter", pa.bool_())]


def roundtrip_stats(rows):
    """The numbers of the round trip over rows (dicts with the dataset's hard_fails, nl, en and the new columns)"""
    chrf = [r["chrf"] for r in rows]
    passed = [r["chrf"] for r in rows if not r["hard_fails"]]
    q = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    return {"n": len(rows),
            "chrf": {f"p{round(p * 100)}": quantile(chrf, p) for p in q} | {"mean": round(statistics.fmean(chrf), 3)},
            "chrf_without_hard_fails": ({f"p{round(p * 100)}": quantile(passed, p) for p in q}
                                        | {"mean": round(statistics.fmean(passed), 3), "n": len(passed)}),
            "chrf_below": {str(t): sum(c < t for c in chrf) for t in (30, 40, 50)},
            "back_not_ended": sum(not r["back_ended"] for r in rows),
            "back_decode_loop": sum(r["back_decode_loop"] for r in rows),
            "back_english_left": sum(r["back_english_left"] for r in rows),
            "back_chatter": sum(r["back_chatter"] for r in rows),
            "back_structure": sum(bool(r["back_structure"]) for r in rows),
            "back_structure_categories": dict(collections.Counter(c for r in rows for c in r["back_structure"])),
            "instruction": sum(bool(INSTRUCTION.search(r["nl_back"])) and not INSTRUCTION.search(r["en"])
                               for r in rows),
            "empty": sum(not r["nl_back"].strip() for r in rows),
            "back_tokens_out_max": max(r["back_tokens_out"] for r in rows)}


def roundtrip(args):
    """Per source of ROUNDTRIP_SOURCES, in --out/<key>/: the files of the -gemma4 dataset (--data, as fetch
    --roundtrip wrote it), each with the same rows in the same order, every column unchanged, and the round trip's
    columns appended (back_schema()); every row gets the one translation of the shards (--shards: a local copy of
    the run's Hugging Face dataset), and every translation is used once. roundtrip.json: roundtrip_stats() per
    source, config and split, the prompt and its text."""
    import concurrent.futures
    import multiprocessing

    import pyarrow as pa
    import pyarrow.parquet as pq

    back, prompts = {}, collections.Counter()
    for path in sorted(Path(args.shards).rglob("*.parquet")):
        if ".cache" in path.relative_to(args.shards).parts:
            continue
        for r in pq.read_table(path, columns=["id", "text_nl", "ended", "tokens_out", "prompt"]).to_pylist():
            if r["id"] in back:
                raise SystemExit(f"{r['id']} twice in the shards ({path.name})")
            back[r["id"]] = r
            prompts[r["prompt"]] += 1
    if len(prompts) != 1:
        raise SystemExit(f"the shards have {len(prompts)} prompts: {dict(prompts)}")
    out = Path(args.out)
    stats, counts = {}, {}
    pool = (concurrent.futures.ProcessPoolExecutor(args.processes, mp_context=multiprocessing.get_context("spawn"))
            if args.processes > 1 else None)
    for part, (key, folder) in PARTS.items():
        for split in SPLITS:
            measured = []
            for path in source_files(args.data, key, folder, split):
                table = pq.read_table(path)
                rows = table.select(["id", "nl", "en", "hard_fails"]).to_pylist()
                missing = [r["id"] for r in rows if r["id"] not in back]
                if missing:
                    raise SystemExit(f"{key}/{folder}/{path.name}: {len(missing)} rows without a translation, e.g. "
                                     f"{missing[:3]}")
                shards = [back.pop(r["id"]) for r in rows]
                triples = [(r["nl"], r["en"], s["text_nl"], s["ended"]) for r, s in zip(rows, shards)]
                chunks = [triples[k:k + 2000] for k in range(0, len(triples), 2000)]
                new = [m for part_rows in (pool.map if pool else map)(_back_chunk, chunks) for m in part_rows]
                columns = {"nl_back": [s["text_nl"] for s in shards],
                           "back_tokens_out": [s["tokens_out"] for s in shards]}
                for name, _ in back_schema()[2:]:
                    columns[name] = [m[name] for m in new]
                result = table
                for name, kind in back_schema():
                    result = result.append_column(pa.field(name, kind), pa.array(columns[name], type=kind))
                target = out / key / folder / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(result, target, compression="zstd")
                written = pq.read_table(target)
                if not written.select(table.column_names).equals(table) or written.num_rows != table.num_rows:
                    raise SystemExit(f"{target}: the written file differs from the source in its columns")
                counts[f"{key}/{folder}/{path.name}"] = written.num_rows
                measured += [dict(r, nl_back=s["text_nl"], back_tokens_out=s["tokens_out"], **m)
                             for r, s, m in zip(rows, shards, new)]
            stats[f"{key}/{folder}/{split}"] = roundtrip_stats(measured)
            print(f"ROUNDTRIP {key}/{folder}/{split} {json.dumps(stats[f'{key}/{folder}/{split}']['chrf'])}",
                  flush=True)
    if pool:
        pool.shutdown()
    if back:
        raise SystemExit(f"{len(back)} translations of the shards belong to no row, e.g. {sorted(back)[:3]}")
    prompt = next(iter(prompts))
    result = {"created": utc(), "files": counts, "rows": sum(counts.values()), "prompt": prompt,
              "prompt_text": prompt_text(prompt, "en", "nl"), "stats": stats,
              "revisions": {k: v[1] for k, v in ROUNDTRIP_SOURCES.items()}}
    (out / "roundtrip.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"ROUNDTRIP OK {result['rows']} rows in {len(counts)} files", flush=True)


# ---------------------------------------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--out", required=True)
    p.add_argument("--roundtrip", action="store_true", help="The -gemma4 datasets (ROUNDTRIP_SOURCES) instead")
    p = sub.add_parser("probeplan")
    p.add_argument("--data", required=True)
    p.add_argument("--pilot", default=None,
                   help="JSON {sample_ids, structure_ids}: the earlier pilot's sample (left out of the strata) and its "
                        "translations that failed `structure` (stratum structure); none without it")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--data", required=True)
    p.add_argument("--text", choices=("nl", "en"), default="nl",
                   help="The column to translate: nl into English, or en back into Dutch (the round trip)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model", default=None, help=f"Default {MODEL}, with --text en {RT_MODEL}")
    p.add_argument("--max-source", type=int, default=None,
                   help="The largest passage in Gemma 4 tokens (default: the model's largest class)")
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    p.add_argument("--expect", type=int, default=EXPECTED_PASSAGES, help="Passages expected (0: no check)")
    p.add_argument("--parts", default=None, help="Only these parts, comma-separated (default: all of PARTS)")
    p.add_argument("--check-prompt", default=None,
                   help="A prompt of models.yml: every passage's prompt must fit the prefill length of its class")
    p.add_argument("--out", required=True)
    p = sub.add_parser("work")
    p.add_argument("--plan", required=True)
    p.add_argument("--root", required=True, help="gs:// directory of this region: shards/, claims/, other-done.txt")
    p.add_argument("--claims-root", default=None,
                   help="gs:// directory whose claims/ holds the claims instead of --root's: the full/ directory of "
                        "another region whose VMs take the same units, so that a unit started there is not started "
                        "here (shards and other-done.txt stay under --root)")
    p.add_argument("--length-class", required=True)
    p.add_argument("--prompt", default=None, help="The prompt of the units without one (the full plan)")
    p.add_argument("--direction", choices=("forward", "backward"), default="forward")
    p.add_argument("--owner", default=f"{socket.gethostname()}/h0")
    p.add_argument("--run", default=str(int(now())))
    p.add_argument("--region", default=None)
    p.add_argument("--lease", type=int, default=LEASE_SECONDS)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--ckpt", required=True, help="Gemma 4's MaxText checkpoint in this region: <checkpoint>/0/items")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--sync-every", type=int, default=8)
    p.add_argument("--slots", type=int, default=None,
                   help="Decode slots of the class instead of models.yml's (a class that ran out of memory there)")
    p = sub.add_parser("status")
    p.add_argument("--plan", required=True)
    p.add_argument("--root", required=True)
    p = sub.add_parser("select")
    p.add_argument("--plan", required=True)
    p.add_argument("--shards", required=True)
    p.add_argument("--qe", default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("publish")
    p.add_argument("--plan", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--owner", default=f"{socket.gethostname()}/publish")
    p.add_argument("--run", default=str(int(now())))
    p.add_argument("--work-dir", required=True)
    p.add_argument("--interval", type=int, default=600)
    p.add_argument("--batch", type=int, default=60)
    p.add_argument("--once", action="store_true")
    p = sub.add_parser("final")
    p.add_argument("--plan", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--owner", default=f"{socket.gethostname()}/final")
    p.add_argument("--run", default=str(int(now())))
    p.add_argument("--work-dir", required=True)
    p = sub.add_parser("card")
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--data", required=True, help="A local copy of the dataset (one directory per config)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("datasets")
    p.add_argument("--data", required=True, help="A local copy of the merged dataset (one directory per part)")
    p.add_argument("--sources", required=True, help="The source datasets as fetch wrote them")
    p.add_argument("--out", required=True)
    p.add_argument("--standalone", action="store_true",
                   help="Without TranslateGemma's columns, Gemma 4's under the plain names (standalone())")
    p = sub.add_parser("roundtrip")
    p.add_argument("--data", required=True, help="The -gemma4 datasets as fetch --roundtrip wrote them")
    p.add_argument("--shards", required=True, help="A local copy of the round trip's Hugging Face dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--processes", type=int, default=1)
    args = parser.parse_args(argv)
    if getattr(args, "expect", None) == 0:
        args.expect = None
    {"fetch": fetch, "probeplan": probeplan, "plan": plan, "work": work, "status": status, "select": select,
     "publish": publish, "final": final, "card": card, "report": report, "datasets": datasets,
     "roundtrip": roundtrip}[args.command](args)


if __name__ == "__main__":
    # As translate.py: a thread that is no daemon would keep a failed process alive and holding the TPU, so the
    # process ends here with the error's traceback and code
    import traceback

    code = 0
    try:
        main()
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
        if not isinstance(stop.code, int) and stop.code is not None:
            print(stop.code, file=sys.stderr)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
