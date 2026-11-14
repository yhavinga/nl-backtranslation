"""gemma4_bt/bt_long.py without a model, a bucket or Hugging Face: the ids, the crawl bound, the paragraphs taken by
existing passages, the stretches of a document, and the steps after the selection on made-up data."""

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gemma4_bt"))

import bt_long as bl  # noqa: E402


class FakeChecker:
    """The selection's filters passed by every text but those holding a word of fail_words"""

    wordset = set()

    def __init__(self, fail_words=(), fixed_by_repair=False):
        self.fail_words, self.fixed_by_repair = fail_words, fixed_by_repair

    def checks(self, source, text):
        if self.fixed_by_repair and "_Kop_ " in text:
            return ["heading_glued"]
        return ["oov"] if any(w in text for w in self.fail_words) else []

    def idioms(self, source, text):
        return [], 0.0


def words(text):
    return len(text.split())


def count(texts):
    return [words(t) for t in texts]


def sentence(i, n):
    """A paragraph of n words that is prose and starts and ends a sentence"""
    return "Dit " + " ".join(f"woord{i}x{k}" for k in range(n - 2)) + " einde."


def test_ids_of_passages_and_context_turns():
    assert bl.parse_id("fineweb:<urn:uuid:ab-cd>:d:3") == ("fineweb", "<urn:uuid:ab-cd>", "d", "3")
    assert bl.parse_id("handelingen:h-tk-2019-1:t12:0") == ("handelingen", "h-tk-2019-1", "t12", "0")
    assert bl.parse_id("handelingen:h-tk-2019-1:t11") == ("handelingen", "h-tk-2019-1", "t11", None)
    assert bl.parse_id("raad:abc:d:p3-9") == ("raad", "abc", "d", "p3-9")


def test_the_crawl_bound_is_cc_main_2023_23_inclusive():
    assert bl.dump_ok("CC-MAIN-2023-23") and bl.dump_ok("CC-MAIN-2014-10")
    assert not bl.dump_ok("CC-MAIN-2023-40") and not bl.dump_ok("CC-MAIN-2024-10")
    with pytest.raises(SystemExit, match="no CC-MAIN"):
        bl.dump_ok("2023-23")


def test_pieces_of_existing_passages_are_found_in_the_paragraphs_of_their_block():
    rows = [("fineweb:doc1:d:0", "Eerste alinea hier.\n\nTweede  alinea\nhier.", "train", "data"),
            ("handelingen:h1:t4", "Een hele beurt.", "validation", "context_turns"),
            ("handelingen:h1:t5:0", "Een zin.", "validation", "data")]
    taken, splits = bl.taken_pieces(rows)
    assert splits == {"doc1": "train", "h1": "validation"}
    assert taken[("h1", "t4")]["whole"] and not taken[("h1", "t5")]["whole"]
    paragraphs = ["Kop", "Eerste alinea hier.", "Iets anders.", "Tweede alinea hier."]
    assert bl.locate(taken[("doc1", "d")]["pieces"], paragraphs) == {1, 3}
    # a run of sentences is found in its paragraph; a council paragraph is found after its speaker label went
    assert bl.locate(["Twee zinnen."], ["Een zin. Twee zinnen. Drie."]) == {0}
    assert bl.locate(["Niet hier."], paragraphs) is None
    with pytest.raises(SystemExit, match="both splits"):
        bl.taken_pieces(rows + [("fineweb:doc1:d:1", "x", "validation", "data")])


def test_runs_of_usable_paragraphs_break_at_a_speaker_label():
    assert bl.runs([True, True, False, True, True, True], [False] * 6) == [(0, 2), (3, 6)]
    assert bl.runs([True, True, True, True], [False, False, True, False]) == [(0, 2), (2, 4)]


def stretch(paragraphs, target=60, checker=None, labels=None, usable=None, seed=0):
    labels = labels or [False] * len(paragraphs)
    counts = count(paragraphs)
    usable = usable or [True] * len(paragraphs)
    return bl.find_stretch("fineweb", paragraphs, labels, counts, usable, words, checker or FakeChecker(),
                           random.Random(seed), __import__("collections").Counter(), target=target)


def test_a_stretch_is_whole_paragraphs_above_the_minimum_and_within_the_target(monkeypatch):
    monkeypatch.setattr(bl, "MIN_TOKENS", 20)
    monkeypatch.setattr(bl, "MAX_TOKENS", 60)
    paragraphs = [sentence(i, 12) for i in range(10)]
    for seed in range(5):
        s = stretch(paragraphs, target=50, seed=seed)
        assert 20 < s["tokens"] <= 50
        assert s["nl"] == "\n\n".join(paragraphs[s["first"]:s["end"]])
        assert s["text"] == "\n".join(paragraphs[s["first"]:s["end"]])
    # a run too short for the minimum gives none
    assert stretch(paragraphs[:1]) is None
    # paragraphs that may not be used split the runs
    assert stretch(paragraphs, usable=[True, False] * 5) is None


def test_a_stretch_must_be_prose_and_pass_the_filters_and_may_pass_by_its_repaired_headings(monkeypatch):
    monkeypatch.setattr(bl, "MIN_TOKENS", 20)
    monkeypatch.setattr(bl, "MAX_TOKENS", 200)
    lists = ["- een twee drie vier vijf zes zeven acht negen tien elf twaalf"] * 4
    assert stretch(lists, target=200) is None
    bad = [sentence(i, 12) + " slecht" for i in range(4)]
    assert stretch(bad, target=200, checker=FakeChecker(fail_words=("slecht",))) is None
    heading = ["_Kop_ Dit " + " ".join(f"w{k}" for k in range(20)) + " einde."] + [sentence(i, 12) for i in range(3)]
    # only a start at the heading is long enough
    s = stretch(heading, target=200, checker=FakeChecker(fixed_by_repair=True), usable=[True, True, False, False])
    assert s["repaired"] and s["headings_repaired"] == 1 and s["nl"].startswith("_Kop_\n\nDit w0")


def fineweb_doc(paragraphs, doc_id="<urn:uuid:1>"):
    return {"id": doc_id, "text": "\n".join(paragraphs), "url": "https://www.voorbeeld.nl/a", "dump": "CC-MAIN-2020-10",
            "date": "2020-02-20T10:00:00Z"}


def test_a_document_gives_a_stretch_outside_its_existing_passage_or_none(monkeypatch):
    import collections

    # at most 40 tokens: every stretch is one paragraph of 21 words
    monkeypatch.setattr(bl, "MIN_TOKENS", 20)
    monkeypatch.setattr(bl, "MAX_TOKENS", 40)
    paragraphs = [sentence(i, 21) for i in range(8)]
    doc = fineweb_doc(paragraphs)
    taken, splits = bl.taken_pieces([("fineweb:<urn:uuid:1>:d:0", paragraphs[3] + "\n\n" + paragraphs[4],
                                      "validation", "data")])
    counters = collections.Counter()
    rows = bl.document_candidates("fineweb", doc, taken, splits, FakeChecker(), count, words, counters)
    assert len(rows) == 1
    r = rows[0]
    assert not {3, 4} & set(range(r["first"], r["last"] + 1))
    assert r["split"] == "validation" and r["doc_known"] and r["host"] == "voorbeeld.nl"
    assert r["id"] == f"fineweb:<urn:uuid:1>:d:p{r['first']}-{r['last']}"
    assert r["dump"] == "CC-MAIN-2020-10" and r["fineweb2_id"] == "<urn:uuid:1>"
    # the same document whose existing passage is in none of its paragraphs: skipped
    taken2, splits2 = bl.taken_pieces([("fineweb:<urn:uuid:1>:d:0", "Een tekst die er niet staat.", "train", "data")])
    assert bl.document_candidates("fineweb", doc, taken2, splits2, FakeChecker(), count, words, counters) == []
    assert counters["docs_piece_not_found"] == 1
    # a new document gets the split of the selection's rule
    rows = bl.document_candidates("fineweb", fineweb_doc(paragraphs, "<urn:uuid:2>"), {}, {}, FakeChecker(), count,
                                  words, counters)
    assert rows[0]["split"] == bl.split_name("fineweb", "<urn:uuid:2>") and not rows[0]["doc_known"]


def test_a_context_turn_is_taken_whole_and_a_turn_beside_it_is_free(monkeypatch):
    import collections

    monkeypatch.setattr(bl, "MIN_TOKENS", 20)
    monkeypatch.setattr(bl, "MAX_TOKENS", 40)
    lines = [sentence(i, 21) for i in range(4)]
    doc = {"id": "h-tk-1", "chamber": "tweede_kamer", "session_year": "2019-2020", "meeting_number": 3,
           "date": "2019-10-01", "title": "Een debat",
           "turns": [{"speaker": "De heer Jansen", "party": "VVD", "text": "\n".join(lines)},
                     {"speaker": "Mevrouw De Vries", "party": "D66", "text": "\n".join(lines[::-1])}]}
    taken, splits = bl.taken_pieces([("handelingen:h-tk-1:t0", "\n\n".join(lines), "train", "context_turns")])
    rows = bl.document_candidates("handelingen", doc, taken, splits, FakeChecker(), count, words,
                                  collections.Counter())
    assert [r["turn"] for r in rows] == [1]
    assert rows[0]["speaker"] == "Mevrouw De Vries" and rows[0]["party"] == "D66" and rows[0]["block"] == "t1"
    assert rows[0]["url"] == "https://zoek.officielebekendmakingen.nl/h-tk-1.html"


def test_ngram_hashes_are_stable_and_match_on_words_case_ignored():
    a = bl.ngram_hashes("Een twee drie vier vijf zes zeven acht negen tien elf twaalf dertien.")
    b = bl.ngram_hashes("een, twee drie vier vijf zes zeven acht negen tien elf twaalf DERTIEN")
    assert a == b and len(a) == 1 and bl.ngram_hashes("te kort") == set()
    assert bl.stable_hash("x") == bl.stable_hash("x")


def test_frequent_idioms_go_to_the_sources_of_their_pool(tmp_path):
    for name, counts, left in (("a", "handelingen=10,raad=5,dpc=0", [["het gaat", 9]]),
                               ("b", "fineweb=93000", [["zorgen voor", 3]])):
        (tmp_path / f"{name}.json").write_text(json.dumps({"args": {"counts": counts},
                                                           "idioms": {"left_out_list": left}}))
    out = bl.frequent_idioms([tmp_path / "a.json", tmp_path / "b.json"])
    assert out == {"handelingen": {"het gaat"}, "raad": {"het gaat"}, "fineweb": {"zorgen voor"}}


def test_bin_quotas_share_n_evenly_and_pass_a_shortfall_on():
    assert bl.bin_quotas({896: 100, 1280: 100, 1664: 100, 2048: 100}, 40) == {896: 10, 1280: 10, 1664: 10, 2048: 10}
    q = bl.bin_quotas({896: 100, 1280: 100, 1664: 100, 2048: 3}, 40)
    assert q[2048] == 3 and sum(q.values()) == 40 and max(q.values()) - min(q[b] for b in (896, 1280, 1664)) <= 1
    assert bl.bin_quotas({896: 5}, 40) == {896: 5, 1280: 0, 1664: 0, 2048: 0}


def test_pick_keeps_the_order_the_bins_and_the_topic_cap():
    rows = [{"order": i, "bin": (896, 1280, 1664, 2048)[i % 4], "topic": "A" if i < 20 else "B"} for i in range(40)]
    chosen = bl.pick(rows, 8)
    assert [r["order"] for r in chosen] == list(range(8))
    # 2 a topic in the first pass (A: 0, 1; B: 20, 21), then the capped ones in their order where their bin has room
    capped = bl.pick(rows, 8, topic_cap=0.25)
    assert [r["order"] for r in capped] == [0, 1, 2, 3, 6, 7, 20, 21]
    assert collections_count(capped) == {896: 2, 1280: 2, 1664: 2, 2048: 2}


def collections_count(rows):
    import collections

    return dict(collections.Counter(r["bin"] for r in rows))


# ---------------------------------------------------------------------------------------------------- plan, models

import bt_gemma4 as bg  # noqa: E402
import gemma4_engine as ge  # noqa: E402


class TokenCounter:
    """Gemma 4 tokens as whitespace words; a prompt's turn adds 114"""

    def __init__(self, *_):
        pass

    def many(self, texts):
        return [t.count("woord") + 114 if "<|turn>" in t else len(t.split()) for t in texts]


def long_selection(root, sizes, folder="data", en=False):
    """bt_long.py choose's layout: per source and split a file with id, nl (and en, tokens_nl for the round trip)"""
    for source in bl.SOURCE_KEYS:
        for split in bg.SPLITS:
            rows = []
            for i, n in enumerate(sizes):
                r = {"id": f"{source}:d{i}:d:p0-1", "nl": " ".join(["woord"] * n)}
                if en:
                    r.update(en=" ".join(["word"] * int(n * 0.8)), tokens_nl=n)
                rows.append(r)
            bg.write_parquet(rows, Path(root) / source / folder / f"{split}-00000-of-00001.parquet")


def test_the_long_models_hold_their_passages_and_the_prompt():
    for name in (bl.MODEL, bl.RT_MODEL):
        m = ge.MODELS[name]
        classes = m["length_classes"]
        assert list(classes) == list(m["slots"]) == list(m["fallback_slots"])
        assert {k: m[k] for k in ("maxtext_model", "revision", "chips", "eos_ids", "turn", "prompt")} == \
            {k: ge.MODELS[bg.MODEL][k] for k in ("maxtext_model", "revision", "chips", "eos_ids", "turn", "prompt")}
        for cls, c in classes.items():
            assert c["prefill_lengths"][-1] - c["max_source"] >= 128
            assert m["slots"][cls] % m["chips"] == 0 and m["fallback_slots"][cls] <= m["slots"][cls]
            # the fit of the memory measured on 14 Nov (models.yml): at most the measured 28.75 GiB of 1152 / 1280
            peak = 15.84 + 0.0003317 * m["slots"][cls] * (c["prefill_lengths"][-1] + c["max_decode"]) \
                + 0.0014 * (c["prefill_lengths"][-1] - 1152)
            assert peak <= 28.76, (name, cls, peak)
    nl = ge.MODELS[bl.MODEL]["length_classes"]
    assert list(nl.values())[-1]["max_source"] == bl.MAX_TOKENS
    for c in nl.values():
        assert c["max_decode"] == 1.25 * c["max_source"]
    rt = ge.MODELS[bl.RT_MODEL]
    assert rt["move_up"] and max(c["max_decode"] for c in rt["length_classes"].values()) >= 1.25 * bl.MAX_TOKENS
    # the longest English an nl->en class can give fits the round trip's largest class
    assert max(c["max_decode"] for c in nl.values()) <= max(c["max_source"] for c in rt["length_classes"].values())


def test_plan_of_the_long_passages_takes_their_parts_classes_and_unit_sizes(tmp_path, monkeypatch):
    monkeypatch.setattr(bg, "Counter", TokenCounter)
    long_selection(tmp_path / "sel", [600, 700, 1000, 2000])
    bg.main(["plan", "--data", str(tmp_path / "sel"), "--model", bl.MODEL, "--tokenizer", "-", "--out",
             str(tmp_path / "plan"), "--expect", "0", "--parts", ",".join(bl.SOURCE_KEYS),
             "--check-prompt", "natural-v1-fmt"])
    info = bg.load_plan(tmp_path / "plan")
    assert info["parts"] == list(bl.SOURCE_KEYS) and info["passages"] == 5 * 2 * 4
    assert {u["class"] for u in info["units"]} == {"l640", "l768", "l1k", "l2k"}
    assert info["shard_size"] == {c: 16 * s for c, s in ge.MODELS[bl.MODEL]["slots"].items()}
    assert info["longest_prompt"]["l2k"] == 2000 + 114
    with pytest.raises(SystemExit, match="are no parts"):
        bg.main(["plan", "--data", str(tmp_path / "sel"), "--model", bl.MODEL, "--tokenizer", "-", "--out",
                 str(tmp_path / "p2"), "--expect", "0", "--parts", "fineweb,mc4"])
    long_selection(tmp_path / "sel2", [2040])
    with pytest.raises(SystemExit, match="more than the prefill length 2176"):
        monkeypatch.setattr(TokenCounter, "many", lambda self, texts: [
            t.count("woord") + 200 if "<|turn>" in t else len(t.split()) for t in texts])
        bg.main(["plan", "--data", str(tmp_path / "sel2"), "--model", bl.MODEL, "--tokenizer", "-", "--out",
                 str(tmp_path / "p3"), "--expect", "0", "--parts", ",".join(bl.SOURCE_KEYS),
                 "--check-prompt", "natural-v1-fmt"])


def test_the_round_trip_of_long_passages_moves_a_passage_up_to_a_class_that_holds_its_dutch(tmp_path, monkeypatch):
    monkeypatch.setattr(bg, "Counter", TokenCounter)
    long_selection(tmp_path / "en", [600, 1000, 2000], en=True)
    bg.main(["plan", "--data", str(tmp_path / "en"), "--text", "en", "--model", bl.RT_MODEL, "--tokenizer", "-",
             "--out", str(tmp_path / "plan"), "--expect", "0", "--parts", ",".join(bl.SOURCE_KEYS)])
    info = bg.load_plan(tmp_path / "plan")
    # 480 English, 600 Dutch: r640; 800 English, 1000 Dutch: r1k (r1k holds 896, 1.25 x 1000 = 1250 <= 1280);
    # 1600 English, 2000 Dutch: r1792 holds the English but not 2500 of decoding, r2k both
    assert sorted({u["class"] for u in info["units"]}) == ["r1k", "r2k", "r640"]
    assert bg.class_holding(1600, 2500, ge.MODELS[bl.RT_MODEL]["length_classes"]) == "r2k"
    assert bg.class_holding(3000, 10, ge.MODELS[bl.RT_MODEL]["length_classes"]) is None


def test_work_takes_the_slots_given_for_its_class():
    args = type("Args", (), {"length_class": "l2k", "slots": None})()
    plan_info = {"model": bl.MODEL, "classes": ge.MODELS[bl.MODEL]["length_classes"]}
    assert bg.Translator(args, plan_info).shapes["slots"] == 4
    args.length_class, args.slots = "l1k", 12
    t = bg.Translator(args, plan_info)
    assert t.shapes["slots"] == 12 and t.shapes["max_decode"] == 1280


# ---------------------------------------------------------------------------------------------------- after the selection

# The columns of the published files (data/) of each -gemma4 dataset, in their order (14 Nov 2026)
TAIL = ["en", "tokens_nl", "sentences_nl", "source_score", "origin_score", "topic", "half", "idiom_class",
        "idiom_score", "headings_repaired", "drawn_in_pass"]
CHECKS = ["ended", "tokens_out", "prompt", "prompt_text", "model", "hard_fails", "structure", "chatter", "decode_loop",
          "dutch_left", "len_ratio", "numbers_missing", "word_ratio", "char_ratio", "region", "vm", "time", "nl_back",
          "back_tokens_out", "back_ended", "chrf", "back_decode_loop", "back_english_left", "back_structure",
          "back_chatter"]
GPTNL = ["id", "gptnl_doc_id", "gptnl_source_id", "content_type", "content_quality", "content_integrity",
         "pii_presence", "nl"]
PUBLISHED = {
    "fineweb": ["id", "fineweb2_id", "url", "host", "dump", "date", "nl"] + TAIL + CHECKS,
    "handelingen": ["id", "handelingen_id", "url", "session_year", "meeting_number", "date", "title", "turn", "speaker",
                    "party", "nl"] + TAIL + ["context_ids"] + CHECKS,
    "raad": GPTNL + TAIL + ["repair_pass"] + CHECKS,
    "dpc": GPTNL + TAIL + CHECKS,
    "rechtspraak": GPTNL + TAIL + CHECKS,
}


def selected(source, i, tokens=600, split="train", order=0):
    meta = {"fineweb": {"fineweb2_id": f"<urn:{i}>", "url": "https://a.nl/x", "host": "a.nl", "dump": "CC-MAIN-2020-10",
                        "date": "2020-02-20T10:00:00Z"},
            "handelingen": {"handelingen_id": f"h-{i}", "url": f"https://zoek.officielebekendmakingen.nl/h-{i}.html",
                            "session_year": "2019-2020", "meeting_number": 3, "date": "2019-10-01", "title": "Debat",
                            "turn": 4, "speaker": "De heer A", "party": "VVD"}}.get(source) or {
        "gptnl_doc_id": f"g{i}", "gptnl_source_id": "s", "content_type": ["news"], "content_quality": "good",
        "content_integrity": "complete", "pii_presence": "no_pii"}
    return {"id": f"{source}:d{i}:d:p0-3", "source": source, "doc_id": f"d{i}", "split": split, "order": order,
            "nl": "Dit is de eerste zin.\n\nEn dit is de tweede zin.", "tokens_nl": tokens, "headings_repaired": 0,
            "repaired": False, "idiom_score": 0.5, "source_score": 0.99, "origin_score": 0.98, "topic": "Health and care",
            "bin": bl.bin_of(tokens), **meta}


def test_a_long_row_has_exactly_the_columns_of_its_dataset():
    en = {"text_en": "This is the first sentence.\n\nAnd this is the second sentence.", "ended": True, "tokens_out": 20,
          "prompt": "natural-v1-fmt", "region": "us-central2", "owner": "t1v-n-1-w-0/h0", "time": "2026-11-15T01:00:00Z"}
    back = {"text_nl": "Dit is de eerste zin.\n\nEn dit is de tweede zin.", "ended": True, "tokens_out": 21}
    for source, columns in PUBLISHED.items():
        row = bl.long_row(source, selected(source, 1), en, back)
        assert set(row) == set(columns), (source, set(row) ^ set(columns))
        assert row["half"] == "random" and row["idiom_class"] is None and row["drawn_in_pass"] is None
        assert row["hard_fails"] == [] and row["chrf"] == 100.0 and row["sentences_nl"] == 2
        assert row["prompt_text"].startswith("Translate the following Dutch text into natural")
    assert bl.long_row("handelingen", selected("handelingen", 1), en, back)["context_ids"] == []
    assert bl.long_row("raad", dict(selected("raad", 1), repaired=True), en, back)["repair_pass"] is True


def write_published(root, source, rows):
    """A fake -gemma4 dataset of one source with the published column order"""
    import pyarrow as pa

    table = pa.Table.from_pylist(rows).select(PUBLISHED[source])
    path = Path(root) / source / "data" / "train-00000-of-00001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq

    pq.write_table(table, path)


def test_choose_english_and_build_make_the_new_rows_of_every_split(tmp_path):
    import pyarrow.parquet as pq

    # candidates of every source: 6 each, one validation; scores: one below the source threshold, and for FineWeb-2
    # one below the origin threshold
    scores = []
    for source in bl.SOURCE_KEYS:
        rows = [dict(selected(source, i, tokens=600 + 400 * (i % 4), split="validation" if i == 5 else "train",
                              order=i)) for i in range(6)]
        bg.write_parquet(rows, tmp_path / "cand" / f"{source}.parquet")
        for i, r in enumerate(rows):
            scores.append({"id": r["id"], "score": 0.2 if i == 1 else 0.9, "origin_score": 0.1 if i == 2 else 0.9,
                           "topic": "D", "skipped": None})
    bg.write_parquet(scores, tmp_path / "scores" / "w0" / "shard-00000.parquet")
    bl.main(["choose", "--candidates", str(tmp_path / "cand"), "--scores", str(tmp_path / "scores"), "--out",
             str(tmp_path / "sel"), "--per-source", "3"])
    result = json.loads((tmp_path / "sel" / "selection.json").read_text())
    assert result["sources"]["fineweb"]["reasons"] == {"eligible": 4, "source_below": 1, "origin_below": 1}
    assert result["sources"]["raad"]["reasons"] == {"eligible": 5, "source_below": 1}
    assert result["sources"]["raad"]["chosen"] == 3
    chosen = bl.selection_rows(tmp_path / "sel", "raad", "train") + bl.selection_rows(tmp_path / "sel", "raad",
                                                                                         "validation")
    assert all(r["topic"] == "Health and care" for r in chosen)
    # the English and the round trip of every chosen passage, as the units' shards
    en_rows, rt_rows = [], []
    for source in bl.SOURCE_KEYS:
        for split in bg.SPLITS:
            for r in bl.selection_rows(tmp_path / "sel", source, split):
                en_rows.append({"id": r["id"], "text_en": "This is the first sentence.\n\nAnd this is the second.",
                                "ended": True, "tokens_in": 9, "tokens_out": 12, "unit": f"{source}/{split}-l640-0000",
                                "prompt": "natural-v1-fmt", "owner": "w-0/h0", "region": "us-central2",
                                "time": "2026-11-15T01:00:00Z"})
                rt_rows.append({"id": r["id"], "text_nl": r["nl"], "ended": True, "tokens_in": 9, "tokens_out": 13,
                                "unit": "x", "prompt": "natural-v1-fmt", "owner": "w-0/h0", "region": "us-central2",
                                "time": "2026-11-15T02:00:00Z"})
    bg.write_parquet(en_rows, tmp_path / "en-shards" / "shards" / "a.parquet")
    bg.write_parquet(rt_rows, tmp_path / "rt-shards" / "shards" / "a.parquet")
    bl.main(["english", "--selection", str(tmp_path / "sel"), "--shards", str(tmp_path / "en-shards"), "--out",
             str(tmp_path / "english")])
    eng = pq.read_table(tmp_path / "english" / "dpc" / "data" / "train-00000-of-00001.parquet")
    assert eng.column_names == ["id", "nl", "en", "tokens_nl"]
    en = {"text_en": "x", "ended": True, "tokens_out": 1, "prompt": "natural-v1-fmt", "region": "r", "owner": "o",
          "time": "t"}
    for source in bl.SOURCE_KEYS:
        write_published(tmp_path / "existing", source, [bl.long_row(source, selected(source, 9), en,
                                                                    {"text_nl": "y", "ended": True, "tokens_out": 1})])
    (tmp_path / "existing" / "heads.json").write_text(json.dumps({s: "abc" for s in bl.SOURCE_KEYS}))
    bl.main(["build", "--selection", str(tmp_path / "sel"), "--existing", str(tmp_path / "existing"), "--en-shards",
             str(tmp_path / "en-shards"), "--rt-shards", str(tmp_path / "rt-shards"), "--out", str(tmp_path / "build")])
    built = json.loads((tmp_path / "build" / "build.json").read_text())
    assert built["rows"]["fineweb/train"] + built["rows"]["fineweb/validation"] == 3
    table = pq.read_table(tmp_path / "build" / "handelingen" / "data" / "train.parquet")
    assert table.column_names == PUBLISHED["handelingen"]
    assert built["stats"]["raad/train"]["n"] == built["rows"]["raad/train"]
    # a chosen passage without its round trip stops the build
    bg.write_parquet(rt_rows[1:], tmp_path / "rt2" / "a.parquet")
    with pytest.raises(SystemExit, match="no English or no round trip"):
        bl.main(["build", "--selection", str(tmp_path / "sel"), "--existing", str(tmp_path / "existing"),
                 "--en-shards", str(tmp_path / "en-shards"), "--rt-shards", str(tmp_path / "rt2"), "--out",
                 str(tmp_path / "b2")])
    # an English twice stops the round trip's input
    bg.write_parquet(en_rows[:1], tmp_path / "en-shards" / "shards" / "b.parquet")
    with pytest.raises(SystemExit, match="twice in the shards"):
        bl.main(["english", "--selection", str(tmp_path / "sel"), "--shards", str(tmp_path / "en-shards"), "--out",
                 str(tmp_path / "e2")])


def test_smoke_plan_has_two_units_a_class_of_the_longest_passages_it_holds(tmp_path):
    for source in bl.SOURCE_KEYS:
        rows = [selected(source, i, tokens=t, order=i) for i, t in enumerate(range(513, 2049, 2))]
        bg.write_parquet(rows, tmp_path / "cand" / f"{source}.parquet")
    bl.main(["smokeplan", "--candidates", str(tmp_path / "cand"), "--out", str(tmp_path / "smoke")])
    for name, model in (("smoke-nl", bl.MODEL), ("smoke-rt", bl.RT_MODEL)):
        info = bg.load_plan(tmp_path / "smoke" / name)
        m = ge.MODELS[model]
        assert [u["name"] for u in info["units"]] == [f"smoke/train-{c}-{k:04d}" for c in m["length_classes"]
                                                     for k in range(bl.SMOKE_UNITS)]
        for u in info["units"]:
            rows = bg.read_rows(tmp_path / "smoke" / name / u["segments"])
            assert len(rows) == bl.SMOKE_PER_SLOT * m["slots"][u["class"]] == u["n"]
            assert max(r["tokens"] for r in rows) <= m["length_classes"][u["class"]]["max_source"]
    assert bg.languages(bg.load_plan(tmp_path / "smoke" / "smoke-rt")) == ("en", "nl")
    # the nl->en classes get passages of their own length range
    info = bg.load_plan(tmp_path / "smoke" / "smoke-nl")
    rows = bg.read_rows(tmp_path / "smoke" / "smoke-nl" / info["units"][-1]["segments"])
    assert min(r["tokens"] for r in rows) > 1792


def smoke_log(cls, units, slots, tokens=(1000, 1000), seconds=(20.0, 10.0)):
    lines = [f"MAXTEXT {json.dumps({'class': cls})}", "engine ready in 50 s"]
    for k in range(units):
        name = f"smoke/train-{cls}-{k:04d}"
        # the order of bt_gemma4.py work: the engine's MEM line before the unit's SHARD line
        lines.append("MEM " + json.dumps({"unit": name, "slots": slots, "peak_gib_max": 23.5}))
        lines.append("SHARD " + json.dumps({"unit": name, "tokens_out": tokens[k], "seconds": seconds[k],
                                            "out_per_s": round(tokens[k] / seconds[k]), "not_ended": 0,
                                            "segments": 8}))
    return lines


def test_smoke_measures_give_the_steady_speed_compilation_and_start_per_class():
    units, starts = bl.smoke_measures(smoke_log("l2k", 2, 4))
    rates = bl.class_rates(units, starts)
    # steady 100 tokens/s; the first unit's 1000 tokens took 20 s, 10 of them compilation
    assert rates["l2k"] == {"out_per_s": 100.0, "compile_s": 10.0, "start_s": 50.0, "slots": 4, "peak_gib": 23.5,
                            "not_ended": 0, "segments": 16}
    with pytest.raises(SystemExit, match="2 needed"):
        bl.class_rates(*bl.smoke_measures(smoke_log("l2k", 1, 4)))


def test_the_budget_lowers_the_passages_per_source_until_the_estimate_fits(tmp_path):
    nl = {c: {"out_per_s": 100.0, "compile_s": 60.0, "start_s": 100.0} for c in ge.MODELS[bl.MODEL]["length_classes"]}
    rt = {c: {"out_per_s": 100.0, "compile_s": 60.0, "start_s": 100.0} for c in ge.MODELS[bl.RT_MODEL]["length_classes"]}
    out_ratio = {c: 0.8 for c in nl}
    rows = [{"id": str(i), "tokens_nl": 1000} for i in range(10)]
    est = bl.estimate(rows, nl, rt, out_ratio, 1.0, workers=8)
    # 10 x (800 / 100 + 1000 / 100) s of decoding; one class each way, 160 s each, on 8 workers
    assert est["decode_worker_hours"] == round(180 / 3600, 2) and est["fixed_worker_hours"] == round(2 * 160 * 8 / 3600, 2)
    smoke = {"nl": nl, "rt": rt, "out_ratio": out_ratio}
    (tmp_path / "smoke.json").write_text(json.dumps(smoke))
    (tmp_path / "report.json").write_text(json.dumps({"back_ratio": 1.0}))
    for source in bl.SOURCE_KEYS:
        bg.write_parquet([{"id": f"{source}{i}", "tokens_nl": 2000, "order": i} for i in range(1000)],
                         tmp_path / "sel" / source / "data" / "train-00000-of-00001.parquet")
    # per passage (1600 + 2000) / 100 = 36 s; 5 sources x n x 36 s + 2 x 160 x 8 s <= 2 worker-hours: n <= 25
    bl.main(["budget", "--smoke", str(tmp_path / "smoke.json"), "--report", str(tmp_path / "report.json"),
             "--selection", str(tmp_path / "sel"), "--worker-hours", "2", "--step", "5", "--out",
             str(tmp_path / "budget.json")])
    b = json.loads((tmp_path / "budget.json").read_text())
    assert b["n"] == 25 and b["per_source"] == {s: 25 for s in bl.SOURCE_KEYS} and b["target"] == 1000
    assert b["estimate"]["worker_hours"] <= 2
