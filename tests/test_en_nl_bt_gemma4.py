"""gemma4_bt/ without a model or a bucket: the claims and the work loop on a store in memory, the units of the plan,
the probe's draw, the rule that chooses the prompt, the rows of the merged dataset, the datasets per source, the report,
the prompts and the decoding loop. No jax: run with --noconftest where the conftest's imports (jax, flax) are not
installed. The test of the real prompt lengths runs where Gemma 4's tokenizer is (GEMMA4_TOKENIZER, default
/dev/shm/bt_gemma4/tokenizer-gemma4, where gemma4_bt/setup.sh puts it)."""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gemma4_bt"))

import bt_gemma4 as bg  # noqa: E402
import gemma4_engine as ge  # noqa: E402

GEMMA4_TOKENIZER = Path(os.environ.get("GEMMA4_TOKENIZER", "/dev/shm/bt_gemma4/tokenizer-gemma4"))
ME = {"owner": "vm-a/h0", "run": "1"}


# ---------------------------------------------------------------------------------------------------- claims

def test_a_claim_is_made_once_and_refused_to_another_while_fresh():
    store = bg.MemoryStore()
    assert bg.claim(store, "claims/u", ME, lease=600, t=1000)
    assert not bg.claim(store, "claims/u", {"owner": "vm-b/h0", "run": "9"}, lease=600, t=1100)
    assert json.loads(store.read("claims/u")[0])["owner"] == "vm-a/h0"


def test_a_stale_claim_is_taken_over():
    store = bg.MemoryStore()
    assert bg.claim(store, "claims/u", ME, lease=600, t=1000)
    assert bg.claim(store, "claims/u", {"owner": "vm-b/h1", "run": "9"}, lease=600, t=1601)
    assert json.loads(store.read("claims/u")[0])["owner"] == "vm-b/h1"


def test_a_claim_of_an_earlier_run_on_the_same_vm_is_taken_over_at_once():
    store = bg.MemoryStore()
    assert bg.claim(store, "claims/u", {"owner": "vm-a/h1", "run": "1"}, lease=600, t=1000)
    # the VM came back after a preemption under the same name: a new run, either half
    assert bg.claim(store, "claims/u", {"owner": "vm-a/h0", "run": "2"}, lease=600, t=1001)
    # the other half of the same run does not take it from its sibling
    assert not bg.claim(store, "claims/u", {"owner": "vm-a/h1", "run": "2"}, lease=600, t=1002)


def test_a_lease_is_renewed_by_its_holder_and_free_once_released():
    store = bg.MemoryStore()
    assert bg.claim(store, "publish.lock", ME, lease=600, t=1000)
    assert bg.claim(store, "publish.lock", ME, lease=600, t=1500)
    assert json.loads(store.read("publish.lock")[0])["time"] == 1500
    other = {"owner": "vm-b/publish", "run": "5"}
    assert not bg.claim(store, "publish.lock", other, lease=600, t=1600)
    bg.release(store, "publish.lock", ME)
    assert bg.claim(store, "publish.lock", other, lease=600, t=1601)


def test_a_replace_with_an_old_generation_fails():
    store = bg.MemoryStore()
    store.create("k", "a")
    gen = store.read("k")[1]
    assert store.replace("k", "b", gen)
    assert not store.replace("k", "c", gen)
    assert store.read("k")[0] == "b"


# ---------------------------------------------------------------------------------------------------- units, work

def segment_rows(prefix, n, tokens=10):
    return [{"id": f"{prefix}:{i:05d}", "text": f"tekst {i}", "tokens": tokens} for i in range(n)]


def test_units_are_consecutive_ids_of_one_part_split_and_class_in_class_order():
    segs = {("raad", "train", "small"): ("segments/raad/train-small.parquet", segment_rows("raad", 5)),
            ("fineweb", "train", "medium"): ("segments/fineweb/train-medium.parquet", segment_rows("fw", 3)),
            ("fineweb", "validation", "small"): ("segments/fineweb/validation-small.parquet", segment_rows("fv", 2))}
    units = bg.make_units(segs, ["tiny", "small", "medium"], 2, list(bg.PARTS))
    assert [u["name"] for u in units] == ["fineweb/validation-small-0000", "raad/train-small-0000",
                                          "raad/train-small-0001", "raad/train-small-0002",
                                          "fineweb/train-medium-0000", "fineweb/train-medium-0001"]
    last = units[3]
    assert (last["start"], last["end"], last["n"], last["first"], last["last"]) == (4, 5, 1, "raad:00004", "raad:00004")
    assert bg.hf_path("raad/train-small-0002") == "raad/train-small-0002.parquet"
    assert bg.plan_hash(units) == bg.plan_hash([dict(u) for u in units])
    assert bg.plan_hash(units) != bg.plan_hash(units[:-1])


def fake_translate(calls):
    def translate(u):
        calls.append(u["name"])
        ids = [f"{u['name']}#{k}" for k in range(u["n"])]
        return {"id": ids, "text_en": ["x"] * u["n"], "ended": [True] * u["n"], "tokens_in": [5] * u["n"],
                "tokens_out": [3] * u["n"]}, 1.0
    return translate


def units_of(names, prompt="natural-v1-fmt"):
    return [{"name": n, "n": 2, "prompt": prompt, "class": "small"} for n in names]


def test_two_processes_share_the_units_without_doing_one_twice(tmp_path):
    store = bg.MemoryStore()
    names = [f"raad/train-small-{k:04d}" for k in range(4)]
    a, b = [], []
    me_a = dict(ME, work_dir=str(tmp_path / "a"), region="us-east1")
    me_b = {"owner": "vm-b/h0", "run": "7", "work_dir": str(tmp_path / "b"), "region": "us-east1"}
    # b already holds the second unit
    assert bg.claim(store, f"claims/{names[1]}", me_b, bg.LEASE_SECONDS)
    done_a = bg.work_units(units_of(names), store, me_a, fake_translate(a), log=lambda s: None)
    assert done_a == [names[0], names[2], names[3]] and a == done_a
    done_b = bg.work_units(units_of(names), store, me_b, fake_translate(b), progress=bg.Progress(store, every=0),
                           log=lambda s: None)
    assert done_b == [names[1]]
    assert store.list("shards/") == {f"{n}.parquet" for n in names}
    shard = tmp_path / "s.parquet"
    store.get(f"shards/{names[0]}.parquet", shard)
    row = bg.read_rows(shard)[0]
    assert (row["unit"], row["prompt"], row["owner"], row["region"]) == (names[0], "natural-v1-fmt", "vm-a/h0", "us-east1")


def test_units_done_in_the_other_region_are_skipped_and_count_for_the_status(tmp_path):
    store = bg.MemoryStore()
    names = [f"dpc/train-long-{k:04d}" for k in range(3)]
    store.create("other-done.txt", names[2] + "\n")
    calls = []
    done = bg.work_units(units_of(names), store, dict(ME, work_dir=str(tmp_path)), fake_translate(calls),
                         log=lambda s: None)
    assert done == names[:2]
    plan_info = {"units": [{"name": n, "class": "long"} for n in names + ["dpc/train-long-0003"]]}
    s = bg.region_status(plan_info, bg.Progress(store))
    assert s["classes"]["long"] == {"units": 4, "here": 2, "other": 1, "pending": 1} and not s["complete"]
    store.create("shards/dpc/train-long-0003.parquet", "x")
    assert bg.region_status(plan_info, bg.Progress(store))["complete"]


def test_backward_is_the_reverse_order():
    assert bg.ordered([1, 2, 3], "backward") == [3, 2, 1] and bg.ordered([1, 2, 3], "forward") == [1, 2, 3]


class WordCounter:
    def __init__(self, *_):
        pass

    def many(self, texts):
        return [len(t.split()) * 10 for t in texts]


def write_source(root, key, folder, split, rows):
    bg.write_parquet(rows, Path(root) / key / folder / f"{split}-00000-of-00001.parquet")


def fake_hf(root, n_per=3):
    for part, (key, folder) in bg.PARTS.items():
        for split in bg.SPLITS:
            rows = [{"id": f"{part}:{split}:{i}", "nl": " ".join(["woord"] * (i + 1)), "en": "word " * (i + 1)}
                    for i in range(n_per)]
            write_source(root, key, folder, split, rows)


def test_plan_puts_every_passage_in_one_unit_and_stops_at_a_passage_too_long(tmp_path, monkeypatch):
    monkeypatch.setattr(bg, "Counter", WordCounter)
    fake_hf(tmp_path / "hf")
    bg.main(["plan", "--data", str(tmp_path / "hf"), "--tokenizer", "-", "--out", str(tmp_path / "plan"),
             "--shard-size", "2", "--expect", str(len(bg.PARTS) * 2 * 3)])
    info = json.loads((tmp_path / "plan" / "units.json").read_text())
    covered = []
    for u in info["units"]:
        rows = bg.read_rows(tmp_path / "plan" / u["segments"])[u["start"]:u["end"]]
        assert [r["id"] for r in rows][0] == u["first"] and len(rows) == u["n"]
        covered += [r["id"] for r in rows]
    assert len(covered) == len(set(covered)) == len(bg.PARTS) * 2 * 3
    # 10, 20, 30 tokens: all tiny (max_source 32)
    assert {u["class"] for u in info["units"]} == {"tiny"}
    with pytest.raises(SystemExit, match="above 15 tokens"):
        bg.main(["plan", "--data", str(tmp_path / "hf"), "--tokenizer", "-", "--out", str(tmp_path / "p2"),
                 "--max-source", "15", "--expect", "0"])
    with pytest.raises(SystemExit, match="expected 5"):
        bg.main(["plan", "--data", str(tmp_path / "hf"), "--tokenizer", "-", "--out", str(tmp_path / "p3"),
                 "--expect", "5"])
    with pytest.raises(SystemExit, match="beyond the largest class"):
        bg.main(["plan", "--data", str(tmp_path / "hf"), "--tokenizer", "-", "--out", str(tmp_path / "p4"),
                 "--max-source", "4096"])


# ---------------------------------------------------------------------------------------------------- probe, rule

def probe_row(i, source, nl_words=10, en_words=10, length_class="short"):
    return {"id": f"{source}:{i:04d}", "nl": " ".join(["woord"] * nl_words), "en": " ".join(["word"] * en_words),
            "qe_en_nl": 1.0, "length_class": length_class, "hard_fails": [], "ended": True, "tokens_nl": 9,
            "tokens_en": 9}


def test_probe_draw_takes_fresh_rows_the_pilots_structure_cases_and_context_turns():
    rows = {s: [probe_row(i, s) for i in range(120)] + [probe_row(500 + i, s, length_class="long") for i in range(20)]
            + [probe_row(700 + i, s, nl_words=5) for i in range(30)] for s in ("dpc", "raad")}
    pilot = {"sample_ids": [f"dpc:{i:04d}" for i in range(50)],
             "structure_ids": [f"dpc:{i:04d}" for i in range(10)] + ["raad:0001"]}
    context = [dict(probe_row(i, "ctx"), length_class=None) for i in range(30)]
    out, info = bg.probe_draw(rows, context, pilot)
    by = {}
    for r in out:
        by.setdefault((r["source"], r["stratum"]), []).append(r)
    assert len(by[("dpc", "random")]) == 40 and len(by[("dpc", "long")]) == 7 and len(by[("dpc", "ratio_low")]) == 7
    assert not {r["id"] for s in ("random", "long", "ratio_low") for r in by[("dpc", s)]} & set(pilot["sample_ids"])
    assert len(by[("dpc", "structure")]) == 6 and {r["id"] for r in by[("dpc", "structure")]} <= set(pilot["structure_ids"])
    assert len(by[("raad", "structure")]) == 1 and info["raad"]["short"]["structure"] == [1, 6]
    assert len(by[("handelingen", "context")]) == 20
    assert all(r["length_class"] == "long" for r in by[("raad", "long")])


def stats(**prompts):
    return {p: {"n": 200, "hard": h, "median": m} for p, (h, m) in prompts.items()}


def test_the_rule_takes_the_eligible_prompt_closest_to_1_06():
    prompt, reason, table = bg.choose({"natural-v1": {"n": 200, "hard": 10, "median": 1.060},
                                       "natural-v1-fmt": {"n": 200, "hard": 3, "median": 1.030},
                                       "natural-v1-fmt-len": {"n": 200, "hard": 2, "median": 1.075}})
    # natural-v1 is closest but not eligible (5%); fmt-len (0.015 away) beats fmt (0.030 away) by more than 0.01
    assert prompt == "natural-v1-fmt-len" and reason.startswith("rule: eligible")
    assert [t["eligible"] for t in table] == [False, True, True]


def test_the_rule_takes_fewer_failures_when_the_distances_differ_by_less_than_0_01():
    prompt, reason, _ = bg.choose({"natural-v1-fmt": {"n": 200, "hard": 3, "median": 1.055},
                                   "natural-v1-fmt-len": {"n": 200, "hard": 1, "median": 1.047}})
    assert prompt == "natural-v1-fmt-len" and "fewest" in reason
    prompt, _, _ = bg.choose({"natural-v1-fmt": {"n": 200, "hard": 3, "median": 1.058},
                              "natural-v1-fmt-len": {"n": 200, "hard": 1, "median": 1.040}})
    assert prompt == "natural-v1-fmt"


def test_the_rule_bound_is_3_of_200_and_without_an_eligible_prompt_the_fewest_failures():
    _, _, table = bg.choose({"a": {"n": 200, "hard": 3, "median": 1.0}, "b": {"n": 200, "hard": 4, "median": 1.06}})
    assert [t["eligible"] for t in table] == [True, False]
    prompt, reason, _ = bg.choose({"a": {"n": 200, "hard": 9, "median": 1.06}, "b": {"n": 200, "hard": 5, "median": 0.9}})
    assert prompt == "b" and reason.startswith("no prompt")


# ---------------------------------------------------------------------------------------------------- dataset rows

def source_table_rows():
    return [{"id": "dpc:a", "nl": "De Kamer stemt in met de motie over de begroting.", "en": "The House agrees with the "
             "motion on the budget of the ministry.", "hard_fails": [], "structure": [], "qe_en_nl": 1.5},
            {"id": "dpc:b", "nl": "Activiteit.\n\nDe raad besluit.", "en": "Activity.\n\nThe council decides.",
             "hard_fails": ["structure"], "structure": ["italic"], "qe_en_nl": 2.0}]


def test_unit_files_keep_the_source_columns_and_one_schema_with_empty_lists(tmp_path):
    import pyarrow.parquet as pq

    write_source(tmp_path / "hf", "dpc", "data", "train", source_table_rows())
    sources = bg.SourceRows(tmp_path / "hf")
    unit = {"name": "dpc/train-small-0000", "part": "dpc", "split": "train"}
    shard = {"id": ["dpc:a", "dpc:b"], "text_en": ["The House agrees with the motion on the budget.",
                                                   "Here is the translation:\n\nActivity.\n\nThe council decides."],
             "ended": [True, False], "tokens_out": [12, 15], "prompt": ["natural-v1-fmt"] * 2,
             "owner": ["vm/h0"] * 2, "region": ["us-east1"] * 2, "time": ["t"] * 2}
    bg.write_parquet([dict(zip(shard, v)) for v in zip(*shard.values())], tmp_path / "shard.parquet")
    n = bg.unit_file(unit, tmp_path / "shard.parquet", sources, lambda u: {"dpc:a": 11, "dpc:b": 7},
                     tmp_path / "out" / "dpc" / "train-small-0000.parquet")
    table = pq.read_table(tmp_path / "out" / "dpc" / "train-small-0000.parquet")
    assert n == 2 and table.column_names[:6] == ["id", "nl", "en", "hard_fails", "structure", "qe_en_nl"]
    import pyarrow as pa

    for name in ("gemma4_numbers_missing", "gemma4_hard_fails", "gemma4_structure"):
        kind = table.schema.field(name).type
        assert pa.types.is_list(kind) and pa.types.is_string(kind.value_type), name
    rows = table.to_pylist()
    assert rows[0]["gemma4_hard_fails"] == [] and rows[0]["gemma4_word_ratio"] == round(10 / 9, 4)
    assert rows[0]["tg_word_ratio"] == round(10 / 12, 4) and rows[0]["gemma4_tokens_nl"] == 11
    assert rows[1]["gemma4_hard_fails"] == ["not_ended", "len_ratio", "structure", "chatter"] and rows[1]["gemma4_chatter"]
    assert rows[0]["gemma4_model"] == bg.MODEL_ID and "{text}" in rows[0]["gemma4_prompt_text"]
    assert rows[1]["hard_fails"] == ["structure"]  # TranslateGemma's own checks unchanged


def test_a_shard_with_a_passage_of_another_part_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="is not a passage"):
        bg.published_rows({"name": "u", "part": "dpc", "split": "train"},
                          {"id": ["x"], "text_en": ["y"], "ended": [True], "tokens_out": [1], "prompt": ["natural-v1"],
                           "owner": ["o"], "region": ["r"], "time": ["t"]}, {}, {"x": 1})


def test_card_has_one_config_per_part_and_the_prompt(tmp_path):
    import yaml

    bg.main(["card", "--prompt", "natural-v1-fmt", "--out", str(tmp_path / "README.md")])
    text = (tmp_path / "README.md").read_text()
    meta = yaml.safe_load(text.split("---\n")[1])
    assert [c["config_name"] for c in meta["configs"]] == list(bg.PARTS)
    assert meta["configs"][2]["data_files"][0] == {"split": "train", "path": "handelingen_context/train-*"}
    assert "Do not add line breaks" in text and "TPU Research Cloud" in text


def test_report_counts_rows_and_ratios_per_part(tmp_path):
    rows = []
    for k in range(10):
        rows.append({"id": f"dpc:{k}", "nl": "een twee drie", "en": "one two three four", "en_gemma4": "one two three",
                     "hard_fails": [], "gemma4_hard_fails": ["structure"] if k == 0 else [],
                     "gemma4_structure": ["italic"] if k == 0 else [], "gemma4_ended": True,
                     "gemma4_word_ratio": 1.0, "tg_word_ratio": 0.75, "gemma4_r2_length_out": False,
                     "gemma4_prompt": "natural-v1-fmt", "gemma4_region": "us-east1"})
    bg.write_parquet(rows, tmp_path / "ds" / "dpc" / "train-small-0000.parquet")
    bg.main(["report", "--data", str(tmp_path / "ds"), "--out", str(tmp_path / "r.md")])
    result = json.loads((tmp_path / "r.json").read_text())
    st = result["stats"]["dpc"]
    assert result["rows"] == result["unique_ids"] == 10 and st["g4_hard"] == 0.1 and st["g4"]["p50"] == 1.0
    assert st["tg"]["under_0_9"] == 1.0 and st["g4"]["within_human_p5_p95"] == 1.0
    assert "dpc" in (tmp_path / "r.md").read_text()


# ---------------------------------------------------------------------------------------------------- prompts

def test_the_new_prompts_are_natural_v1_with_one_sentence_each():
    base = ge.PROMPTS["natural-v1"]["plain"]
    fmt = ge.PROMPTS["natural-v1-fmt"]["plain"]
    fmtlen = ge.PROMPTS["natural-v1-fmt-len"]["plain"]
    added = "Do not add line breaks, paragraphs, lists, headings, emphasis, Markdown or quotation marks that the source does not have. "
    assert fmt.replace(added, "") == base
    assert fmtlen.replace("Make the {target} about as long as a professional human translation would be; do not add "
                          "words or explanations. ", "") == fmt
    assert set(bg.PROMPTS) <= set(ge.PROMPTS) and ge.MODELS[bg.MODEL]["prompt"] in bg.PROMPTS


def test_the_prompt_is_the_users_turn_of_gemma_4_around_the_stripped_text():
    codec = ge.TurnCodec(None, ge.MODELS[bg.MODEL], "natural-v1-fmt", "nl", "en")
    text = codec.text("  Dit is een zin.\n")
    assert text.startswith("<bos><|turn>user\nTranslate the following Dutch text into natural, idiomatic English")
    assert text.endswith("without explanation or commentary.\n\n\nDit is een zin.<turn|>\n<|turn>model\n"
                         "<|channel>thought\n<channel|>")
    assert bg.prompt_text("natural-v1-fmt").endswith("\n\n\n{text}")
    with pytest.raises(ValueError, match="no prompt"):
        ge.TurnCodec(None, ge.MODELS[bg.MODEL], "chat-v1")


def test_every_class_leaves_room_for_the_prompt_and_has_slots():
    model = ge.MODELS[bg.MODEL]
    classes = model["length_classes"]
    assert list(classes) == list(model["slots"])
    sources = [c["max_source"] for c in classes.values()]
    assert sources == sorted(sources) and sources[-1] == 512
    for c in classes.values():
        assert c["prefill_lengths"] == sorted(c["prefill_lengths"])
        assert c["prefill_lengths"][-1] - c["max_source"] >= 160


class FakeDevice:
    """A device whose every slot emits the tokens of its script, then repeats the last one"""

    def __init__(self, scripts):
        self.scripts, self.slot_of, self.pos = scripts, {}, {}

    def prefill(self, tokens):
        script = self.scripts[tuple(tokens)]
        return script, script[0]

    def insert(self, prefix, slot):
        self.slot_of[slot], self.pos[slot] = prefix, 1

    def generate(self):
        out = []
        for slot in range(max(self.slot_of) + 1):
            script = self.slot_of.get(slot, [0])
            k = self.pos.get(slot, 0)
            out.append(script[min(k, len(script) - 1)])
            self.pos[slot] = k + 1
        return out

    @staticmethod
    def fetch(arrays):
        return arrays


def test_decoding_stops_at_any_of_the_models_stop_tokens_or_the_limit():
    scripts = {(1,): [7, 8, 50], (2, 2): [9, 106], (3, 3, 3): [5] * 20}
    outputs = ge.decode_all([[1], [2, 2], [3, 3, 3]], 2, FakeDevice(scripts), max_decode=6, sync_every=2,
                            eos_ids=ge.MODELS[bg.MODEL]["eos_ids"])
    assert outputs == [([7, 8], True), ([9], True), ([5] * 6, False)]


# ---------------------------------------------------------------------------------------------------- datasets

def test_datasets_keeps_the_files_and_the_row_order_of_each_source(tmp_path):
    import pyarrow.parquet as pq

    sources, merged = tmp_path / "src", tmp_path / "merged"
    expected = {}
    for part, (key, folder) in bg.PARTS.items():
        rows = {}
        for split in bg.SPLITS:
            names = [f"{split}-00000-of-00002.parquet", f"{split}-00001-of-00002.parquet"] if split == "train" \
                else [f"{split}-00000-of-00001.parquet"]
            for k, name in enumerate(names):
                file_rows = [{"id": f"{part}:{split}:{k}:{i}", "nl": f"tekst {i}", "en": f"text {i}", "chrf": float(i)}
                             for i in (3, 1, 2)]  # not sorted by id
                bg.write_parquet(file_rows, sources / key / folder / name)
                expected[f"{key}/{folder}/{name}"] = [r["id"] for r in file_rows]
                for r in file_rows:
                    rows[r["id"]] = dict(r, en_gemma4=f"g4 {r['id']}", gemma4_hard_fails=[])
        ordered = sorted(rows.values(), key=lambda r: r["id"])  # the merged files hold other groups of rows
        bg.write_parquet(ordered[:4], merged / part / "train-small-0000.parquet")
        bg.write_parquet(ordered[4:], merged / part / "validation-small-0000.parquet")
    bg.main(["datasets", "--data", str(merged), "--sources", str(sources), "--out", str(tmp_path / "out")])
    for rel, ids in expected.items():
        table = pq.read_table(tmp_path / "out" / rel)
        assert table.column("id").to_pylist() == ids, rel
        assert table.column_names == ["id", "nl", "en", "chrf", "en_gemma4", "gemma4_hard_fails"]
        assert table.column("en_gemma4").to_pylist() == [f"g4 {i}" for i in ids]
    # a merged dataset that lacks rows of a source file
    bg.write_parquet([{"id": "dpc:train:0:3", "nl": "anders", "en": "text 3", "chrf": 3.0, "en_gemma4": "x",
                       "gemma4_hard_fails": []}], merged / "dpc" / "train-small-0000.parquet")
    with pytest.raises(SystemExit, match="not .or twice. in the merged dataset"):
        bg.main(["datasets", "--data", str(merged), "--sources", str(sources), "--out", str(tmp_path / "o2")])


def test_datasets_refuses_changed_source_columns_and_rows_of_no_source(tmp_path):
    sources, merged = tmp_path / "src", tmp_path / "merged"
    for part, (key, folder) in bg.PARTS.items():
        for split in bg.SPLITS:
            bg.write_parquet([{"id": f"{part}:{split}", "nl": "tekst", "en": "text"}],
                             sources / key / folder / f"{split}-00000-of-00001.parquet")
        bg.write_parquet([{"id": f"{part}:{split}", "nl": "tekst", "en": "text", "en_gemma4": "g"} for split in bg.SPLITS],
                         merged / part / "train-small-0000.parquet")
    changed = [{"id": f"fineweb:{s}", "nl": "tekst", "en": "other" if s == "train" else "text", "en_gemma4": "g"}
               for s in bg.SPLITS]
    bg.write_parquet(changed, merged / "fineweb" / "train-small-0000.parquet")
    with pytest.raises(SystemExit, match="differ from the source's"):
        bg.main(["datasets", "--data", str(merged), "--sources", str(sources), "--out", str(tmp_path / "o")])
    extra = [{"id": f"fineweb:{s}", "nl": "tekst", "en": "text", "en_gemma4": "g"} for s in (*bg.SPLITS, "x")]
    bg.write_parquet(extra, merged / "fineweb" / "train-small-0000.parquet")
    with pytest.raises(SystemExit, match="1 rows of the merged dataset are in no file of the source"):
        bg.main(["datasets", "--data", str(merged), "--sources", str(sources), "--out", str(tmp_path / "o3")])


def per_source_row(i):
    """A row of a dataset per source as datasets writes it without --standalone (the columns in that order)"""
    return {"id": f"dpc:{i}", "gptnl_doc_id": "d", "nl": f"tekst {i}", "en": f"tg {i}", "nl_back": "terug",
            "qe_en_nl": 1.0, "chrf": 60.0, "tokens_nl": 7, "tokens_en": 8, "length_class": "short", "len_ratio": 1.1,
            "sentences_nl": 1, "sentences_en": 1, "numbers_missing": [], "structure": [], "ended": True,
            "decode_loop": False, "dutch_left": False, "back_ended": True, "back_decode_loop": False,
            "back_structure": [], "hard_fails": ["len_ratio"], "source_score": 0.9, "topic": "t",
            "en_gemma4": f"g4 {i}", "gemma4_ended": False, "gemma4_tokens_nl": 6 + i, "gemma4_tokens_out": 9,
            "gemma4_prompt": "natural-v1-fmt", "gemma4_hard_fails": ["chatter"], "gemma4_structure": ["bold"],
            "gemma4_len_ratio": 1.2, "gemma4_numbers_missing": ["3"], "gemma4_word_ratio": 1.0,
            "tg_word_ratio": 0.9, "gemma4_char_ratio": 1.05, "gemma4_r2_length_out": True, "gemma4_time": "t"}


def test_standalone_leaves_out_translategemma_and_drops_the_prefix_of_gemma_4():
    import pyarrow as pa

    table = bg.standalone(pa.Table.from_pylist([per_source_row(0), per_source_row(1)]))
    assert table.column_names == [
        "id", "gptnl_doc_id", "nl", "en", "tokens_nl", "sentences_nl", "source_score", "topic", "ended", "tokens_out",
        "prompt", "hard_fails", "structure", "len_ratio", "numbers_missing", "word_ratio", "char_ratio", "time"]
    rows = table.to_pylist()
    assert [r["en"] for r in rows] == ["g4 0", "g4 1"] and [r["tokens_nl"] for r in rows] == [6, 7]
    assert rows[0]["hard_fails"] == ["chatter"] and rows[0]["structure"] == ["bold"] and rows[0]["ended"] is False
    assert rows[0]["numbers_missing"] == ["3"] and rows[0]["len_ratio"] == 1.2


def test_standalone_refuses_a_table_without_gemma_4_or_with_a_name_twice():
    import pyarrow as pa

    row = per_source_row(0)
    with pytest.raises(SystemExit, match=r"\['en_gemma4'\] missing"):
        bg.standalone(pa.Table.from_pylist([{k: v for k, v in row.items() if k != "en_gemma4"}]))
    with pytest.raises(SystemExit, match=r"twice .*\['topic'\]"):
        bg.standalone(pa.Table.from_pylist([dict(row, gemma4_topic="x")]))


def test_datasets_standalone_writes_the_published_schema(tmp_path):
    import pyarrow.parquet as pq

    sources, merged = tmp_path / "src", tmp_path / "merged"
    # the source dataset's columns: those of a row per source without the merged dataset's new ones
    tg_columns = [k for k in per_source_row(0) if not k.startswith("gemma4_") and k not in ("en_gemma4", "tg_word_ratio")]
    for part, (key, folder) in bg.PARTS.items():
        rows = []
        for split in bg.SPLITS:
            row = dict(per_source_row(0), id=f"{part}:{split}")
            bg.write_parquet([{k: row[k] for k in tg_columns}],
                             sources / key / folder / f"{split}-00000-of-00001.parquet")
            rows.append(row)
        bg.write_parquet(rows, merged / part / "train-small-0000.parquet")
    bg.main(["datasets", "--data", str(merged), "--sources", str(sources), "--out", str(tmp_path / "out"),
             "--standalone"])
    table = pq.read_table(tmp_path / "out" / "handelingen" / "context_turns" / "validation-00000-of-00001.parquet")
    assert table.column("id").to_pylist() == ["handelingen_context:validation"]
    assert table.column("en").to_pylist() == ["g4 0"] and "en_gemma4" not in table.column_names
    assert not set(table.column_names) & {"nl_back", "qe_en_nl", "chrf", "tg_word_ratio", "r2_length_out",
                                          "length_class", "tokens_en"}


@pytest.mark.skipif(not (GEMMA4_TOKENIZER / "tokenizer.json").exists(), reason="Gemma 4's tokenizer is not here")
def test_the_probes_prompts_fit_every_class_of_the_bt_model():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(GEMMA4_TOKENIZER))
    classes = ge.MODELS[bg.MODEL]["length_classes"]
    headroom = min(c["prefill_lengths"][-1] - c["max_source"] for c in classes.values())
    for name in bg.PROMPTS:
        codec = ge.TurnCodec(tok, ge.MODELS[bg.MODEL], name, "nl", "en")
        codec.check(["Dit is een zin."])
        assert len(codec.encode("x")) - 1 <= headroom


# ---------------------------------------------------------------------------------------------------- round trip

def fake_gemma4_datasets(root, n_per=3, tokens_nl=5):
    """The -gemma4 datasets as fetch --roundtrip writes them: the published columns that the round trip reads"""
    for part, (key, folder) in bg.PARTS.items():
        for split in bg.SPLITS:
            rows = [{"id": f"{part}:{split}:{i}", "nl": "Dit is tekst nummer een.", "en": " ".join(["word"] * (i + 1)),
                     "tokens_nl": tokens_nl, "hard_fails": ["structure"] if i == 0 else []} for i in range(n_per)]
            write_source(root, key, folder, split, rows)


def test_plan_of_the_round_trip_takes_the_english_and_checks_the_decode_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(bg, "Counter", WordCounter)
    fake_gemma4_datasets(tmp_path / "hf")
    n = len(bg.PARTS) * 2 * 3
    bg.main(["plan", "--data", str(tmp_path / "hf"), "--text", "en", "--tokenizer", "-", "--out",
             str(tmp_path / "plan"), "--expect", str(n)])
    info = bg.load_plan(tmp_path / "plan")
    assert info["model"] == bg.RT_MODEL and bg.languages(info) == ("en", "nl")
    assert info["revisions"] == {k: v[1] for k, v in bg.ROUNDTRIP_SOURCES.items()}
    texts = [r["text"] for u in info["units"] for r in bg.read_rows(tmp_path / "plan" / u["segments"])[u["start"]:u["end"]]]
    assert sorted(texts) == sorted(["word", "word word", "word word word"] * (n // 3))
    # 96 decode tokens in tiny hold 76 tokens of Dutch with the margin of 1.25, not 77
    fake_gemma4_datasets(tmp_path / "hf2", tokens_nl=77)
    with pytest.raises(SystemExit, match="exceeds the decode limit"):
        bg.main(["plan", "--data", str(tmp_path / "hf2"), "--text", "en", "--tokenizer", "-", "--out",
                 str(tmp_path / "p2"), "--expect", "0"])
    # nl -> en as before: the plan names no languages and has the model of the back-translation
    fake_hf(tmp_path / "hf3")
    bg.main(["plan", "--data", str(tmp_path / "hf3"), "--tokenizer", "-", "--out", str(tmp_path / "p3"), "--expect", "0"])
    old = bg.load_plan(tmp_path / "p3")
    assert old["model"] == bg.MODEL and "source" not in old and bg.languages(old) == ("nl", "en")


def test_the_round_trip_asks_for_dutch_from_the_english():
    plan_info = {"model": bg.RT_MODEL, "classes": ge.MODELS[bg.RT_MODEL]["length_classes"], "source": "en",
                 "target": "nl"}
    args = type("Args", (), {"length_class": "small"})()
    codec = bg.Translator(args, plan_info).codec("natural-v1-fmt")
    assert codec.target == "nl"
    assert codec.text("Hello.").startswith("<bos><|turn>user\nTranslate the following English text into natural, "
                                           "idiomatic Dutch, as a native speaker would write it")
    assert bg.prompt_text("natural-v1-fmt", "en", "nl").startswith("Translate the following English text into")


def test_the_round_trip_publishes_a_shard_as_it_is(tmp_path):
    shard = tmp_path / "s.parquet"
    bg.write_parquet([{"id": "a", "text_nl": "Hallo.", "ended": True}, {"id": "b", "text_nl": "Ja.", "ended": True}],
                     shard)
    path, rows = bg.hf_file({"source": "en", "target": "nl"}, {"name": "dpc/train-tiny-0000"}, shard, None, None,
                            tmp_path / "out.parquet")
    assert (path, rows) == (str(shard), 2) and not (tmp_path / "out.parquet").exists()


def rt_shards(root, data, skip=(), extra=()):
    """The round trip's shards for every row of fake_gemma4_datasets: the Dutch is the original but for i == 2"""
    for part, (key, folder) in bg.PARTS.items():
        for split in bg.SPLITS:
            rows = bg.read_rows(next((Path(data) / key / folder).glob(f"{split}-*.parquet")))
            out = [{"id": r["id"], "text_nl": r["nl"] if not r["id"].endswith(":2") else "Iets heel anders.",
                    "ended": not r["id"].endswith(":1"), "tokens_out": 7, "tokens_in": 9, "prompt": "natural-v1-fmt"}
                   for r in rows if r["id"] not in skip]
            out += [dict(out[0], id=i) for i in extra if i.startswith(f"{part}:{split}:")]
            bg.write_parquet(out, Path(root) / part / f"{split}-tiny-0000.parquet")


def test_roundtrip_appends_the_columns_to_every_file_and_keeps_the_rest(tmp_path):
    import pyarrow.parquet as pq
    import sacrebleu

    fake_gemma4_datasets(tmp_path / "hf")
    rt_shards(tmp_path / "shards", tmp_path / "hf")
    bg.main(["roundtrip", "--data", str(tmp_path / "hf"), "--shards", str(tmp_path / "shards"), "--out",
             str(tmp_path / "out")])
    source = pq.read_table(tmp_path / "hf" / "handelingen" / "context_turns" / "train-00000-of-00001.parquet")
    table = pq.read_table(tmp_path / "out" / "handelingen" / "context_turns" / "train-00000-of-00001.parquet")
    assert table.column_names == source.column_names + [n for n, _ in bg.back_schema()]
    assert table.select(source.column_names).equals(source)
    rows = table.to_pylist()
    assert rows[0]["chrf"] == 100.0 and rows[0]["nl_back"] == rows[0]["nl"] and rows[0]["back_tokens_out"] == 7
    assert rows[1]["back_ended"] is False and rows[0]["back_ended"] is True
    assert rows[2]["chrf"] == round(sacrebleu.sentence_chrf("Iets heel anders.", ["Dit is tekst nummer een."]).score, 3)
    assert rows[2]["back_structure"] == [] and rows[2]["back_chatter"] is False
    result = json.loads((tmp_path / "out" / "roundtrip.json").read_text())
    assert result["rows"] == len(bg.PARTS) * 2 * 3 and result["prompt"] == "natural-v1-fmt"
    s = result["stats"]["handelingen/context_turns/train"]
    assert s["n"] == 3 and s["back_not_ended"] == 1 and s["chrf_without_hard_fails"]["n"] == 2
    assert s["chrf"]["p95"] == 100.0 and s["chrf_below"]["50"] == 1


def test_roundtrip_refuses_a_row_without_a_translation_and_a_translation_of_no_row(tmp_path):
    fake_gemma4_datasets(tmp_path / "hf")
    rt_shards(tmp_path / "s1", tmp_path / "hf", skip={"raad:validation:1"})
    with pytest.raises(SystemExit, match="1 rows without a translation"):
        bg.main(["roundtrip", "--data", str(tmp_path / "hf"), "--shards", str(tmp_path / "s1"), "--out",
                 str(tmp_path / "o1")])
    rt_shards(tmp_path / "s2", tmp_path / "hf", extra={"dpc:train:9"})
    with pytest.raises(SystemExit, match="1 translations of the shards belong to no row"):
        bg.main(["roundtrip", "--data", str(tmp_path / "hf"), "--shards", str(tmp_path / "s2"), "--out",
                 str(tmp_path / "o2")])
    rt_shards(tmp_path / "s3", tmp_path / "hf", extra={"dpc:train:0"})
    with pytest.raises(SystemExit, match="twice in the shards"):
        bg.main(["roundtrip", "--data", str(tmp_path / "hf"), "--shards", str(tmp_path / "s3"), "--out",
                 str(tmp_path / "o3")])


def test_back_columns_find_english_left_chatter_and_a_loop():
    en = "The council will decide on the new plan next week, after the consultation with all of the residents."
    left = bg.back_columns("De raad beslist.", en, "De raad will decide on the new plan next week, after the "
                           "consultation with all of the residents.", True)
    assert left["back_english_left"] is True
    chat = bg.back_columns("De raad beslist.", "The council decides.", "Hier is de vertaling: De raad beslist.", True)
    assert chat["back_chatter"] is True and chat["back_english_left"] is False
    loop = bg.back_columns("Wij gaan.", "We go.", "Wij gaan, gaan, gaan, gaan, gaan, gaan, gaan, gaan.", False)
    assert loop["back_decode_loop"] is True and loop["back_ended"] is False


def test_every_class_of_the_round_trip_leaves_room_for_the_prompt_and_the_dutch():
    model = ge.MODELS[bg.RT_MODEL]
    classes = model["length_classes"]
    assert list(classes) == list(model["slots"]) and model["prompt"] == "natural-v1-fmt"
    assert {k: model[k] for k in ("maxtext_model", "revision", "chips", "eos_ids", "turn")} == \
        {k: ge.MODELS[bg.MODEL][k] for k in ("maxtext_model", "revision", "chips", "eos_ids", "turn")}
    # the most Gemma 4 tokens of an original Dutch passage per class of its English (the -gemma4 datasets)
    most_dutch = {"tiny": 72, "small": 227, "medium": 425, "long": 511, "xlong": 511}
    for name, c in classes.items():
        assert c["prefill_lengths"] == sorted(c["prefill_lengths"])
        assert c["prefill_lengths"][-1] - c["max_source"] >= 160
        assert c["max_decode"] >= most_dutch[name] * bg.DECODE_MARGIN
        assert model["slots"][name] % model["chips"] == 0
        assert model["slots"][name] * (c["prefill_lengths"][-1] + c["max_decode"]) <= 46_080


@pytest.mark.skipif(not (GEMMA4_TOKENIZER / "tokenizer.json").exists(), reason="Gemma 4's tokenizer is not here")
def test_the_round_trips_prompt_fits_every_class():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(GEMMA4_TOKENIZER))
    classes = ge.MODELS[bg.RT_MODEL]["length_classes"]
    headroom = min(c["prefill_lengths"][-1] - c["max_source"] for c in classes.values())
    codec = ge.TurnCodec(tok, ge.MODELS[bg.RT_MODEL], ge.MODELS[bg.RT_MODEL]["prompt"], "en", "nl")
    codec.check(["This is a sentence."])
    assert len(codec.encode("x")) - 1 <= headroom
