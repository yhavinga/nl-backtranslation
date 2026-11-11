# nl-backtranslation

This repository holds the pipeline that made back-translation pairs for an English-to-Dutch translator: it chooses
passages of original Dutch text, has Gemma 4 31B judge them, translates them into English and the English back into
Dutch with TranslateGemma 27B on TPU, and scores every pair (hard checks, chrF, MetricX-24 in quality-estimation mode).
A training pair is the synthetic English as the source and the original Dutch as the target.

The comments in the code refer to the design of the experiment the pairs were made for ("design §3"), which is not
published, and to the dates of its decisions. "Worker" and "host" are the 4 hosts of a TPU v4-32.

## Published datasets

The pairs are published on Hugging Face as one dataset per source of original Dutch text. Each dataset card describes
the selection from that source, the scores, the license and the known limitations of the pairs.

| dataset | Dutch source | license |
|---|---|---|
| [yhavinga/fineweb-2-nl-backtranslated-en](https://huggingface.co/datasets/yhavinga/fineweb-2-nl-backtranslated-en) | web text of `.nl` and `.be` domains (FineWeb-2 `nld_Latn`) | ODC-By 1.0 |
| [yhavinga/handelingen-tweede-kamer-nl-backtranslated-en](https://huggingface.co/datasets/yhavinga/handelingen-tweede-kamer-nl-backtranslated-en) | debates of the Tweede Kamer (`yhavinga/dutch-handelingen-modern`), with the speaking turns before a passage as context (config `context_turns`) | CC0 1.0 |
| [yhavinga/openraadsinformatie-nl-backtranslated-en](https://huggingface.co/datasets/yhavinga/openraadsinformatie-nl-backtranslated-en) | documents of municipal, provincial and water-board councils (GPT-NL collection `openraadsinformatie`) | CC BY 4.0 |
| [yhavinga/rijksoverheid-websites-nl-backtranslated-en](https://huggingface.co/datasets/yhavinga/rijksoverheid-websites-nl-backtranslated-en) | websites of the Dutch central government (GPT-NL collection `dienst_publiek_en_communicatie`) | CC BY 4.0 |
| [yhavinga/rechtspraak-nl-backtranslated-en](https://huggingface.co/datasets/yhavinga/rechtspraak-nl-backtranslated-en) | court rulings of uitspraken.rechtspraak.nl (GPT-NL collection `de_rechtspraak`) | CC BY 4.0 |

The English translations in all five were made with TranslateGemma and fall under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) as well.

## Sources and models

| what | Hugging Face id or URL | revision |
|---|---|---|
| Dutch web text | `HuggingFaceFW/fineweb-2` (`nld_Latn`) | `af9c13333eb981300149d5ca60a8e9d659b276b9` |
| Tweede Kamer debates | `yhavinga/dutch-handelingen-modern` | `abdba14ac9271b6d43ccefe6143202b7bdd4e900` (see below) |
| GPT-NL council reports, government information, court rulings | `tvosch/GPT-NL-propella-annotations` | `db7f5947be4445e396aa0dd0df52ab6c1f418efc` |
| translation model | `google/translategemma-27b-it` | `7d10f0b72f89a2d0f268cea30727d8b77c0d25c2` |
| judge of the Dutch passages | `google/gemma-4-31B-it` | `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| quality estimation | `google/metricx-24-hybrid-xxl-v2p6-bfloat16` | `3d9c6f7fd11b58d3eceeb68ee39fe7d2c761ee20` |
| MetricX's tokenizer | `google/mt5-xl` | `63fc6450d80515b48e026b69ef2fbbd426433e84` |
| word list | `OpenTaal/opentaal-wordlist` (GitHub) | `b250510dda431785f962019167d1415198ff3905` |
| idiom list | kaikki.org's extract of the Dutch entries of the English Wiktionary | not versioned; `setup.sh` records its Last-Modified and sha256 |

`select_passages.py` pins `yhavinga/dutch-handelingen-modern` at `abdba14`, the revision the published selection read.
The Hub no longer has that revision: the dataset's history was squashed into one commit, `ef8e7e5`. The published
Handelingen passages were checked against `ef8e7e5` (same agenda items, dates, speakers and turn texts).

The test sets that no passage may share a 13-gram with (`mainset.py testsets`, `TESTSETS`): DBRD, SICK-NL, FLORES-200
devtest, WMT24++ en-nl_NL and NTREX-128, each at a fixed revision. They are read only as a block list.

Software: MaxText 0.2.5 (both models run in MaxText on TPU), JAX 0.11.1, `tokenizers` 0.23.2, `sacrebleu` 2.5.1,
`langid` 1.1.6 (the versions `setup.sh` installs).

## Files

| file | purpose |
|---|---|
| `select_passages.py` | step 0: passages of 48-768 tokens from the sources, the deterministic filters, the idiom score, the split per document (`idioms`, `select`, `annotate`, `repair`) |
| `mainset.py` | the test sets as a block list, the judge's input shards, FineWeb-2's hosts, the minipilot of long passages, the final selection (`final`), and steps of the experiment's fine-tune data (`eval-rows`, `replay`) |
| `structure.py` | the formatting of a text and of its translation compared |
| `publish.py` | `decode_loop` and `english_left` of FLAN-nl |
| `pairs.py` | the English as segments for the back-translation, the measures and hard checks of every pair, the QE input, the summary |
| `metricx_qe.py` | MetricX-24 in JAX on the 4 chips of a TPU host; `check` compares it with `transformers` on the CPU |
| `translate.py`, `run_translate.sh` | TranslateGemma on the 4 chips of a TPU host, greedy, in resumable shards, in either direction |
| `gemma_judge/judge.py`, `gemma_judge/run_source_judge.sh` | Gemma 4 31B-it as a yes/no judge: the source check, the origin question and the topic of a Dutch passage, or a translation pair |
| `gemma_judge/data.py`, `gemma_judge/report.py`, `gemma_judge/hf_reference.py` | the judge's calibration: Europarl paragraphs with deterministic errors, summaries of its scores, and its numbers from `transformers` |
| `setup.sh`, `gemma_judge/setup.sh` | the environments on the TPU hosts, the word list and the idiom source, and Gemma 4 31B-it as a MaxText checkpoint |
| `tests/` | tests on made-up data; no network, model or TPU |

## Requirements

- **CPU steps**: Python 3.10 or later with `pyarrow`, `numpy`, `tokenizers`, `huggingface_hub`, `sacrebleu` and
  `langid==1.1.6`. `setup.sh` builds such a venv in `/dev/shm/translator/venv` (with JAX for TPU, for MetricX) and
  downloads the word list and the idiom source to `/dev/shm/translator/resources/`.
- **TranslateGemma** runs in the environment of [FLAN-nl](https://github.com/yhavinga/flan-nl): its `setup.sh` builds
  `/dev/shm/flan_nl/venv` (MaxText 0.2.5) and the tokenizer in `/dev/shm/flan_nl/tokenizer-27b`, and
  `ROOT=gs://<bucket>/<dir> bash setup.sh convert` writes the MaxText checkpoint to `$ROOT/translategemma-27b-maxtext`.
- **The judge**: `CKPT=gs://<bucket>/<dir>/gemma-4-31b-it-maxtext bash gemma_judge/setup.sh` builds
  `/dev/shm/judge/venv`; with `convert` it also converts Gemma 4 31B-it to that checkpoint.
- **TPU**: Cloud TPU VMs with v4 hosts of 4 chips (each host runs as a TPU of its own), `gsutil`, and a `gs://`
  directory for checkpoints, inputs and outputs. The run scripts read the worker number from the hostname (`...-w-<N>`).
- **Hugging Face token**: TranslateGemma is a gated model. Put a token in `~/.cache/huggingface/token` on each machine
  (for example with `hf auth login`); the scripts never take it as an argument and never print it.

## Steps

Run the commands from this directory on the TPU hosts. `T=/dev/shm/flan_nl/tokenizer-27b` is TranslateGemma's
tokenizer, `RES=/dev/shm/translator/resources`, and `G=gs://<bucket>/<dir>` is a directory of your own. The run
scripts (steps 3, 6 and 7) and step 9 run on every host, each on its own share (`w{w}` and `<k>` are the host's
number); the other steps run on one host.

1. **Idioms and test sets.**

   ```
   python select_passages.py idioms --kaikki $RES/kaikki-dutch.jsonl --out $RES/idioms.json
   python mainset.py testsets --out testsets
   ```

2. **Pools of candidate passages** (`select_passages.py select`). The FineWeb-2 pool:

   ```
   python select_passages.py select --out pool-fineweb --idioms $RES/idioms.json --wordlist $RES/opentaal-wordlist.txt \
       --tokenizer $T --row-groups handelingen=0,raad=0,dpc=0,rechtspraak=0,fineweb=240 --max-docs fineweb=400000 \
       --counts fineweb=93000 --eval-counts fineweb=1860 --n 93000 --n-eval 1860 --eval-permille 20 --seed 0 \
       --block-texts testsets/*.txt --processes 200
   ```

   The other sources went into a pool of their own with `--row-groups handelingen=100,raad=100,dpc=32,rechtspraak=24
   --max-per-doc handelingen=15,raad=5 --max-docs -`. `select_passages.py repair --pool <pool> ...` takes up the
   passages whose only fault was a marked heading glued to its paragraph.

3. **Gemma 4 31B judges every passage of the pools** (source check, origin and topic in one forward pass):

   ```
   python mainset.py judge-shards --select pool-fineweb --out judge-in --workers 4
   gsutil -m rsync -r judge-in $G/judge-in
   CKPT=$G/gemma-4-31b-it-maxtext LOGS=$G/logs TASK=source_topic_origin FSDP=1 TENSOR=4 TOKEN_BUDGET=16384 \
       SHARDS=$G/judge-in/w{w} OUT=$G/source-scores/w{w} NAME=source_1 bash gemma_judge/run_source_judge.sh
   ```

4. **FineWeb-2's hosts** (for the cap per host): `python mainset.py fineweb-hosts --select pool-fineweb --out hosts.parquet`.

5. **The final selection** (source score and origin score >= 0.5, the shares per source, topic and host caps, half of
   the passages idiom-rich, the two edits before translation; `segments-w<k>.parquet` per host):

   ```
   python mainset.py final --select pool pool-fineweb repaired --scores scores-pool scores-repaired \
       --hosts hosts.parquet --tokenizer $T --block-texts testsets/*.txt --n 200000 --n-eval 4000 --max-tokens 384 \
       --out select
   ```

   The second run of 385-511 tokens: the same with `--min-tokens 384 --max-tokens 511 --exclude select --no-context
   --seed en-nl-translator/mainset-long/0 --n 66000 --n-eval 1320`. Passages above 511 tokens are not translated: in
   the minipilot (`mainset.py minipilot`, `minipilot-segments`, `minipilot-check`) TranslateGemma looped on 166 of 253
   passages of 512-767 tokens.

6. **Dutch to English** (copy `select/` to `$G/select/` first):

   ```
   CKPT=$G/translategemma-27b-maxtext LOGS=$G/logs SOURCE=nl TARGET=en SEGMENTS=$G/select/segments-w{w}.parquet \
       OUT=$G/nl-en/w{w} NAME=nl_en bash run_translate.sh
   ```

7. **The English as segments, and back to Dutch** (copy `$G/nl-en` to `nl-en/` and `back/` to `$G/back/`):

   ```
   python pairs.py back-segments --select select --nl-en nl-en --tokenizer $T --out back
   CKPT=$G/translategemma-27b-maxtext LOGS=$G/logs SOURCE=en TARGET=nl SEGMENTS=$G/back/segments-w{w}.parquet \
       OUT=$G/en-nl/w{w} NAME=en_nl bash run_translate.sh
   ```

8. **Measures and hard checks** (copy `$G/en-nl` to `en-nl/`):

   ```
   python pairs.py measures --select select --nl-en nl-en --en-nl en-nl --out pairs/pairs-measured.parquet --processes 64
   python pairs.py qe-split --pairs pairs/pairs-measured.parquet --workers 4 --out qe-in
   ```

9. **MetricX-24-QE** on each host, with the host's 4 chips as a TPU of their own (the `TPU_*` variables of
   `run_translate.sh`):

   ```
   python metricx_qe.py score --pairs qe-in/qe-in-w<k>.parquet --pair en:nl --model xxl --root /dev/shm/translator/metricx \
       --out qe-w<k>.parquet
   ```

10. **Summary**: `python pairs.py summary --pairs pairs/pairs-measured.parquet --qe qe-w*.parquet --out pairs-final
    --title "<title>"` writes `pairs-final/pairs.parquet` (every pair with its measures and `qe_en_nl`), `summary.json`
    and `summary.md`.

Nothing is filtered after translation: a training set is a query on `pairs.parquet` (for example `hard_fails` empty and
`qe_en_nl` <= 7.5).

## Tests

The tests use made-up data and need `pytest`, `numpy`, `pyarrow`, `sacrebleu`, `tokenizers` and `langid==1.1.6`; two of
them skip without `jax` or without TranslateGemma's tokenizer (`TRANSLATEGEMMA_TOKENIZER`). The code is checked with
ruff's pycodestyle, pyflakes and isort rules:

```
python -m pytest -q
ruff check --line-length 100 --target-version py310 --select E,F,I,W --ignore E501 .
```

`translate.py`'s LoRA merge (`--lora`) has no test here: it needs an adapter of a fine-tune script that is not in
this repository (`sft_lora.py`) and MaxText.

## Acknowledgements

Research supported with Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).

## Licences

- **Code**: Apache License 2.0 (`LICENSE`).
- **FineWeb-2**: ODC-By 1.0; its use is also subject to Common Crawl's Terms of Use.
- **GPT-NL Public Corpus** and **GPT-NL propella annotations**: CC BY 4.0. **dutch-handelingen-modern**: CC0 1.0.
- **TranslateGemma**: provided under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Section 3.3 says:
  "Google claims no rights in Outputs you generate using Gemma." If you train a model on the translations, read the
  definition of "Model Derivatives" in Section 1.1(e) of those terms.
- **Gemma 4 31B-it** and **MetricX-24**: Apache License 2.0.
