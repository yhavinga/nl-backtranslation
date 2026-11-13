# nl-backtranslation

This repository holds the pipeline that made back-translation pairs for an English-to-Dutch translator: it chooses
passages of original Dutch text, has Gemma 4 31B judge them, translates them into English and the English back into
Dutch with TranslateGemma 27B on TPU, and scores every pair (hard checks, chrF, MetricX-24 in quality-estimation mode).
A training pair is the synthetic English as the source and the original Dutch as the target.

TranslateGemma's English is wordier than a human translation (a median of 0.960 Dutch words per English word on 200
random passages, against about 1.06 for human translations), and a translator trained on it learns to compress.
`gemma4_bt/` therefore translates the same 271,710 Dutch passages into English once more, with Gemma 4 31B-it and a
prompt chosen by a rule fixed in advance (see [Gemma 4 back-translation](#gemma-4-back-translation)). It adds a
column `en_gemma4` and its checks to every row of the five datasets; the TranslateGemma pipeline and its columns stay
as they are.

The comments in the code refer to the design of the experiment the pairs were made for ("design §3"), which is not
published, and to the dates of its decisions. "Worker" and "host" are the 4 hosts of a TPU v4-32.

## Published datasets

The pairs are published on Hugging Face as one dataset per source of original Dutch text. Each dataset card describes
the selection from that source, the scores, the license and the known limitations of the pairs. Together they form the collection
[Dutch back-translation (nl→en)](https://huggingface.co/collections/yhavinga/dutch-back-translation-nlen-6af45e2ba548b444c6362255).

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
| judge of the Dutch passages; translation model of `gemma4_bt/` | `google/gemma-4-31B-it` | `842da3794eaa0b77d5f08bae87a17459d91ff475` |
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
`langid` 1.1.6 (the versions `setup.sh` installs). `gemma4_bt/` runs Gemma 4 in MaxText from git: commit `2249474` of
[github.com/yhavinga/maxtext](https://github.com/yhavinga/maxtext) (MaxText's main of 9 Nov 2026, `d1c0d0a`, with the
NNX decode cache made from shapes inside the init jit), installed by `gemma4_bt/setup.sh`.

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
| `gemma4_bt/bt_gemma4.py` | the Gemma 4 back-translation: the source datasets (`fetch`), the probe and the rule that chooses the prompt (`probeplan`, `select`), the units of the full run (`plan`), claiming and translating them (`work`, `status`), the merged dataset on Hugging Face (`publish`, `final`, `report`, `card`) and one dataset per source (`datasets`) |
| `gemma4_bt/gemma4_engine.py`, `gemma4_bt/models.yml` | Gemma 4 31B-it in MaxText on 4 chips (greedy, continuous batching); the prompts, the length classes and the decode slots |
| `gemma4_bt/run_bt_gemma4.sh`, `gemma4_bt/setup.sh` | all steps on one v6e-8 VM, as two processes of 4 chips; its environment in `/dev/shm/bt_gemma4` |
| `gemma4_bt/keeper.sh`, `gemma4_bt/stage.sh` | on a control machine: the spot VMs of every region requested, started, restarted and deleted, and the units of the other regions announced; the code put into each region's bucket |
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
- **Gemma 4 back-translation**: spot v6e-8 TPU VMs in one or more regions, a `gs://` directory in a bucket of each
  region, Gemma 4 31B-it as a MaxText checkpoint in each of those regions (`gemma_judge/setup.sh convert`), a private
  dataset on Hugging Face for the merged output, and a control machine with `gcloud` and `gsutil` for `keeper.sh`.
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

## Gemma 4 back-translation

`gemma4_bt/` translates the 271,710 Dutch passages and context turns of the five datasets (`bt_gemma4.py`, `SOURCES`:
the revisions of 11 Nov 2026) from Dutch into English again, with `google/gemma-4-31B-it` at `842da37`, greedy, in
MaxText on spot v6e-8 VMs. Gemma 4 31B allows at most 4-way tensor parallelism, so each VM runs two processes of 4
chips. Nothing is filtered out: every row keeps all columns of its source dataset (TranslateGemma's English stays in
`en`) and gets `en_gemma4`, the checks of the pair (`nl`, `en_gemma4`) as `gemma4_*` columns (the hard checks of
`pairs.py`, and `chatter`: an introduction or a note around the translation), and `tg_word_ratio`.

### The prompt

The prompts are in `gemma4_bt/models.yml`, each as one user turn in Gemma 4's chat template with an empty thought
channel. `natural-v1` asks for natural, complete and faithful English; in an earlier pilot it added formatting the
source does not have. `natural-v1-fmt` adds one sentence against that, and `natural-v1-fmt-len` one more on length.
Before the probe translated anything, the rule that picks one was fixed (`bt_gemma4.choose`): on the 200 passages of
the stratum `random` (40 per source), a prompt is eligible when at most 1.5% of its pairs fail a hard check; of those,
the one whose median Dutch words per English word is closest to 1.06 (human translations); where the distances differ
by less than 0.01, the one with fewer failures. The probe of 13 Nov 2026:

| prompt | pairs | hard-check failures | median Dutch words per English word |
|---|---:|---:|---:|
| `natural-v1` | 200 | 7 (3.5%) | 1.018 |
| `natural-v1-fmt` | 200 | 0 (0.0%) | 1.000 |
| `natural-v1-fmt-len` | 200 | 2 (1.0%) | 1.000 |

`natural-v1` was not eligible; the other two were equally close to 1.06, and `natural-v1-fmt` had fewer failures, so
it was chosen. TranslateGemma's English on the same 200 passages had a median of 0.960. No prompt reached 1.06: the
sentence on length did not change the median.

### Steps

`G1`, `G2`, ... are the `gs://` directories of the run, one in a bucket of each region; `C1`, `C2`, ... Gemma 4's
MaxText checkpoints in those regions (`<checkpoint>/0/items`); `REPO` the private Hugging Face dataset of the merged
output.

1. **Stage the code** (from a clean checkout): `bash gemma4_bt/stage.sh $G1 $G2`. With `PILOT=<file>` also the ids of
   an earlier pilot (`bt_gemma4.py probeplan --pilot`: its sample is left out of the probe's strata, and its
   translations that failed `structure` form the stratum `structure`).
2. **Mark the region of the probe**: `echo | gsutil cp - $G1/probe/run-here`.
3. **Start the keeper** on the control machine. `TPU_UP` and `TPU_DOWN` are your commands to request and delete a spot
   v6e-8 (the keeper calls them with `--accelerator v6e-8 --zone <zone> --name <name> [--replace]` and
   `--name <name>`); put a quota guard there.

   ```
   PROJECT=<project> VMS="bt-a-0:us-east1-d bt-b-0:europe-west4-a" REPO=$REPO \
       REGIONS="us-east1=$G1,forward,$C1 europe-west4=$G2,backward,$C2" TPU_UP="<command>" TPU_DOWN="<command>" \
       bash gemma4_bt/keeper.sh
   ```

   Each VM runs `run_bt_gemma4.sh all`: `setup.sh`, the source datasets from Hugging Face (`fetch`), the plan (558
   units of at most 512 consecutive ids of one source, split and length class; every VM checks its plan's hash
   against `full/plan-hash.txt`), and in the probe's region the probe and `selection.md`.
4. **Set the prompt** the rule chose, in every region: `echo natural-v1-fmt | gsutil cp - $G1/full/prompt.txt` (and
   `$G2`). The VMs then take units, one region forwards and the other backwards, and each region's publisher puts its
   shards into `REPO` as `<part>/<split>-<class>-<k>.parquet`. Once all 558 are there, `final` writes the report to
   `full/report/`, and the keeper deletes the VMs and ends.
5. **One dataset per source**, on any machine with a copy of `REPO` and of the sources: `python gemma4_bt/bt_gemma4.py
   fetch --out sources`, then `python gemma4_bt/bt_gemma4.py datasets --data <copy of REPO> --sources sources --out
   per-source` writes per source the files of the source dataset (`data/`, and `context_turns/` for the Handelingen),
   each with the same rows in the same order and the new columns; it checks that every source column is unchanged.

The length of a passage is a setting: a set of longer passages needs length classes of its own in `models.yml` (a new
model entry) and `bt_gemma4.py plan --model <entry> --max-source <tokens>`.

## Tests

The tests use made-up data and need `pytest`, `numpy`, `pyarrow`, `pyyaml`, `sacrebleu`, `tokenizers` and
`langid==1.1.6`; three of them skip without `jax`, without TranslateGemma's tokenizer (`TRANSLATEGEMMA_TOKENIZER`) or
without Gemma 4's (`GEMMA4_TOKENIZER`; with `transformers`). The code is checked with
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
