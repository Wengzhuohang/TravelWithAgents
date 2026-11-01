# TravelWithAgents

A multi-agent travel planner (IS469 GenAI class project). You describe a trip in plain English, and a team of
role-specialised LLM agents plus plain-code components turns it into a day-by-day itinerary you could actually
follow. Every plan is compared against a **single-prompt baseline** that does the same job in one LLM call.

Everything lives in one notebook: [`TravelWithAgents.ipynb`](TravelWithAgents.ipynb).

## How it works

| Step | Component | Type | Output |
|:--|:--|:--|:--|
| A | Ingestion | code + Google Places API, run once per city | `data/places/places_<city>.csv` |
| 0 | Parser agent | LLM | structured trip request (dates, budget, pace, must-include/avoid) |
| 1 | Scout agent | LLM + tools | a pool of ~25–35 candidate places |
| 2 | Persona agents: Adventurer, Foodie, Culture buff, Budget-keeper | 4 × LLM | each persona's ratings and draft timetable |
| 3 | Selector | code | final places, split between personas by **weights that sum to 1** |
| 4 | Route optimiser | code + Google Routes API | places grouped into days by area, shortest order per day, timed schedule |
| 5 | Validator + Feasibility agent | code + LLM with tools, up to 3 repair rounds | a plan with no hard violations |
| 6 | Baseline | one LLM prompt | the same task in a single call, for comparison |
| 7 | Evaluator | code metrics + pairwise LLM judge + evaluator agent | scores for both plans |

An LLM never runs code itself. When it wants a tool, it returns a structured request, and the harness
(the "LLM harness" cell) runs the Python function, sends the result back, enforces a step budget and logs every
call.

Notebook cells are tagged **[A]** (data & routing), **[B]** (agents & evaluation) or **[A+B]** (shared
contracts: change them together).

## Models

All models are called over the API, so a plain CPU runtime is enough.

| Role | Model | Provider |
|:--|:--|:--|
| `strong`: feasibility agent and baseline | `openai/gpt-oss-120b` | Groq |
| `small`: parser, scout, personas | `openai/gpt-oss-20b` | Groq |
| `judge`: pairwise judge and evaluator agent | `qwen/qwen3.8-27b` | Groq |

The baseline and the feasibility agent share the same model, so the multi-agent system never gets a bigger model
than the baseline. The judge comes from a different model family than the planners, to avoid self-preference
bias. Models are set in the `MODELS` dict in the configuration cell.

## Setup

### 1. API keys

| Key | Needed for | Required? |
|:--|:--|:--|
| `GROQ_API_KEY` | every LLM call | **yes** |
| `GOOGLE_MAPS_API_KEY` | ingesting new cities (Places API) and real travel times (Routes API) | no |
| `GEMINI_API_KEY` | not used by default (the judge runs on Groq) | no |

The notebook looks for each key in this order: environment variable → Colab Secrets → `.env` file in the repo
root. If `GROQ_API_KEY` is missing everywhere, it asks you to type it in.

- **Local:** fill in [`.env`](.env) (`KEY=value`, one per line). It is gitignored, so it is never committed.
- **Colab:** add the keys under Secrets (🔑 icon in the left sidebar) and turn on notebook access.

Never paste a key into a notebook cell: outputs get committed.

Without a Maps key the notebook still runs. It uses the cached city files in `data/places/`, and it estimates
travel times from straight-line distance for any pair not already in `data/cache/routes.json` (marked
`estimate` in the outputs).

### 2. Packages

The first code cell installs them:

```
%pip install -q "openai>=1.58,<2" pandas requests scikit-learn
```

Restart the kernel after it installs or upgrades anything. The `openai` pin matters:

- **1.58 or newer**, because the harness sends `reasoning_effort` for the `gpt-oss` models.
- **Below 2**, because 3.x switches to `httpx2`, which crashes with older Brotli installs (common in Anaconda)
  with `TypeError: process() takes no keyword arguments`.

### 3. Run

Open the notebook and run the cells from the top. Steps 0–8 walk through one sample request (Hanoi) so you can
see each agent's output.

## Running the experiments

The **Experiments** section runs the full pipeline over a test set of 12 requests across 5 cities:

| City | Test ids |
|:--|:--|
| Hanoi | `han-01`, `han-02`, `han-03` |
| Ho Chi Minh City | `hcm-01`, `hcm-02` |
| Da Nang | `dn-01`, `dn-02` |
| Hoi An | `ha-01`, `ha-02` |
| Bangkok | `bkk-01`, `bkk-02`, `bkk-03` |

In the batch cell, set:

```python
ONLY_IDS = ["hcm-01", "hcm-02"]   # None runs all 12
RUN_BATCH = True                   # off by default: each case makes many LLM calls
```

- Each finished case is saved to `eval/results/<id>__t0.7__r0.json`. Cases that already have a file are skipped,
  so the batch is **resumable**: if you hit Groq's daily limit, run it again later.
- Split the ids between teammates, each with their own Groq key, to stay inside the free tier.
- To redo a case, delete its result file.
- Ids must match exactly. A typo is skipped without any warning.

The cells after the batch build the report material: a results summary, a weight sweep, a temperature experiment,
a "judge the judge" check against human labels, and a cost breakdown (LLM calls, tokens, seconds per agent).

## Evaluation

Each test case is scored three ways, from most to least trustworthy:

1. **Code metrics**: validator violations, made-up places, cost vs budget, total travel minutes, backtracking
   ratio (actual travel ÷ shortest order of the same stops; 1.00 is ideal) and persona alignment (how closely
   visiting time per persona matches the weights).
2. **Pairwise LLM judge**: compares the two plans twice, once in each order. A verdict only counts if both orders
   agree; otherwise it is marked `inconsistent` (this catches position bias). The judge sees the code-computed
   travel time between stops, so it can spot plans that can't be done in time.
3. **Evaluator agent**: an LLM with a `get_metrics` tool that writes a scored report. Plans are labelled A/B in a
   random but repeatable order, so it doesn't know which one is the baseline. Off by default in batch runs
   (`with_agent_report=False`).

### Read the results with these in mind

- **Two metrics favour the multi-agent plan by construction.** Code writes its times and orders its stops with the
  same shortest-path solver the metric uses. So 0 violations and a 1.00 backtracking ratio show the code works,
  not that multiple agents plan better. Say so in the report.
- **Persona ratings are saturated.** The persona agents rate almost every place 5/5, so the personas barely steer
  which places get picked. When scores tie, the budget trim drops the most expensive place first.
- **The Qwen judge favours whichever plan is shown first.** The two-order check marks those verdicts
  `inconsistent`, so expect many of them.
- **The places data has chain duplicates**, e.g. three branches of the same restaurant, which the selector can
  pick together.
- Three Hanoi cases are not enough to call a winner.

## Repository layout

```
TravelWithAgents.ipynb      the whole project
.env                        your API keys (gitignored, never committed)
data/
  places/places_<city>.csv  cached Google Places data for 5 cities
  cache/routes.json         cached travel times between place pairs
  cache/persona_cache.json  cached persona agent outputs (saves LLM calls on re-runs)
eval/
  results/                  one JSON file per finished test case
```

Commit `data/` and `eval/results/` so your teammate can run without a Maps key and doesn't redo finished cases.

### Adding a city

Whoever has the Maps key runs `ingest_city("City", "Country")` once (the "Ingest your evaluation cities" cell
does this for all five cities when `GOOGLE_MAPS_API_KEY` is set) and commits the new CSV. Cities that are already
cached are skipped. Routing has a hard cap of 2,500 Routes API elements per session
(`ROUTES_ELEMENT_BUDGET`), to stay inside Google's free allowance.

## Troubleshooting

| Error | Cause | Fix |
|:--|:--|:--|
| `unexpected keyword argument 'reasoning_effort'` | `openai` older than 1.58 | rerun the install cell, then restart the kernel |
| `process() takes no keyword arguments` | `openai` 3.x with old Brotli | keep the `<2` pin, rerun the install cell, restart |
| `RuntimeError: No API key for provider 'groq'` | key not found | add `GROQ_API_KEY` to `.env` or Colab Secrets |
| `GOOGLE_MAPS_API_KEY is needed to ingest a new city` | new city, no Maps key | ask whoever has the key to ingest it, or use a cached city |
| `⏳ pacing ... waiting Ns` | staying under Groq's tokens-per-minute limit | not an error; just wait |
| `pip DEPRECATION: Loading egg ...` | an old package in your Python install | harmless warning, ignore it |
