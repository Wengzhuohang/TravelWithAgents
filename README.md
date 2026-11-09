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
| 2 | Persona agents: Adventurer, Foodie, Culture buff | 3 × LLM | each persona's ratings and draft timetable |
| 3 | Selector | code | final places, split between personas by **weights that sum to 1** |
| 4 | Route optimiser | code + Google Routes API | places grouped into days by area, shortest order per day, timed schedule |
| 5 | Validator + Feasibility agent | code + LLM with tools, up to 3 repair rounds | a plan with no hard violations |
| 6 | Baseline | one LLM prompt | the same task in a single call, for comparison |
| 7 | Evaluator | code metrics + pairwise LLM judge + evaluator agent | scores for both plans |
| 8 | Re-plan | code (cached persona outputs) | a new plan for different weights, with no new LLM calls |

An LLM never runs code itself. When it wants a tool, it returns a structured request, and the harness
(the "LLM harness" cell) runs the Python function, sends the result back, enforces a step budget and logs every
call.

## Models

All models are called over the API, so a plain CPU runtime is enough.

| Role | Model | Provider |
|:--|:--|:--|
| `strong`: feasibility agent and baseline | `openai/gpt-oss-120b` | Groq |
| `small`: parser, scout, personas | `openai/gpt-oss-20b` | Groq |
| `judge`: pairwise judge and evaluator agent | `gemini-3.5-flash` (needs `GEMINI_API_KEY`) | Gemini |

The baseline and the feasibility agent share the same model, so the multi-agent system never gets a bigger model
than the baseline. The judge comes from a different model family than the planners, to avoid self-preference
bias. If that Gemini model isn't available on your key, the harness tries the next one in `GEMINI_CANDIDATES`.
Without a Gemini key (or if no candidate works), the judge falls back to `openai/gpt-oss-20b` on Groq: the same
family as the planners, so say so in the report. Models are set in the `MODELS` dict in the configuration cell.

## Setup

### 1. API keys

| Key | Needed for | Required? |
|:--|:--|:--|
| `GROQ_API_KEY` | every LLM call | **yes** |
| `GOOGLE_MAPS_API_KEY` | ingesting new cities (Places API) and real travel times (Routes API) | no |
| `GEMINI_API_KEY` | the judge and evaluator agent (a different model family from the planners) | no, but recommended |

Copy `.env.example` to `.env` and fill in the keys (`KEY=value`, one per line). The `.env` file goes next to the notebook or one folder up. It is
gitignored, so it is never committed. Keys already set as environment variables also work. If `GROQ_API_KEY` is
missing, the notebook asks you to type it in.

Never paste a key into a notebook cell: outputs get committed.

Without a Maps key the notebook still runs. It uses the cached city files in `data/places/`, and it estimates
travel times from straight-line distance for any pair not already in `data/cache/routes.json` (marked
`estimate` in the outputs).

### 2. Packages

The first code cell installs them:

```
%pip install -q "openai>=1.40" pandas requests scikit-learn matplotlib
```

Restart the kernel after it installs or upgrades anything. The harness sends `reasoning_effort` for the `gpt-oss`
models, so an old `openai` package will fail (see Troubleshooting). It also asks for gzip instead of Brotli, which
avoids a crash with older Brotli installs (common in Anaconda).

### 3. Run

Open the notebook and run the cells from the top. Steps 0–8 walk through one sample request (Hanoi) so you can
see each agent's output. Edit `TRIP_TEXT` (Step 0) and `WEIGHTS` (the cell after it) to try your own.

- **The request must name the city and the trip length** ("3 days", "one day", or a start *and* end date).
  Vague requests like "a weekend in Hanoi" are rejected with a message saying what to add. Add `"start_date"` or
  `"budget_usd"` to `REQUIRED_TRIP_FIELDS` to make it stricter.
- Anything else not stated gets a default: start date 2 weeks from today, 09:00–20:00, normal pace, a budget of
  40 USD × days.
- **Weights must sum to 1** (they are re-normalised with a warning if they don't). They decide how much each
  persona counts when places are selected.

## Run the tests yourself

These steps take you from a fresh clone to your own run of the 12-case test set, using your own API keys.

**1. Clone the repo and add your keys.**

```bash
git clone https://github.com/Wengzhuohang/TravelWithAgents.git
cd TravelWithAgents
cp .env.example .env      # then open .env and paste in your keys
```

You need `GROQ_API_KEY` (free at https://console.groq.com/keys). `GEMINI_API_KEY` is recommended, so the judge comes
from a different model family. `GOOGLE_MAPS_API_KEY` is optional (see step 2).

**2. Check that the city data is there.**

```bash
ls data/places
```

All 12 cases need 5 files: `places_bangkok.csv`, `places_da_nang.csv`, `places_hanoi.csv`,
`places_ho_chi_minh_city.csv` and `places_hoi_an.csv`. If one is missing, you have two choices:

- add a `GOOGLE_MAPS_API_KEY`, and the notebook fetches that city automatically (about 10 Places API calls per city,
  inside Google's free monthly allowance), or
- leave that city's cases out of `ONLY_IDS` in step 4.

**3. Move the existing results aside.** The batch skips any case that already has a result file, so it would
just reuse ours. Move them out of the way first:

```bash
mv eval/results eval/results_original
mv eval/blind/plans.md eval/blind/plans_original.md
mv eval/blind/key.json eval/blind/key_original.json
rm -f data/cache/persona_cache.json   # optional: forces fresh persona calls instead of reusing ours
```

Keep `data/cache/routes.json`. It only holds travel times between places, and keeping it saves Maps calls.

**4. Run the notebook.**

1. Open `TravelWithAgents.ipynb` in Jupyter or VS Code.
2. Run the first code cell (package install), then restart the kernel.
3. Check the batch cell in the Experiments section: `RUN_BATCH = True`, and `ONLY_IDS` lists the cases you want.
   Set `RUN_TEMP_EXPERIMENT = False` if you want to skip the temperature experiment (18 extra LLM calls).
4. Use **Run All**.

The notebook first walks through the Hanoi demo (Steps 0–8), then runs the batch. Each case takes about 2–5
minutes and 15–35k tokens. All 12 take roughly 35 minutes and about 270k tokens. If Groq's free-tier limit runs
out partway, the remaining cases print `FAILED`. Run the batch cell again later: finished cases are skipped, so
it carries on where it stopped.

**5. Read the results.** The "Results summary" cell compares the multi-agent plans with the baseline
(violations, feasibility, backtracking, persona alignment, judge verdicts). The last cell shows the cost per
agent. Our original results are in `eval/results_original/` if you want to compare.

**6. Add your own human labels (optional).** On its first run, the cell above "Judge the judge" writes
`eval/blind/plans.md`: every finished case as plan A and plan B, with no hint of which is which. Fill in each
case's `Your pick:` line with `A`, `B` or `tie`, then re-run the "Judge the judge" cells to see how often the
LLM judge agrees with you. The sheet is only built when `plans.md` doesn't exist yet, so your picks are never
overwritten.

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
RUN_BATCH = True                   # set False to skip: each case makes many LLM calls
```

- Each finished case is saved to `eval/results/<id>__t0.7__r0.json`. Cases that already have a file are skipped,
  so the batch is **resumable**: if you hit Groq's daily limit, run it again later.
- Run a few ids at a time (`ONLY_IDS`) to stay inside Groq's free tier.
- To redo a case, delete its result file.
- Ids must match exactly. A typo is skipped without any warning.

The cells after the batch build the report material: a results summary, a weight sweep, a temperature experiment,
a "judge the judge" check against human labels, and a cost breakdown (LLM calls, tokens, seconds per agent).

### Human labels for "judge the judge"

1. The notebook writes every finished case to `eval/blind/plans.md` as plan A and plan B the first time it runs
   the cell above "Judge the judge" (only if `plans.md` doesn't exist yet). It keeps which is which in
   `eval/blind/key.json`. `python3 eval/blind/make_plans.py` rebuilds the sheet for new cases, but it also
   blanks every pick already filled in.
2. Read each case and fill in its `Your pick:` line with `A`, `B` or `tie`.
3. Run the "Judge the judge" cell. It unblinds the picks with `key.json` and reports how often the LLM judge
   agrees with you.

## Evaluation

Each test case is scored three ways, from most to least trustworthy:

1. **Code metrics**: validator violations, made-up places, cost vs budget, total travel minutes, backtracking
   ratio (actual travel ÷ shortest order of the same stops; 1.00 is ideal) and persona alignment (how closely
   visiting time per persona matches the weights).
2. **Pairwise LLM judge**: compares the two plans twice, once in each order. A verdict only counts if both orders
   agree; otherwise it is marked `inconsistent` (this catches position bias). The judge sees each stop's times,
   name, category and estimated cost.
3. **Evaluator agent**: an LLM with a `get_metrics` tool that writes a scored report. Plans are labelled A/B in a
   random but repeatable order, so it doesn't know which one is the baseline. Off by default in batch runs
   (`with_agent_report=False`).

### Read the results with these in mind

- **Two metrics favour the multi-agent plan by construction.** Code writes its times and orders its stops with the
  same travel matrix the metrics use. So low violations and backtracking show the code works, not that multiple
  agents plan better. Backtracking is not exactly 1.00, because meals are inserted after the route is ordered.
- **Check whether persona ratings are saturated.** If the persona agents rate almost every place 5/5, the
  personas barely steer which places get picked. The weight sweep shows whether the weights still work.
- **Position bias shows up as `inconsistent`.** Count those verdicts separately; they are not wins for either side.
- **The places data has chain duplicates**, e.g. three branches of the same restaurant, which the selector can
  pick together.
- 12 cases across 5 cities is a small sample: report it as such.

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
  blind/                    blind human read: make_plans.py, plans.md, key.json
```

`eval/` is committed, so a fresh clone has the finished cases and the human labels. `data/` is in `.gitignore`.
Commit it with `git add -f` when you want the cached places and travel times kept with the code, so the notebook
can be re-run without a Maps key.

### The `eval/blind/` folder

This folder checks the LLM judge against people. The judge decides which plan is better, multi-agent or
baseline. To know whether its verdicts mean anything, we read the same pairs of plans ourselves and compared our
picks with the judge's. The read is **blind**: each plan is shown only as "Plan A" or "Plan B", with no meal labels
or other hints about which system wrote it, so our own expectations can't sway the pick.

| File | What it is |
|:--|:--|
| `plans.md` | the review sheet: for each test case, the trip request, then Plan A and Plan B (times and place names only), then a `Your pick:` line (`A`, `B` or `tie`) and a `Notes:` line explaining the pick |
| `key.json` | the answer key: for each case, which system is A and which is B (assigned at random). Don't open it until all the picks are filled in |
| `make_plans.py` | rebuilds `plans.md` from `eval/results/`, keeping the A/B assignment in `key.json`. It blanks every pick, so only use it before you start reviewing |

The "Judge the judge" cell reads the picks, unblinds them with `key.json`, and reports how often the judge agrees
with the human reader and how many judge verdicts flipped when the plan order was swapped (position bias). If
`plans.md` doesn't exist, the cell above it builds a blank sheet from the current results.

The committed `plans.md` holds our picks and notes for all 12 cases. To do your own blind read, move it and
`key.json` aside first (see [Run the tests yourself](#run-the-tests-yourself), step 3).

### Adding a city

With a Maps key, a request for a city that isn't cached is ingested automatically. You can also run
`ingest_city("City", "Country")` yourself. Ingestion searches inside Google's own boundary for the city: 1 boundary call + 9 searches per city. Commit
the new CSV. Cached cities are skipped unless you pass `force=True`. Routing has a hard cap of 2,500 Routes API elements per session
(`ROUTES_ELEMENT_BUDGET`), to stay inside Google's free allowance.

## Troubleshooting

| Error | Cause | Fix |
|:--|:--|:--|
| `unexpected keyword argument 'reasoning_effort'` | old `openai` package | `%pip install -U openai`, then restart the kernel |
| `RuntimeError: No API key for provider 'groq'` | key not found | add `GROQ_API_KEY` to `.env` |
| `TripRequestIncomplete: Please make the request more specific` | request has no city or trip length | add what the message asks for |
| `⚠️ Gemini model '...' is not available` | that model is retired or not on your key | not an error; it tries the next one |
| `GOOGLE_MAPS_API_KEY is needed to ingest a new city` | new city, no Maps key | add a Maps key, or use a cached city |
| `⏳ pacing ... waiting Ns` | staying under Groq's tokens-per-minute limit | not an error; just wait |
| `pip DEPRECATION: Loading egg ...` | an old package in your Python install | harmless warning, ignore it |
