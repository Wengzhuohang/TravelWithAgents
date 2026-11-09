"""Rebuild eval/blind/plans.md from eval/results/*.json for the blind human read.
Keeps the A/B assignment in key.json (new cases get a random one) and leaves "Your pick:" blank.
Run from the repo root: python3 eval/blind/make_plans.py"""
import glob, json, os, random

BLIND = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BLIND, "..", "results")
key_path = os.path.join(BLIND, "key.json")
key = json.load(open(key_path)) if os.path.exists(key_path) else {}


def plan_text(plan):
    out = []
    for d in plan["days"]:
        out.append(f"Day {d['day']} ({d['weekday']} {d['date']})")
        # names and times only: meal labels would reveal which system wrote the plan
        out += [f"  {s['start']}-{s['end']}  {s.get('name') or s.get('place_id')}" for s in d["stops"]]
    return "\n".join(out)


sections = []
for f in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
    tid = os.path.basename(f).split("__")[0]
    res = json.load(open(f, encoding="utf-8"))
    r = res["request"]
    if tid not in key:
        a = random.choice(["multi_agent", "baseline"])
        key[tid] = {"A": a, "B": "baseline" if a == "multi_agent" else "multi_agent"}
    plans = {"multi_agent": res["plan_multi"], "baseline": res["plan_baseline"]}
    head = (f"# {tid}: {r['city']}, {r['days']} days, {r['daily_start']}-{r['daily_end']}, "
            f"budget ${r['budget_usd']:.0f}, pace {r['pace']}, interests {', '.join(r['interests']) or '-'}, "
            f"avoid {', '.join(r['must_avoid']) or '-'}, {r.get('group_notes') or ''}")
    sections.append(f"{head}\n\n## Plan A\n{plan_text(plans[key[tid]['A']])}\n\n"
                    f"## Plan B\n{plan_text(plans[key[tid]['B']])}\n\nYour pick: \nNotes: \n---\n")

open(os.path.join(BLIND, "plans.md"), "w", encoding="utf-8").write("\n".join(sections))
json.dump(key, open(key_path, "w"), indent=1)
print(f"Wrote {len(sections)} cases to plans.md")
