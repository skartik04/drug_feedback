"""Run AI evaluation on all 120 traces (20 patients × 3 visits × 2 systems).

All 120 calls are fired in parallel (semaphore-limited to 20 concurrent).
Produces evaluations/eval_ai.csv in the exact same format as human evals.

Usage:
    conda run -n global_llm python doctor_cons_self/ai_eval/run_ai_eval.py
"""

import os
import sys
import json
import asyncio
import csv

import dotenv
dotenv.load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "consilium", ".env"))

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "consilium"))

from llm.client import LLMClient
from loader import build_patient_case, load_raw_data

# ── paths ──────────────────────────────────────────────────────────────────────

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(ROOT, "pdf_outputs")

SELECTED_IDS = os.path.join(ROOT, "selected_patients.json")
EVAL_DIR     = os.path.join(ROOT, "evaluations")
os.makedirs(EVAL_DIR, exist_ok=True)
OUT_CSV = os.path.join(EVAL_DIR, "eval_ai.csv")

CONS_CSV_OUTPUT = {v: os.path.join(ROOT, f"outputs/consilium/csv_v{v}.json") for v in [1, 2, 3]}
CONS_PDF_OUTPUT = {v: os.path.join(ROOT, f"outputs/consilium/pdf_v{v}.json") for v in [1, 2, 3]}
SL_CSV_OUTPUT   = {v: os.path.join(ROOT, f"outputs/self/csv_v{v}.json")      for v in [1, 2, 3]}
SL_PDF_OUTPUT   = {v: os.path.join(ROOT, f"outputs/self/pdf_v{v}.json")      for v in [1, 2, 3]}

CSV_FIELDS = [
    "pid", "visit", "system", "cohort",
    "seizure_type", "seizure_activity", "medications", "circumstances", "usefulness",
    "comment",
]

SYSTEM_PROMPT = """You are an expert epileptologist evaluating AI-generated clinical reasoning for epilepsy patients in a low-resource African hospital setting.

You will be given:
1. Patient clinical notes
2. AI-generated reasoning about the patient's drug management

Answer each of the following 5 questions strictly based on what is written in the reasoning. Do not infer or add clinical knowledge beyond what is present.

Return your answer as valid JSON only, with this exact structure:
{
  "seizure_type": "<Yes|Partially|No>",
  "seizure_activity": "<Yes|Partially|No>",
  "medications": "<Yes|Partially|No>",
  "circumstances": "<Yes|Partially|No>",
  "usefulness": "<Not useful|Somewhat useful|Very useful>",
  "comment": "<one sentence rationale>"
}

Questions:
1. seizure_type: Was the seizure type correctly identified?
2. seizure_activity: Was the current seizure burden / severity accurately assessed?
3. medications: Were the patient's current medications correctly accounted for?
4. circumstances: Was the drug selection reasoning clinically sound for this patient?
5. usefulness: Overall: Was this reasoning useful for managing this patient?"""


# ── helpers ────────────────────────────────────────────────────────────────────

def build_pdf_input(pid, visit_num, pdf_split):
    visits = pdf_split.get(pid, {})
    parts = []
    for v in range(1, visit_num + 1):
        vdata = visits.get(f"Visit_{v}", {})
        text = vdata.get("input_text", "")
        if not text:
            continue
        if v == visit_num:
            parts.append(f"[Visit {v} - Clinical Notes]\n{text}")
        else:
            parts.append(
                f"[Visit {v} - Clinical Notes]\n{text}\n\n"
                f"[Visit {v} - Prescription]\n{vdata.get('output_text', '(not recorded)')}"
            )
    return "\n\n".join(parts) or "(no data)"


def get_consilium_reasoning(sys_data):
    """Last epileptologist rebuttal — what's open in the UI."""
    debate = sys_data.get("debate", [])
    if debate:
        return debate[-1].get("epileptologist", "")
    epi = sys_data.get("epileptologist", {})
    return epi.get("reasoning", "") if isinstance(epi, dict) else ""


def get_self_reasoning(sys_data):
    """All 4 agent outputs + predictor reasoning — all open in the UI."""
    LABELS = {
        "SeizureBurdenWatcher":    "Seizure Burden",
        "SeizureSemiologyMapper":  "Seizure Semiology",
        "TreatmentIntentDetector": "Treatment Intent",
        "PriorMedicationExtractor":"Prior Medications",
    }
    parts = []
    for name, text in sys_data.get("agent_outputs", {}).items():
        parts.append(f"[{LABELS.get(name, name)}]\n{text}")
    reasoning = sys_data.get("reasoning", "")
    if reasoning:
        parts.append(f"[Reasoning]\n{reasoning}")
    return "\n\n".join(parts)


def parse_response(text):
    defaults = {
        "seizure_type": "", "seizure_activity": "", "medications": "",
        "circumstances": "", "usefulness": "", "comment": "",
    }
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            return defaults
        parsed = json.loads(text[start:end])
        for k in defaults:
            defaults[k] = str(parsed.get(k, "")).strip()
        return defaults
    except Exception:
        return defaults


# ── core eval coroutine ────────────────────────────────────────────────────────

async def eval_one(client, pid, visit_num, system, cohort, notes, reasoning, done_count, total):
    user_content = f"=== PATIENT CLINICAL NOTES ===\n{notes}\n\n=== AI REASONING ===\n{reasoning}"
    _, content = await client.call(SYSTEM_PROMPT, user_content)
    parsed = parse_response(content)
    n = done_count[0] = done_count[0] + 1
    print(f"  [{n}/{total}] {pid} v{visit_num} {system}")
    return {
        "pid":              pid,
        "visit":            visit_num,
        "system":           system,
        "cohort":           cohort,
        "seizure_type":     parsed["seizure_type"],
        "seizure_activity": parsed["seizure_activity"],
        "medications":      parsed["medications"],
        "circumstances":    parsed["circumstances"],
        "usefulness":       parsed["usefulness"],
        "comment":          parsed["comment"],
    }


# ── main ───────────────────────────────────────────────────────────────────────

async def main():
    split_results, clean_output, drug_gt, pid_to_row = load_raw_data()
    pdf_split = json.load(open(os.path.join(PDF_DIR, "split_results.json")))

    cons_csv = {v: json.load(open(p)) for v, p in CONS_CSV_OUTPUT.items() if os.path.exists(p)}
    cons_pdf = {v: json.load(open(p)) for v, p in CONS_PDF_OUTPUT.items() if os.path.exists(p)}
    sl_csv   = {v: json.load(open(p)).get("records", {}) for v, p in SL_CSV_OUTPUT.items() if os.path.exists(p)}
    sl_pdf   = {v: json.load(open(p)).get("records", {}) for v, p in SL_PDF_OUTPUT.items() if os.path.exists(p)}

    entries  = json.load(open(SELECTED_IDS))
    pid_meta = {e["pid"]: e for e in entries}
    ALL_PIDS = [e["pid"] for e in entries]

    UNITS = (
        [(p, "consilium") for p in ALL_PIDS[:10]] +
        [(p, "self")      for p in ALL_PIDS[:10]] +
        [(p, "consilium") for p in ALL_PIDS[10:]] +
        [(p, "self")      for p in ALL_PIDS[10:]]
    )

    # build all 120 (pid, visit, system, cohort, notes, reasoning) tuples
    traces = []
    for pid, system in UNITS:
        cohort = pid_meta[pid]["cohort"]
        is_pdf = cohort == "pdf"
        for visit_num in [1, 2, 3]:
            if is_pdf:
                notes = build_pdf_input(pid, visit_num, pdf_split)
            else:
                pc    = build_patient_case(pid, f"Visit_{visit_num}", split_results, clean_output, drug_gt, pid_to_row)
                notes = pc.build_input_text() if pc else "(no data)"

            if system == "consilium":
                sys_data  = (cons_pdf if is_pdf else cons_csv).get(visit_num, {}).get(pid, {})
                reasoning = get_consilium_reasoning(sys_data) if sys_data else ""
            else:
                sys_data  = (sl_pdf if is_pdf else sl_csv).get(visit_num, {}).get(pid, {})
                reasoning = get_self_reasoning(sys_data) if sys_data else ""

            if not reasoning:
                print(f"[SKIP] {pid} v{visit_num} {system} — no reasoning")
                continue
            traces.append((pid, visit_num, system, cohort, notes, reasoning))

    total  = len(traces)
    client = LLMClient(thinking_budget=0, max_tokens=512, max_concurrency=8)

    MAX_RETRY_ROUNDS = 3
    rows    = [None] * total
    pending = list(range(total))  # indices still needing a valid response

    for round_num in range(1, MAX_RETRY_ROUNDS + 1):
        if not pending:
            break
        done_count = [0]
        label = "Firing" if round_num == 1 else f"Retry round {round_num - 1} —"
        print(f"{label} {len(pending)} calls in parallel (concurrency=8)...")

        results = await asyncio.gather(*[
            eval_one(client, *traces[i], done_count, len(pending))
            for i in pending
        ])

        still_pending = []
        for i, row in zip(pending, results):
            if row["seizure_type"]:
                rows[i] = row
            else:
                still_pending.append(i)

        if still_pending:
            print(f"  {len(still_pending)} empty — retrying...")
        pending = still_pending

    for i in range(total):
        if rows[i] is None:
            pid, visit_num, system, cohort, _, _ = traces[i]
            rows[i] = {
                "pid": pid, "visit": visit_num, "system": system, "cohort": cohort,
                "seizure_type": "", "seizure_activity": "", "medications": "",
                "circumstances": "", "usefulness": "", "comment": "",
            }
            print(f"[FAILED] {pid} v{visit_num} {system}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    empty = sum(1 for r in rows if not r["seizure_type"])
    print(f"\nSaved {total} rows ({total - empty} valid, {empty} failed) → {OUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
