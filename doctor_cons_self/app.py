"""Consilium vs Self-Learning — Doctor Evaluation App (20 patients)."""

import os
import json
import pandas as pd
import streamlit as st

from loader import build_patient_case, load_raw_data

# ── paths ──────────────────────────────────────────────────────────────────────

ROOT     = os.path.dirname(os.path.abspath(__file__))
PDF_DIR  = os.path.join(ROOT, "pdf_outputs")

SELECTED_IDS = os.path.join(ROOT, "selected_patients.json")
EVAL_DIR     = os.path.join(ROOT, "evaluations")
os.makedirs(EVAL_DIR, exist_ok=True)

CONS_CSV_OUTPUT = {
    1: os.path.join(ROOT, "outputs/consilium/csv_v1.json"),
    2: os.path.join(ROOT, "outputs/consilium/csv_v2.json"),
    3: os.path.join(ROOT, "outputs/consilium/csv_v3.json"),
}
CONS_PDF_OUTPUT = {
    1: os.path.join(ROOT, "outputs/consilium/pdf_v1.json"),
    2: os.path.join(ROOT, "outputs/consilium/pdf_v2.json"),
    3: os.path.join(ROOT, "outputs/consilium/pdf_v3.json"),
}
SL_CSV_OUTPUT = {
    1: os.path.join(ROOT, "outputs/self/csv_v1.json"),
    2: os.path.join(ROOT, "outputs/self/csv_v2.json"),
    3: os.path.join(ROOT, "outputs/self/csv_v3.json"),
}
SL_PDF_OUTPUT = {
    1: os.path.join(ROOT, "outputs/self/pdf_v1.json"),
    2: os.path.join(ROOT, "outputs/self/pdf_v2.json"),
    3: os.path.join(ROOT, "outputs/self/pdf_v3.json"),
}

CONS_AGENT_LABELS = {
    "diagnostician":     "Diagnostician",
    "treatment_analyst": "Treatment Analyst",
    "pediatrician":      "Pediatrician",
    "formulary":         "Formulary",
    "tropical_medicine": "Tropical Medicine",
}
SL_AGENT_LABELS = {
    "SeizureBurdenWatcher":    "Seizure Burden",
    "SeizureSemiologyMapper":  "Seizure Semiology",
    "TreatmentIntentDetector": "Treatment Intent",
    "PriorMedicationExtractor":"Prior Medications",
}

INVISIBLE_CHARS = ("\u200b", "\u200c", "\u200d", "\u2060")

FEEDBACK_Q14 = [
    ("seizure_type",     "Was the seizure type correctly identified?"),
    ("seizure_activity", "Was the current seizure burden / severity accurately assessed?"),
    ("medications",      "Were the patient's current medications correctly accounted for?"),
    ("circumstances",    "Was the drug selection reasoning clinically sound for this patient?"),
]
OPT_Q14 = ["Yes", "Partially", "No"]
OPT_Q5  = ["Not useful", "Somewhat useful", "Very useful"]

# ── block / unit setup ─────────────────────────────────────────────────────────

entries  = json.load(open(SELECTED_IDS))
pid_meta = {e["pid"]: e for e in entries}
ALL_PIDS = [e["pid"] for e in entries]
PIDS_A   = ALL_PIDS[:10]
PIDS_B   = ALL_PIDS[10:]

UNITS = (
    [(p, "consilium") for p in PIDS_A] +
    [(p, "self")      for p in PIDS_A] +
    [(p, "consilium") for p in PIDS_B] +
    [(p, "self")      for p in PIDS_B]
)

TOTAL_EVALS = len(UNITS) * 3  # 120

# ── persistence ────────────────────────────────────────────────────────────────

def _eval_path(reviewer):
    return os.path.join(EVAL_DIR, f"eval_{reviewer}.csv")

def _nav_path(reviewer):
    return os.path.join(EVAL_DIR, f"nav_{reviewer}.json")

def load_evals(reviewer):
    path = _eval_path(reviewer)
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str).fillna("")
        return {(r["pid"], int(r["visit"]), r["system"]): r.to_dict() for _, r in df.iterrows()}
    return {}

def save_evals(reviewer, evals):
    pd.DataFrame(list(evals.values())).to_csv(_eval_path(reviewer), index=False)

def delete_evals(reviewer):
    p = _eval_path(reviewer)
    if os.path.exists(p): os.remove(p)

def load_nav(reviewer):
    p = _nav_path(reviewer)
    return json.load(open(p)) if os.path.exists(p) else {}

def save_nav(reviewer, unit_idx, visit):
    with open(_nav_path(reviewer), "w") as f:
        json.dump({"unit_idx": unit_idx, "visit": visit}, f)

def delete_nav(reviewer):
    p = _nav_path(reviewer)
    if os.path.exists(p): os.remove(p)

# ── data loading ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_data():
    split_results, clean_output, drug_gt, pid_to_row = load_raw_data()

    cons_csv = {v: json.load(open(p)) for v, p in CONS_CSV_OUTPUT.items() if os.path.exists(p)}
    cons_pdf = {v: json.load(open(p)) for v, p in CONS_PDF_OUTPUT.items() if os.path.exists(p)}

    sl_csv = {v: json.load(open(p)).get("records", {}) for v, p in SL_CSV_OUTPUT.items() if os.path.exists(p)}
    sl_pdf = {v: json.load(open(p)).get("records", {}) for v, p in SL_PDF_OUTPUT.items() if os.path.exists(p)}

    pdf_split = json.load(open(os.path.join(PDF_DIR, "split_results.json")))

    return split_results, clean_output, drug_gt, pid_to_row, cons_csv, cons_pdf, sl_csv, sl_pdf, pdf_split

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
    return "\n\n".join(parts) or "(no data for this visit)"


def _render_regimen(regimen):
    if not regimen:
        return
    with st.container(border=True):
        for n in [1, 2, 3]:
            opt = regimen.get(f"option_{n}", {})
            if not opt:
                continue
            label = opt.get("label", f"Option {n}")
            drugs = opt.get("drugs", {})
            if isinstance(drugs, dict):
                drugs_str = " | ".join(f"`{d}` {a}" for d, a in drugs.items())
            elif isinstance(drugs, list):
                drugs_str = " | ".join(f"`{d.get('drug','?')}` {d.get('action','')}" for d in drugs)
            else:
                drugs_str = str(drugs)
            rationale = opt.get("rationale", "")
            st.markdown(f"**Option {n}: {label}**")
            st.markdown(drugs_str)
            if rationale:
                st.caption(rationale)
            if n < 3:
                st.divider()


def build_feedback_record(pid, visit_num, system, cohort, fb_key):
    record = {
        "pid": pid, "visit": visit_num, "system": system, "cohort": cohort,
        "comment": st.session_state.get(f"comment_{fb_key}", ""),
    }
    complete = True
    for field, _ in FEEDBACK_Q14:
        val = st.session_state.get(f"{field}_{fb_key}")
        record[field] = val or ""
        if not val:
            complete = False
    val5 = st.session_state.get(f"usefulness_{fb_key}")
    record["usefulness"] = val5 or ""
    if not val5:
        complete = False
    return record, complete


def maybe_autosave(reviewer, pid, visit_num, system, cohort, fb_key):
    record, complete = build_feedback_record(pid, visit_num, system, cohort, fb_key)
    if complete:
        st.session_state.evals[(pid, visit_num, system)] = record
        save_evals(reviewer, st.session_state.evals)


def save_feedback(reviewer, pid, visit_num, system, cohort, fb_key):
    record, complete = build_feedback_record(pid, visit_num, system, cohort, fb_key)
    if not complete:
        st.warning("Please fill all questions before saving.")
        return
    st.session_state.evals[(pid, visit_num, system)] = record
    save_evals(reviewer, st.session_state.evals)


# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Doctor Evaluation — Reasoning Review", layout="wide")
st.title("Doctor Evaluation — Reasoning Review (20 Patients)")
st.markdown(
    """
    <style>
    div[data-testid="stCode"] pre {
        white-space: pre-wrap !important;
        overflow-x: hidden !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }
    div[data-testid="stCode"] pre > div { white-space: pre-wrap !important; }
    div[data-testid="stCode"] pre code {
        white-space: pre-wrap !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

(split_results, clean_output, drug_gt, pid_to_row,
 cons_csv, cons_pdf, sl_csv, sl_pdf, pdf_split) = load_data()

# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    existing_reviewer = st.session_state.get("reviewer", "")
    if st.button("Start Fresh"):
        target = existing_reviewer.strip()
        if not target:
            st.warning("Enter your name first.")
        else:
            delete_evals(target)
            delete_nav(target)
            st.session_state.evals = {}
            st.session_state.unit_idx = 0
            st.session_state["visit_select"] = 1
            for key in [k for k in list(st.session_state.keys()) if f"__{target}" in k]:
                del st.session_state[key]
            st.rerun()

    st.divider()
    st.header("Reviewer")
    reviewer = st.text_input("Enter your name", value=existing_reviewer)
    if not reviewer.strip():
        st.warning("Enter your name above to begin.")
        st.stop()
    reviewer = reviewer.strip()

    if st.session_state.get("reviewer") != reviewer:
        st.session_state.reviewer = reviewer
        st.session_state.evals = load_evals(reviewer)
        nav = load_nav(reviewer)
        st.session_state.unit_idx = min(nav.get("unit_idx", 0), len(UNITS) - 1)
        st.session_state["visit_select"] = nav.get("visit", 1) if nav.get("visit") in [1, 2, 3] else 1

    evals = st.session_state.evals
    st.caption(f"Progress: {len(evals)} / {TOTAL_EVALS} reviewed")

    if evals:
        csv_bytes = pd.DataFrame(list(evals.values())).to_csv(index=False).encode()
        st.download_button(
            "Download feedback", data=csv_bytes,
            file_name=f"eval_{reviewer}.csv", mime="text/csv",
        )

    st.divider()
    st.header("Navigation")

    visit_num = st.radio(
        "Visit", [1, 2, 3], format_func=lambda v: f"Visit {v}",
        horizontal=True, key="visit_select",
    )

    if "unit_idx" not in st.session_state:
        st.session_state.unit_idx = 0
    st.session_state.unit_idx = max(0, min(st.session_state.unit_idx, len(UNITS) - 1))

    col_prev, col_next = st.columns(2)
    if col_prev.button("← Prev") and st.session_state.unit_idx > 0:
        st.session_state.unit_idx -= 1
        st.rerun()
    if col_next.button("Next →") and st.session_state.unit_idx < len(UNITS) - 1:
        st.session_state.unit_idx += 1
        st.rerun()

    unit_idx = st.session_state.unit_idx
    unit_labels = [f"{i+1}. {p}" for i, (p, _) in enumerate(UNITS)]
    selected = st.selectbox("Patient", unit_labels, index=unit_idx, key="pat_select")
    new_idx = unit_labels.index(selected)
    if new_idx != unit_idx:
        st.session_state.unit_idx = new_idx
        st.rerun()
    st.caption(f"{unit_idx + 1} of {len(UNITS)}")

    save_nav(reviewer, st.session_state.unit_idx, visit_num)

# ── main ───────────────────────────────────────────────────────────────────────

unit_idx = st.session_state.unit_idx
pid, system = UNITS[unit_idx]
meta    = pid_meta[pid]
cohort  = meta["cohort"]
is_pdf  = cohort == "pdf"

expander_scope = (pid, visit_num, system)
if st.session_state.get("expander_scope") != expander_scope:
    st.session_state.expander_scope = expander_scope
    st.session_state.expander_nonce = st.session_state.get("expander_nonce", 0) + 1
expander_nonce = st.session_state.get("expander_nonce", 0)

def scoped(label, tag):
    token  = f"{expander_nonce}:{tag}"
    hidden = "".join(INVISIBLE_CHARS[ord(ch) % len(INVISIBLE_CHARS)] for ch in token)
    return f"{label}{hidden}"

# patient input text
if is_pdf:
    full_input = build_pdf_input(pid, visit_num, pdf_split)
else:
    visit_name   = f"Visit_{visit_num}"
    patient_case = build_patient_case(pid, visit_name, split_results, clean_output, drug_gt, pid_to_row)
    full_input   = patient_case.build_input_text() if patient_case else "(no data for this visit)"

# system output
if system == "consilium":
    sys_data = (cons_pdf if is_pdf else cons_csv).get(visit_num, {}).get(pid, {})
else:
    sys_data = (sl_pdf if is_pdf else sl_csv).get(visit_num, {}).get(pid, {})

system_label = "System A" if system == "consilium" else "System B"
st.subheader(f"{pid}  —  Visit {visit_num}  —  {system_label}")
st.divider()

left, right = st.columns([1, 1], gap="large")

# ── LEFT: patient notes + feedback ────────────────────────────────────────────

with left:
    st.markdown("### Patient Notes")
    st.code(full_input, language=None)

    st.markdown("---")
    st.markdown("### Feedback")

    fb_key   = f"{pid}__v{visit_num}__{system}__{reviewer}"
    existing = evals.get((pid, visit_num, system), {})
    st.caption(
        "Answers auto-save once all 5 questions are filled. "
        "Use Save Feedback to save comment changes."
    )

    for field, question in FEEDBACK_Q14:
        q_key = f"{field}_{fb_key}"
        if q_key not in st.session_state:
            st.session_state[q_key] = existing.get(field) or None
        st.radio(
            question, OPT_Q14, index=None, horizontal=True, key=q_key,
            on_change=maybe_autosave,
            args=(reviewer, pid, visit_num, system, cohort, fb_key),
        )

    u_key = f"usefulness_{fb_key}"
    if u_key not in st.session_state:
        st.session_state[u_key] = existing.get("usefulness") or None
    st.radio(
        "Overall: Was this reasoning useful for managing this patient?",
        OPT_Q5, index=None, horizontal=True, key=u_key,
        on_change=maybe_autosave,
        args=(reviewer, pid, visit_num, system, cohort, fb_key),
    )

    st.text_area(
        "Comment (optional)", value=existing.get("comment", ""),
        height=80, key=f"comment_{fb_key}",
    )
    if st.button("Save Feedback", key=f"save_{fb_key}"):
        save_feedback(reviewer, pid, visit_num, system, cohort, fb_key)

    if (pid, visit_num, system) in evals:
        st.success("Saved")
    else:
        filled = sum(1 for field, _ in FEEDBACK_Q14 if st.session_state.get(f"{field}_{fb_key}"))
        filled += 1 if st.session_state.get(u_key) else 0
        if filled < 5:
            st.caption(f"{5 - filled} question(s) left before auto-save.")

# ── RIGHT: system output ───────────────────────────────────────────────────────

with right:
    if system == "consilium":
        if not sys_data:
            st.warning("No pipeline output for this patient/visit.")
        else:
            st.markdown("### Phase 0 — Orchestrator")
            orch = sys_data.get("orchestrator", {})
            decisions = orch.get("decisions", [])
            if decisions:
                with st.expander(scoped("Agent Activation Decisions", "orchestrator"), expanded=False):
                    for d in decisions:
                        icon = "+" if d.get("activated") else "-"
                        st.markdown(f"**{icon} {d.get('agent','?')}** — {d.get('reason','')}")
            else:
                st.info("No orchestrator output")

            st.markdown("### Phase 1 — Specialist Agents")
            phase1 = sys_data.get("phase1", {})
            if phase1:
                for agent_name, agent_text in phase1.items():
                    label = CONS_AGENT_LABELS.get(agent_name, agent_name.title())
                    with st.expander(scoped(label, f"phase1_{agent_name}"), expanded=False):
                        if isinstance(agent_text, str):
                            st.markdown(agent_text)
                        elif isinstance(agent_text, dict):
                            st.json(agent_text)
                        else:
                            st.text(str(agent_text))
            else:
                st.info("No Phase 1 outputs")

            st.markdown("### Phase 2")
            epi = sys_data.get("epileptologist", {})
            if epi:
                with st.expander(scoped("Epileptologist", "phase2_epi"), expanded=False):
                    if epi.get("reasoning"):
                        st.markdown(epi["reasoning"])
                    _render_regimen(epi.get("regimen", {}))
            else:
                st.info("No epileptologist output")

            st.markdown("### Adversarial Review")
            pharma = sys_data.get("pharmacologist", "")
            if pharma:
                with st.expander(scoped("Pharmacologist", "pharmacologist"), expanded=False):
                    if isinstance(pharma, str):
                        st.markdown(pharma)
                    else:
                        st.json(pharma)
            else:
                st.info("No pharmacologist output")

            debate = sys_data.get("debate", [])
            if debate:
                st.markdown("### Epileptologist Prediction")
                last_idx = len(debate) - 1
                for idx, rnd in enumerate(debate):
                    with st.expander(scoped("Epileptologist Rebuttal", f"debate_{idx}"), expanded=(idx == last_idx)):
                        if rnd.get("epileptologist"):
                            st.markdown(rnd["epileptologist"])
                        _render_regimen(rnd.get("epileptologist_regimen", {}))

    else:  # self-learning
        if not sys_data:
            st.warning("No output found for this patient / visit.")
            st.stop()
        # Agent outputs — bordered box, all expanded
        agent_outputs = sys_data.get("agent_outputs", {})
        if agent_outputs:
            st.markdown("### Agent Outputs")
            with st.container(border=True):
                for name, text in agent_outputs.items():
                    label = SL_AGENT_LABELS.get(name, name)
                    with st.expander(scoped(label, f"agent_{name}"), expanded=True):
                        st.markdown(str(text))

        # Reasoning — main
        st.markdown("### Reasoning")
        reasoning = sys_data.get("reasoning", "")
        if reasoning:
            st.markdown(reasoning)
        else:
            st.info("No reasoning output.")

        regimen = sys_data.get("final_regimen", {})
        if regimen:
            if isinstance(regimen, dict):
                _render_regimen(regimen)
            else:
                st.markdown(str(regimen))
