"""Consilium — Doctor Review App (70 sampled patients)."""

import os
import json
import pandas as pd
import streamlit as st

from loader import build_patient_case, load_raw_data

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(__file__)
DATA = os.path.join(ROOT, "data")
PROCESSED = os.path.join(DATA, "processed")
PDF_DIR = os.path.join(ROOT, "pdf_outputs")

REVIEW_IDS = os.path.join(os.path.dirname(__file__), "doctor_review_ids.json")
EVAL_DIR = os.path.join(os.path.dirname(__file__), "evaluations")
os.makedirs(EVAL_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA, "combined_dataset.csv")

V2_OUTPUT = {
    1: os.path.join(ROOT, "outputs/v2/consilium_v2_openai.gptoss120b1:0_v1_d0.json"),
    2: os.path.join(ROOT, "outputs/v2/consilium_v2_openai.gptoss120b1:0_v2_d0.json"),
    3: os.path.join(ROOT, "outputs/v2/consilium_v2_openai.gptoss120b1:0_v3_d0.json"),
}
BASELINE_OUTPUT = {
    1: os.path.join(ROOT, "outputs/baseline/openai.gptoss120b1:0_v1_baseline.json"),
    2: os.path.join(ROOT, "outputs/baseline/openai.gptoss120b1:0_v2_baseline.json"),
    3: os.path.join(ROOT, "outputs/baseline/openai.gptoss120b1:0_v3_baseline.json"),
}

PDF_V2_OUTPUT = {
    1: os.path.join(PDF_DIR, "predictions/pdf_consilium_openai.gptoss120b1:0_v1_d0.json"),
    2: os.path.join(PDF_DIR, "predictions/pdf_consilium_openai.gptoss120b1:0_v2_d0.json"),
    3: os.path.join(PDF_DIR, "predictions/pdf_consilium_openai.gptoss120b1:0_v3_d0.json"),
}
PDF_BL_OUTPUT = {
    1: os.path.join(PDF_DIR, "predictions/pdf_baseline_openai.gptoss120b1:0_v1.json"),
    2: os.path.join(PDF_DIR, "predictions/pdf_baseline_openai.gptoss120b1:0_v2.json"),
    3: os.path.join(PDF_DIR, "predictions/pdf_baseline_openai.gptoss120b1:0_v3.json"),
}

DR_COLS = {
    1: "Current dose",
    2: "Medication dosage and if there was a change in medication(6 months)",
    3: "Medication dosage and if there was a change in medication(12 months)",
}
ENTRY_COLS = {
    1: [
        ("History of Presenting Illness", "History of Presenting Illness"),
        ("Medication status", "Medication Status"),
        ("Current drug regimen", "Current Drug Regimen"),
    ],
    2: [
        ("Second Entry(6 months)", "Second Entry (6 months)"),
        ("Medication dosage and if there was a change in medication(6 months)", "Medication Dosage (6 months)"),
    ],
    3: [
        ("Third Entry(12 months)", "Third Entry (12 months)"),
        ("Medication dosage and if there was a change in medication(12 months)", "Medication Dosage (12 months)"),
    ],
}

AGENT_LABELS = {
    "diagnostician": "Diagnostician",
    "treatment_analyst": "Treatment Analyst",
    "pediatrician": "Pediatrician",
    "formulary": "Formulary",
    "tropical_medicine": "Tropical Medicine",
}
PROMPTS_DIR = os.path.join(ROOT, "prompts")
PROMPT_FILES = [
    ("orchestrator", "orchestrator.txt"),
    ("diagnostician", "diagnostician.txt"),
    ("treatment_analyst", "treatment_analyst.txt"),
    ("pediatrician", "pediatrician.txt"),
    ("formulary", "formulary.txt"),
    ("tropical_medicine", "tropical_medicine.txt"),
    ("epileptologist", "epileptologist.txt"),
    ("pharmacologist", "pharmacologist.txt"),
]
INVISIBLE_CHARS = ("\u200b", "\u200c", "\u200d", "\u2060")
FEEDBACK_OPTIONS = ["Yes", "No", "N/A"]
FEEDBACK_QUESTIONS = [
    ("stage_1", "Stage 1: Was seizure-type compatibility reasoning appropriate?"),
    ("stage_2", "Stage 2: Were safety / special-population modifiers handled appropriately?"),
    ("stage_3", "Stage 3: Were practicality / cost / LMIC constraints handled appropriately?"),
    ("stage_4", "Stage 4: Were side-effect and comorbidity tradeoffs handled appropriately?"),
    ("stage_5", "Stage 5: Was prior medication history interpreted appropriately?"),
    ("stage_6", "Stage 6: Was monotherapy optimization considered appropriately?"),
    ("stage_7", "Stage 7: Was polytherapy reasoning appropriate, or appropriately avoided?"),
    ("final_regimen", "Overall: Is the final regimen clinically acceptable?"),
]


# ── helpers ───────────────────────────────────────────────────────────────────


def load_evals(reviewer):
    path = os.path.join(EVAL_DIR, f"dr_review_{reviewer}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str).fillna("")
        return {(r["pid"], int(r["visit"])): r.to_dict() for _, r in df.iterrows()}
    return {}


def save_evals(reviewer, evals):
    path = os.path.join(EVAL_DIR, f"dr_review_{reviewer}.csv")
    pd.DataFrame(list(evals.values())).to_csv(path, index=False)


def build_pdf_input(pid, visit_num, pdf_split):
    """Build cumulative context string for a PDF patient up to visit_num."""
    visits = pdf_split.get(pid, {})
    parts = []
    for v in range(1, visit_num + 1):
        vk = f"Visit_{v}"
        vdata = visits.get(vk, {})
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


# ── load all data ─────────────────────────────────────────────────────────────


@st.cache_resource
def load_data():
    split_results, clean_output, drug_gt, pid_to_row = load_raw_data()

    v2_outputs = {}
    for v, path in V2_OUTPUT.items():
        v2_outputs[v] = json.load(open(path)) if os.path.exists(path) else {}

    baseline_outputs = {}
    for v, path in BASELINE_OUTPUT.items():
        baseline_outputs[v] = json.load(open(path)) if os.path.exists(path) else {}

    df = pd.read_csv(
        CSV_PATH, sep=";", engine="python", quotechar='"', doublequote=True, escapechar="\\"
    )
    df = df.drop_duplicates(subset=["Record ID"])
    df["_rid"] = pd.to_numeric(df["Record ID"], errors="coerce")
    csv_lookup = df.set_index("_rid").to_dict("index")

    pdf_split = json.load(open(os.path.join(PDF_DIR, "split_results.json")))

    pdf_v2_outputs = {}
    for v, path in PDF_V2_OUTPUT.items():
        pdf_v2_outputs[v] = json.load(open(path)) if os.path.exists(path) else {}

    pdf_bl_outputs = {}
    for v, path in PDF_BL_OUTPUT.items():
        pdf_bl_outputs[v] = json.load(open(path)) if os.path.exists(path) else {}

    return (
        split_results, clean_output, drug_gt, pid_to_row,
        v2_outputs, baseline_outputs, csv_lookup,
        pdf_split, pdf_v2_outputs, pdf_bl_outputs,
    )


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Consilium — Doctor Review", layout="wide")
st.title("Consilium — Doctor Review (70 Patients)")
st.markdown(
    """
    <style>
    div[data-testid="stCode"] pre {
        white-space: pre-wrap !important;
        overflow-x: hidden !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }
    div[data-testid="stCode"] pre > div {
        white-space: pre-wrap !important;
    }
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
 v2_outputs, baseline_outputs, csv_lookup,
 pdf_split, pdf_v2_outputs, pdf_bl_outputs) = load_data()

review_entries = json.load(open(REVIEW_IDS))
pid_meta = {e["pid"]: e for e in review_entries}
PIDS = [e["pid"] for e in review_entries]

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Reviewer")
    reviewer = st.text_input("Enter your name", value="")

    if not reviewer.strip():
        st.warning("Enter your name above to begin.")
        st.stop()

    reviewer = reviewer.strip()

    if st.session_state.get("reviewer") != reviewer:
        st.session_state.reviewer = reviewer
        st.session_state.evals = load_evals(reviewer)

    evals = st.session_state.evals
    st.caption(f"Feedback saved: {len(evals)}")

    if evals:
        csv_bytes = pd.DataFrame(list(evals.values())).to_csv(index=False).encode()
        st.download_button(
            "Download feedback",
            data=csv_bytes,
            file_name=f"dr_review_{reviewer}.csv",
            mime="text/csv",
        )

    st.divider()
    st.header("Navigation")

    visit_num = st.radio(
        "Visit", [1, 2, 3], format_func=lambda v: f"Visit {v}", horizontal=True
    )

    if "pat_idx" not in st.session_state:
        st.session_state.pat_idx = 0
    st.session_state.pat_idx = min(st.session_state.pat_idx, len(PIDS) - 1)
    pending_pid = None

    col_prev, col_next = st.columns(2)
    if col_prev.button("Prev"):
        new_idx = max(0, st.session_state.pat_idx - 1)
        pending_pid = PIDS[new_idx]
    if col_next.button("Next"):
        new_idx = min(len(PIDS) - 1, st.session_state.pat_idx + 1)
        pending_pid = PIDS[new_idx]

    st.divider()
    search = st.text_input("Search patient name/ID", "")
    if search.strip():
        matches = [p for p in PIDS if search.lower() in p.lower()]
        if matches:
            st.caption(f"{len(matches)} matches")
            for m in matches[:15]:
                if st.button(m, key=f"search_{m}"):
                    pending_pid = m
        else:
            st.caption("No matches")

    if pending_pid is not None:
        st.session_state.pat_idx = PIDS.index(pending_pid)
        st.session_state["pat_select"] = pending_pid
    elif "pat_select" not in st.session_state:
        st.session_state["pat_select"] = PIDS[st.session_state.pat_idx]

    selected_pid = st.selectbox(
        "Patient", PIDS, index=st.session_state.pat_idx, key="pat_select"
    )
    st.session_state.pat_idx = PIDS.index(selected_pid)
    st.caption(f"{st.session_state.pat_idx + 1} / {len(PIDS)} patients")

# ── main ──────────────────────────────────────────────────────────────────────

pid = selected_pid
meta = pid_meta[pid]
cohort = meta["cohort"]
therapy_cls = meta["therapy_cls"]
visit_name = f"Visit_{visit_num}"
is_pdf = cohort == "pdf"

expander_scope = (pid, visit_num)
if st.session_state.get("expander_scope") != expander_scope:
    st.session_state.expander_scope = expander_scope
    st.session_state.expander_nonce = st.session_state.get("expander_nonce", 0) + 1
expander_nonce = st.session_state.get("expander_nonce", 0)


def scoped_expander_label(label, tag):
    token = f"{expander_nonce}:{tag}"
    hidden = "".join(INVISIBLE_CHARS[ord(ch) % len(INVISIBLE_CHARS)] for ch in token)
    return f"{label}{hidden}"

if is_pdf:
    v2_patient = pdf_v2_outputs.get(visit_num, {}).get(pid, {})
else:
    v2_patient = v2_outputs.get(visit_num, {}).get(pid, {})

# Patient input context
if is_pdf:
    full_input = build_pdf_input(pid, visit_num, pdf_split)
    patient_case = None
else:
    patient_case = build_patient_case(pid, visit_name, split_results, clean_output, drug_gt, pid_to_row)
    full_input = patient_case.build_input_text() if patient_case else "(no data for this visit)"

st.subheader(f"{pid}  —  Visit {visit_num}")
st.divider()

left, right = st.columns([1, 1], gap="large")

# ── LEFT: Patient data + doctor output + feedback ─────────────────────────────

with left:
    st.markdown("### Patient Input (Full Agent Context)")
    st.code(full_input, language=None)

    # ── Feedback ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Feedback")

    fb_key = f"{pid}__v{visit_num}__{reviewer}"
    existing = st.session_state.evals.get((pid, visit_num), {})
    st.caption("Answer each item with Yes, No, or N/A. Use N/A if not applicable or not assessable.")

    responses = {}
    for field, question in FEEDBACK_QUESTIONS:
        q_key = f"{field}_{fb_key}"
        if q_key not in st.session_state:
            st.session_state[q_key] = existing.get(field, None) or None
        responses[field] = st.radio(
            question,
            FEEDBACK_OPTIONS,
            index=None,
            horizontal=True,
            key=q_key,
        )

    comment = st.text_area(
        "Comment (optional)",
        value=existing.get("comment", ""),
        height=100,
        key=f"comment_{fb_key}",
    )
    if st.button("Save Feedback", key=f"submit_{fb_key}"):
        missing = [question for field, question in FEEDBACK_QUESTIONS if not responses.get(field)]
        if missing:
            st.warning("Please answer all questions. Use N/A where needed.")
        else:
            st.session_state.evals[(pid, visit_num)] = {
                "pid": pid,
                "visit": visit_num,
                "cohort": cohort,
                "therapy_cls": therapy_cls,
                "comment": comment,
                **responses,
            }
            save_evals(reviewer, st.session_state.evals)
            st.success("Saved!")


# ── RIGHT: Multi-Agent vs Baseline ────────────────────────────────────────────

with right:
    view_mode = st.radio(
        "View", ["Multi-Agent", "Prompts"],
        horizontal=True, label_visibility="collapsed",
    )

    def _render_baseline_regimen(bl):
        opts = [bl.get(f"option_{n}") for n in [1, 2, 3] if bl.get(f"option_{n}")]
        if not opts:
            return
        with st.container(border=True):
            for i, opt in enumerate(opts):
                label = opt.get("label", f"Option {i+1}")
                drugs = opt.get("drugs", [])
                if isinstance(drugs, list):
                    drugs_str = " | ".join(f"`{d['drug']}` {d.get('action','')}" for d in drugs)
                else:
                    drugs_str = str(drugs)
                rationale = opt.get("rationale", "")
                st.markdown(f"**Option {i+1}: {label}**")
                st.markdown(drugs_str)
                if rationale:
                    st.caption(rationale)
                if i < len(opts) - 1:
                    st.divider()

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
                else:
                    drugs_str = str(drugs)
                rationale = opt.get("rationale", "")
                st.markdown(f"**Option {n}: {label}**")
                st.markdown(drugs_str)
                if rationale:
                    st.caption(rationale)
                if n < 3:
                    st.divider()

    if view_mode == "Prompts":
        st.markdown("### Agent Prompts")
        for agent_name, filename in PROMPT_FILES:
            path = os.path.join(PROMPTS_DIR, filename)
            with st.expander(
                scoped_expander_label(agent_name.replace("_", " ").title(), f"prompt_{agent_name}"),
                expanded=False,
            ):
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as f:
                        st.text(f.read())
                else:
                    st.warning(f"Prompt file not found: {filename}")

    elif view_mode == "Multi-Agent":
        if not v2_patient:
            st.warning("No pipeline output for this patient/visit.")
        else:
            st.markdown("### Phase 0 — Orchestrator")
            orch = v2_patient.get("orchestrator", {})
            decisions = orch.get("decisions", [])
            if decisions:
                with st.expander(
                    scoped_expander_label("Agent Activation Decisions", "orchestrator"),
                    expanded=False,
                ):
                    for d in decisions:
                        icon = "+" if d.get("activated") else "-"
                        st.markdown(f"**{icon} {d.get('agent','?')}** — {d.get('reason','')}")
            else:
                st.info("No orchestrator output")

            st.markdown("### Phase 1 — Specialist Agents")
            phase1 = v2_patient.get("phase1", {})
            if phase1:
                for agent_name, agent_text in phase1.items():
                    label = AGENT_LABELS.get(agent_name, agent_name.title())
                    with st.expander(
                        scoped_expander_label(label, f"phase1_{agent_name}"),
                        expanded=False,
                    ):
                        if isinstance(agent_text, str):
                            st.markdown(agent_text)
                        elif isinstance(agent_text, dict):
                            st.json(agent_text)
                        else:
                            st.text(str(agent_text))
            else:
                st.info("No Phase 1 outputs")

            st.markdown("### Phase 2")
            epi = v2_patient.get("epileptologist", {})
            if epi:
                with st.expander(scoped_expander_label("Epileptologist", "phase2_epi"), expanded=False):
                    if epi.get("reasoning"):
                        st.markdown(epi["reasoning"])
                    _render_regimen(epi.get("regimen", {}))
            else:
                st.info("No epileptologist output")

            st.markdown("### Adversarial Review")
            pharma = v2_patient.get("pharmacologist", "")
            if pharma:
                with st.expander(scoped_expander_label("Pharmacologist", "pharmacologist"), expanded=False):
                    if isinstance(pharma, str):
                        st.markdown(pharma)
                    else:
                        st.json(pharma)
            else:
                st.info("No pharmacologist output")

            debate = v2_patient.get("debate", [])
            if debate:
                st.markdown("### Epileptologist Prediction")
                last_idx = len(debate) - 1
                for idx, rnd in enumerate(debate):
                    with st.expander(
                        scoped_expander_label("Epileptologist Rebuttal", f"debate_{idx}"),
                        expanded=(idx == last_idx),
                    ):
                        if rnd.get("epileptologist"):
                            st.markdown(rnd["epileptologist"])
                        _render_regimen(rnd.get("epileptologist_regimen", {}))
