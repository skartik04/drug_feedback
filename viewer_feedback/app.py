import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import json
import pandas as pd
import streamlit as st
from predict_drugs import build_extracted_content
from patients import REVIEW_PATIENTS, CATEGORY_LABELS

# ── paths ─────────────────────────────────────────────────────────────────────

SCRIPTS  = os.path.join(os.path.dirname(__file__), '..', 'scripts')
DATA     = os.path.join(os.path.dirname(__file__), '..', 'data')
EVAL_DIR = os.path.join(os.path.dirname(__file__), 'evaluations')
os.makedirs(EVAL_DIR, exist_ok=True)

RECON_JSON = os.path.join(SCRIPTS, 'csv_reconciled_gpt_oss.json')
CSV_PATH   = os.path.join(DATA,    'combined_dataset.csv')

OUTPUT_JSONS = {
    1: os.path.join(SCRIPTS, 'drug/all_drugs/openai_gptoss120b_v1_ext_options.json'),
    2: os.path.join(SCRIPTS, 'drug/all_drugs/openai_gptoss120b_v2_ext_options.json'),
    3: os.path.join(SCRIPTS, 'drug/all_drugs/openai_gptoss120b_v3_ext_options.json'),
}

DR_COLS = {
    1: 'Current dose',
    2: 'Medication dosage and if there was a change in medication(6 months)',
    3: 'Medication dosage and if there was a change in medication(12 months)',
}

# ── eval helpers ──────────────────────────────────────────────────────────────

def load_evals(reviewer):
    path = os.path.join(EVAL_DIR, f'feedback_{reviewer}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str).fillna('')
        return {(r['pid'], int(r['visit'])): r.to_dict() for _, r in df.iterrows()}
    return {}

def save_evals(reviewer, evals):
    path = os.path.join(EVAL_DIR, f'feedback_{reviewer}.csv')
    pd.DataFrame(list(evals.values())).to_csv(path, index=False)

# ── load data ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_data():
    with open(RECON_JSON) as f:
        recon = json.load(f)
    outputs = {}
    for v, path in OUTPUT_JSONS.items():
        outputs[v] = json.load(open(path)) if os.path.exists(path) else {}
    df = pd.read_csv(CSV_PATH, sep=';', engine='python', quotechar='"', doublequote=True, escapechar='\\')
    df = df.drop_duplicates(subset=['Record ID'])
    df['_rid'] = pd.to_numeric(df['Record ID'], errors='coerce')
    csv_lookup = df.set_index('_rid').to_dict('index')
    return recon, outputs, csv_lookup

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Drug Pred · Feedback Review", layout="wide")
st.title("Drug Prediction — Feedback Review")

recon, outputs, csv_lookup = load_data()

PIDS    = [p["pid"] for p in REVIEW_PATIENTS]
CAT_MAP = {p["pid"]: p["category"] for p in REVIEW_PATIENTS}
TOTAL   = len(PIDS) * 3  # 20 patients × 3 visits = 60

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Reviewer")
    reviewer = st.text_input("Enter your name", value="")

    if not reviewer.strip():
        st.warning("Enter your name above to begin.")
        st.stop()

    reviewer = reviewer.strip()

    # reload evals when reviewer changes + auto-navigate to last reviewed patient
    if st.session_state.get('reviewer') != reviewer:
        st.session_state.reviewer = reviewer
        st.session_state.evals    = load_evals(reviewer)
        if st.session_state.evals:
            last = list(st.session_state.evals.values())[-1]
            last_pid = last.get('pid', '')
            if last_pid in PIDS:
                st.session_state.pat_idx = PIDS.index(last_pid)

    evals    = st.session_state.evals
    reviewed = len(evals)
    st.caption(f"Progress: {reviewed} / {TOTAL} reviewed")

    col_resume, col_fresh = st.columns(2)
    if col_resume.button("Resume", use_container_width=True):
        st.session_state.evals = load_evals(reviewer)
        if st.session_state.evals:
            last = list(st.session_state.evals.values())[-1]
            last_pid = last.get('pid', '')
            if last_pid in PIDS:
                st.session_state.pat_idx = PIDS.index(last_pid)
        st.rerun()
    if col_fresh.button("Start Fresh", use_container_width=True):
        st.session_state.evals = {}
        st.session_state.pat_idx = 0
        save_evals(reviewer, {})
        st.rerun()

    if evals:
        csv_bytes = pd.DataFrame(list(evals.values())).to_csv(index=False).encode()
        st.download_button("⬇ Download my responses", data=csv_bytes,
                           file_name=f"feedback_{reviewer}.csv", mime="text/csv")

    st.divider()
    st.header("Navigation")

    visit_num = st.radio("Visit", [1, 2, 3], format_func=lambda v: f"Visit {v}", horizontal=True)

    if 'pat_idx' not in st.session_state:
        st.session_state.pat_idx = 0
    st.session_state.pat_idx = min(st.session_state.pat_idx, len(PIDS) - 1)

    col_prev, col_next = st.columns(2)
    if col_prev.button("◀ Prev"):
        st.session_state.pat_idx = max(0, st.session_state.pat_idx - 1)
        st.session_state.pat_select = PIDS[st.session_state.pat_idx]
        st.rerun()
    if col_next.button("Next ▶"):
        st.session_state.pat_idx = min(len(PIDS) - 1, st.session_state.pat_idx + 1)
        st.session_state.pat_select = PIDS[st.session_state.pat_idx]
        st.rerun()

    selected_pid = st.selectbox(
        "Patient",
        PIDS,
        index=st.session_state.pat_idx,
        key="pat_select",
    )
    st.session_state.pat_idx = PIDS.index(selected_pid)
    st.caption(f"{st.session_state.pat_idx + 1} / {len(PIDS)} patients")

# ── main ──────────────────────────────────────────────────────────────────────

pid        = selected_pid
visit_name = f"Visit_{visit_num}"
visits     = recon.get(pid, {})
out        = outputs[visit_num].get(pid, {})

visit_feats = visits.get(visit_name, {})
model_input = (
    build_extracted_content(pid, visit_name, visit_feats, visits)
    if visit_feats else "(no extracted features for this visit)"
)

rid = int(pid.split("_")[0]) if pid.split("_")[0].isdigit() else None
dr_col = DR_COLS[visit_num]
doctor_output = "(not found in CSV)"
if rid and rid in csv_lookup:
    raw = csv_lookup[rid].get(dr_col, "")
    doctor_output = str(raw) if pd.notna(raw) else "(no entry)"

st.subheader(f"{pid}  —  Visit {visit_num}")
st.divider()

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### Model Input")
    st.code(model_input, language=None)

    st.markdown("### Doctor's Actual Output")
    st.info(doctor_output)

    if rid and rid in csv_lookup:
        row = csv_lookup[rid]
        if visit_num == 1:
            for col, label in [
                ('History of Presenting Illness', 'History of Presenting Illness'),
            ]:
                val = row.get(col, '')
                if pd.notna(val) and val:
                    st.markdown(f"**{label}**")
                    st.write(val)
        elif visit_num == 2:
            for col, label in [
                ('Second Entry(6 months)',    'Second Entry (6 months)'),
                ('Medication dosage and if there was a change in medication(6 months)', 'Medication Dosage (6 months)'),
            ]:
                val = row.get(col, '')
                if pd.notna(val) and val:
                    st.markdown(f"**{label}**")
                    st.write(val)
        elif visit_num == 3:
            for col, label in [
                ('Third Entry(12 months)',    'Third Entry (12 months)'),
                ('Medication dosage and if there was a change in medication(12 months)', 'Medication Dosage (12 months)'),
            ]:
                val = row.get(col, '')
                if pd.notna(val) and val:
                    st.markdown(f"**{label}**")
                    st.write(val)

    # ── Feedback ──────────────────────────────────────────────────────────────
    existing = st.session_state.evals.get((pid, visit_num), {})

    st.markdown("---")
    st.markdown("### Feedback")

    judgment_options = ["Yes", "No"]
    default_idx = judgment_options.index(existing["judgment"]) if existing.get("judgment") in judgment_options else None
    fb_key   = f"{pid}__v{visit_num}__{reviewer}"
    st.markdown("Do you agree with the model reasoning for the drug prediction task?")
    judgment = st.radio("", judgment_options, index=default_idx, horizontal=True, key=f"judgment_{fb_key}")
    comment  = st.text_area("Comment (optional)", value=existing.get("comment", ""), height=100, key=f"comment_{fb_key}")

    if st.button("Save Comment", key=f"submit_{fb_key}"):
        if judgment is None:
            st.warning("Please select Yes or No first.")
        else:
            st.session_state.evals[(pid, visit_num)] = {
                "pid": pid, "visit": visit_num, "judgment": judgment, "comment": comment
            }
            save_evals(reviewer, st.session_state.evals)
            st.success("Saved!")

    # auto-save judgment on selection
    if judgment is not None:
        current = st.session_state.evals.get((pid, visit_num), {})
        if current.get("judgment") != judgment:
            st.session_state.evals[(pid, visit_num)] = {
                "pid": pid, "visit": visit_num,
                "judgment": judgment,
                "comment": current.get("comment", ""),
            }
            save_evals(reviewer, st.session_state.evals)

with right:
    think = out.get('think', '')
    with st.expander("🧠 Model Thinking", expanded=False):
        st.text(think if think else "(empty)")

    st.markdown("### Section 1 — Clinical Reasoning")
    reasoning = out.get('reasoning', '(not found)')
    st.write(reasoning)

    st.markdown("### Section 2 — Regimen Options")
    with st.container(border=True):
        if 'option_1' in out:
            for n in [1, 2, 3]:
                opt = out.get(f'option_{n}', {})
                if not opt:
                    continue
                drugs_str = " · ".join(f"`{d['drug']}` {d['action']}" for d in opt.get('drugs', []))
                st.markdown(f"**Option {n}: {opt.get('label', '')}**  \n{drugs_str or '_no drugs parsed_'}")
                st.markdown(opt.get('rationale', ''))
                if n < 3:
                    st.divider()
        elif 'rank_1' in out:
            for n in [1, 2, 3]:
                drugs  = out.get(f'rank_{n}', [])
                reason = out.get(f'rank_{n}_reason', '')
                if not drugs and not reason:
                    continue
                drugs_str = " · ".join(f"`{d}`" for d in drugs) if isinstance(drugs, list) else str(drugs)
                st.markdown(f"**Option {n}:** {drugs_str}")
                st.markdown(reason)
                if n < 3:
                    st.divider()
        else:
            st.warning("No options found in output JSON.")
