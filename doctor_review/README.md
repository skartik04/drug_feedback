# Doctor Review App

Standalone Streamlit bundle for the 70-patient doctor review set.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Contents

- `app.py`: Streamlit app
- `doctor_review_ids.json`: fixed 70-patient sample
- `data/`: CSV cohort data used to build CSV patient contexts
- `outputs/v2/`: CSV multi-agent predictions for visits 1-3
- `pdf_outputs/`: PDF split data and multi-agent predictions for visits 1-3
- `prompts/`: agent prompt files shown in the UI

Runtime files created by the app:

- `evaluations/`: saved reviewer feedback CSVs
- `review_selections.json`: keep/remove marks for the sample
