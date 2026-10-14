---
title: Student Burnout Risk Predictor
emoji: 🎓
sdk: gradio
app_file: app.py
python_version: 3.11
---

# Student Burnout Risk Predictor

Run locally with `python app.py`. Install dependencies first with
`pip install -r requirements.txt`.

The **Upload data & insights** tab accepts CSV and Excel (`.xlsx`, `.xls`,
`.xlsm`) student-impact datasets. Required columns are `Major_Category`,
`Year_of_Study`, `Pre_Semester_GPA`, `Post_Semester_GPA`,
`Weekly_GenAI_Hours`, `Primary_Use_Case`, `Prompt_Engineering_Skill`,
`Tool_Diversity`, `Paid_Subscription`, `Traditional_Study_Hours`,
`Perceived_AI_Dependency`, `Institutional_Policy`,
`Anxiety_Level_During_Exams`, `Skill_Retention_Score`, and
`Burnout_Risk_Level`. `Student_ID` is optional; it is removed before records
are stored. Duplicate and incomplete rows are removed, and target labels and
numeric ranges are validated.

Successful imports are written to SQLite in `student_burnout.sqlite3` by
default. Set `BURNOUT_DATABASE_PATH` to choose another database file. When
deploying to Hugging Face Spaces, attach persistent storage mounted at `/data`
and set `BURNOUT_DATABASE_PATH=/data/student_burnout.sqlite3`; otherwise, data
written to the Space's local filesystem may not persist across restarts.

To deploy, create a Gradio Space and upload `app.py`, `app1.py`,
`requirements.txt`, the trained `burnout_hybrid_model_b.joblib`, and the
`ai_student_impact_dataset.csv` used to train that model. Do not include
identifiable student data in public uploads. The Space must have the dataset
and model files available at startup because the predictor loads them when the
app starts.
