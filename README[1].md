# Campus Facility Reservation and Monitoring — Streamlit

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Upload `facility_reservations.xlsx` or a CSV in the sidebar. The app also looks for a local file with that name.

## Pages / modules
1. Overview dashboard
2. Data & Cleaning
3. Exploratory Analysis
4. Model & Evaluation
5. Predict a Reservation
6. Requirements & Workflow

## Deploy
Push this folder to GitHub. In Streamlit Community Cloud choose **New app**, select the repository, branch, and `app.py` as main file. Add packages from `requirements.txt`. Add your deployed URL to `DEPLOYMENT_REPORT.md`.

Predictions are decision support only; administrators make final decisions.
