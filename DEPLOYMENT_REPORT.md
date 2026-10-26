# Deployment Report (fill after deployment)

## Project
Campus Facility Reservation and Monitoring System — Streamlit decision-support app.

## Reused code
Adapted the uploaded midterm notebook's dataset understanding, duplicate/category cleaning, derived lead-time category, EDA, Decision Tree workflow, and requirements/use-case discussion. The original notebook is retained as `midterm_original.ipynb`.

## Changes
- Converted notebook cells into a navigable Streamlit application.
- Supports CSV/XLSX upload and workbook's actual field names (`User_Type`, `Days_Before_Event`, `Purpose`) with mapping to the app's modeling names.
- Added observed dashboard metrics, EDA, cleaning preview/download, model metrics, sample prediction form, and workflow/use-case page.
- Uses preprocessing pipeline (imputation + one-hot encoding) fit only on training split.
- Excludes `Usage_Status`, `Actual_Attendance`, `Cancellation_Days_Before`, `Attendance_Ratio`, and target from predictors.

## Tests to record
- [ ] App starts locally with `streamlit run app.py`
- [ ] Upload XLSX and CSV both work
- [ ] Overview rates and charts render
- [ ] Cleaned CSV downloads
- [ ] Model evaluation renders metrics/confusion matrix
- [ ] Sample prediction returns class/probabilities
- [ ] Empty/missing target or insufficient classes show helpful message
- [ ] Deployed Streamlit URL opens

## Deployment
GitHub repository URL: **[paste repository URL]**
Deployed Streamlit URL: **[paste Streamlit URL]**
Screenshots: **[attach screenshots of five modules and five sample predictions]**

## Limitations
Predictions are estimates, not causal explanations or observed outcomes. Data quality, missing values, class imbalance, limited historical records, and distribution changes may affect performance. Administrator review is required; do not automatically reject reservations.

## Reflection
[Briefly describe what you learned, issues encountered, how the app improves access to the analysis, and what you would improve next.]
