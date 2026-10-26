import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title="Campus Facility Reservation", page_icon="🏫", layout="wide")
st.title("🏫 Campus Facility Reservation and Monitoring")
st.caption("Decision-support dashboard • predictions are estimates, not observed outcomes")

@st.cache_data
def load_data(uploaded):
    if uploaded is not None:
        return pd.read_excel(uploaded) if uploaded.name.lower().endswith((".xlsx",".xls")) else pd.read_csv(uploaded)
    for name in ["facility_reservations.xlsx","facility_reservations.csv"]:
        if __import__("os").path.exists(name):
            return pd.read_excel(name) if name.endswith("xlsx") else pd.read_csv(name)
    return None

uploaded=st.sidebar.file_uploader("Upload facility reservations (CSV/XLSX)",type=["csv","xlsx","xls"])
df=load_data(uploaded)
if df is None:
    st.info("Upload facility_reservations.xlsx or CSV to begin.")
    st.stop()
df_original=df.copy()

# Harmonize dataset names from the uploaded midterm notebook and Excel workbook.
rename={}
for old,new in {"User_Type":"User_Category","Days_Before_Event":"Lead_Time_Days","Purpose":"Reservation_Purpose"}.items():
    if old in df.columns and new not in df.columns: rename[old]=new
df=df.rename(columns=rename)
if "Lead_Time_Days" not in df and {"Reservation_Date","Event_Date"}.issubset(df.columns):
    df["Lead_Time_Days"]=(pd.to_datetime(df["Event_Date"],errors="coerce")-pd.to_datetime(df["Reservation_Date"],errors="coerce")).dt.days
if "Reservation_Weekday" not in df and "Event_Date" in df:
    df["Reservation_Weekday"]=pd.to_datetime(df["Event_Date"],errors="coerce").dt.day_name()
if "Start_Hour" not in df and "Start_Time" in df:
    df["Start_Hour"]=pd.to_datetime(df["Start_Time"].astype(str),errors="coerce").dt.hour
if "Reservation_Purpose" not in df and "Purpose" in df: df["Reservation_Purpose"]=df["Purpose"]

# Derive outcome only from observed Usage_Status. Do not use outcome fields as predictors.
if "Reservation_Outcome" not in df and "Usage_Status" in df:
    df["Reservation_Outcome"]=df["Usage_Status"].astype(str).str.strip().str.lower().map(
        {"used":"Completed","cancelled":"Not Completed","canceled":"Not Completed","no-show":"Not Completed","no show":"Not Completed"})
if "Lead_Time_Category" not in df and "Lead_Time_Days" in df:
    def lead_cat(x):
        if pd.isna(x): return np.nan
        if x<=2:return "Last Minute"
        if x<=7:return "Short Lead"
        if x<=14:return "Medium Lead"
        return "Early Booking"
    df["Lead_Time_Category"]=df["Lead_Time_Days"].apply(lead_cat)

pages=["Overview","Data & Cleaning","Exploratory Analysis","Model & Evaluation","Predict a Reservation","Requirements & Workflow"]
page=st.sidebar.radio("Navigate",pages)
st.sidebar.caption(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

if page=="Overview":
    st.subheader("Observed reservation summary")
    status=df.get("Usage_Status",pd.Series(dtype=str)).astype(str).str.strip().str.lower()
    total=len(df)
    used=(status=="used").sum(); cancelled=status.isin(["cancelled","canceled"]).sum(); noshow=status.isin(["no-show","no show"]).sum()
    a,b,c,d=st.columns(4)
    a.metric("Observed total reservations",f"{total:,}")
    b.metric("Observed completion rate",f"{used/total:.1%}" if total else "N/A")
    c.metric("Observed cancellation rate",f"{cancelled/total:.1%}" if total else "N/A")
    d.metric("Observed no-show rate",f"{noshow/total:.1%}" if total else "N/A")
    st.warning("Observed rates use all loaded records as denominator. Missing/other Usage_Status values remain in the denominator.")
    if "Facility_Type" in df:
        st.subheader("Reservations by facility type")
        st.bar_chart(df["Facility_Type"].value_counts())
    if "User_Category" in df and "Usage_Status" in df:
        st.subheader("Cancellation / no-show percentage by user category")
        tmp=df.assign(_bad=df["Usage_Status"].astype(str).str.strip().str.lower().isin(["cancelled","canceled","no-show","no show"]))
        rates=tmp.groupby("User_Category",dropna=False)["_bad"].mean().sort_values(ascending=False)
        st.bar_chart(rates)
        st.caption("Denominator: all records within each user category.")
    st.subheader("Data preview")
    st.dataframe(df.head(10),use_container_width=True)

elif page=="Data & Cleaning":
    st.subheader("Dataset understanding")
    st.write("Original uploaded data is retained in memory as `df_original`; no values are fabricated.")
    st.write("Rows/columns:",df.shape)
    st.write("Data types")
    st.dataframe(df.dtypes.astype(str).rename("dtype"))
    st.write("Missing values")
    st.dataframe(df.isna().sum().rename("missing_count"))
    st.write("Exact duplicate rows:",int(df.duplicated().sum()))
    st.write("Summary statistics")
    st.dataframe(df.describe(include="all").T,use_container_width=True)
    cleaned=df.copy()
    cleaned=cleaned.drop_duplicates()
    for col in cleaned.select_dtypes(include=["object","string"]).columns:
        cleaned[col]=cleaned[col].astype("string").str.strip().replace({"":pd.NA})
    st.subheader("Cleaned preview (duplicate removal + trim text)")
    st.dataframe(cleaned.head(20),use_container_width=True)
    st.download_button("Download cleaned CSV",cleaned.to_csv(index=False).encode("utf-8"),"facility_reservations_cleaned.csv","text/csv")

elif page=="Exploratory Analysis":
    st.subheader("Exploratory analysis")
    if "Facility_Type" in df:
        st.markdown("**Reservation count by facility type**")
        st.bar_chart(df["Facility_Type"].value_counts())
    if {"User_Category","Usage_Status"}.issubset(df.columns):
        s=df["Usage_Status"].astype(str).str.strip().str.lower()
        bad=s.isin(["cancelled","canceled","no-show","no show"])
        rate=bad.groupby(df["User_Category"]).mean().sort_values(ascending=False)
        st.markdown("**Cancellation/no-show rate by user category**")
        st.bar_chart(rate)
        st.caption("Rate = cancellation or no-show records / all records in that category.")
    if {"Lead_Time_Category","Reservation_Outcome"}.issubset(df.columns):
        st.markdown("**Lead-time category vs reservation outcome**")
        st.dataframe(pd.crosstab(df["Lead_Time_Category"],df["Reservation_Outcome"],normalize="index").style.format("{:.1%}"))
        st.caption("Rows normalized: outcome percentages within each lead-time category.")

elif page=="Model & Evaluation":
    st.subheader("Decision Tree model")
    target="Reservation_Outcome"
    predictors=[c for c in ["Facility_Type","User_Category","Reservation_Purpose","Purpose","Lead_Time_Days","Participants","Facility_Capacity","Duration_Hours","Reservation_Weekday","Start_Hour","Days_Before_Event","After_Office_Hours","Approval_Status"] if c in df.columns]
    # Exclude leakage/post-outcome fields explicitly.
    forbidden={"Usage_Status","Actual_Attendance","Cancellation_Days_Before","Attendance_Ratio",target}
    predictors=[c for c in predictors if c not in forbidden]
    if target not in df:
        st.error("Target Reservation_Outcome cannot be derived: Usage_Status column is missing.")
    elif len(predictors)<1:
        st.error("No eligible booking-time predictors found.")
    else:
        modeldata=df[predictors+[target]].copy()
        modeldata=modeldata[modeldata[target].isin(["Completed","Not Completed"])]
        X=modeldata[predictors]; y=modeldata[target]
        if len(y)<5 or y.nunique()<2: st.warning("Insufficient labeled records or only one target class.")
        else:
            cats=X.select_dtypes(include=["object","string","category"]).columns.tolist()
            nums=[c for c in predictors if c not in cats]
            prep=ColumnTransformer([("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cats),
                                  ("num",SimpleImputer(strategy="median"),nums)])
            pipe=Pipeline([("prep",prep),("tree",DecisionTreeClassifier(max_depth=5,min_samples_leaf=5,class_weight="balanced",random_state=42))])
            strat=y if y.value_counts().min()>=2 else None
            Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=strat)
            pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Accuracy",f"{accuracy_score(yte,pred):.3f}")
            m2.metric("Precision (weighted)",f"{precision_score(yte,pred,average='weighted',zero_division=0):.3f}")
            m3.metric("Recall (weighted)",f"{recall_score(yte,pred,average='weighted',zero_division=0):.3f}")
            m4.metric("F1 (weighted)",f"{f1_score(yte,pred,average='weighted',zero_division=0):.3f}")
            st.write("Confusion matrix (actual rows, predicted columns)")
            st.dataframe(pd.DataFrame(confusion_matrix(yte,pred,labels=["Completed","Not Completed"]),index=["Actual Completed","Actual Not Completed"],columns=["Predicted Completed","Predicted Not Completed"]))
            st.text(classification_report(yte,pred,zero_division=0))
            st.write("Booking-time predictors:",predictors)
            st.caption("Preprocessing is fitted inside the training pipeline. Post-outcome fields are excluded.")
            st.session_state["trained_pipeline"]=pipe
            st.session_state["predictors"]=predictors

elif page=="Predict a Reservation":
    st.subheader("Sample reservation prediction")
    target="Reservation_Outcome"
    predictors=[c for c in ["Facility_Type","User_Category","Reservation_Purpose","Purpose","Lead_Time_Days","Participants","Facility_Capacity","Duration_Hours","Reservation_Weekday","Start_Hour","Days_Before_Event","After_Office_Hours","Approval_Status"] if c in df.columns]
    predictors=[c for c in predictors if c not in {"Usage_Status","Actual_Attendance","Cancellation_Days_Before","Attendance_Ratio",target}]
    if not predictors: st.error("No eligible predictors in dataset.")
    else:
        with st.form("predict_form"):
            values={}
            for col in predictors:
                series=df[col]
                if series.dtype=="object" or str(series.dtype).startswith("string") or str(series.dtype)=="category":
                    opts=sorted(series.dropna().astype(str).unique().tolist())
                    values[col]=st.selectbox(col,opts) if opts else st.text_input(col)
                else:
                    med=float(pd.to_numeric(series,errors="coerce").median()) if pd.to_numeric(series,errors="coerce").notna().any() else 0.0
                    values[col]=st.number_input(col,value=med)
            submitted=st.form_submit_button("Predict outcome")
        if submitted:
            if "trained_pipeline" not in st.session_state:
                st.warning("Go to Model & Evaluation first to train the pipeline.")
            else:
                row=pd.DataFrame([{c:values[c] for c in st.session_state["predictors"]}])
                pipe=st.session_state["trained_pipeline"]
                pred=pipe.predict(row)[0]
                st.metric("Predicted outcome",str(pred))
                if hasattr(pipe,"predict_proba"):
                    probs=pipe.predict_proba(row)[0]
                    st.write("Model class probabilities")
                    st.dataframe(pd.DataFrame({"class":pipe.classes_,"probability":probs}))
                st.warning("This is an estimate for administrator review, not an automatic approval or rejection.")

else:
    st.subheader("Requirements and decision-support workflow")
    st.markdown("""
**FR-DM-01** The system shall display observed reservation counts and completion, cancellation, and no-show rates.

**FR-DM-02** The system shall summarize cancellation/no-show rates by user category and reservation counts by facility type.

**FR-DM-03** The system shall generate a reservation outcome estimate from booking-time attributes and display it for administrator review.

**Workflow:** Reservation submitted → booking-time attributes validated → model estimates outcome → risk/estimate shown with limitations → administrator reviews → administrator makes decision → actual usage outcome recorded → dashboard aggregates observed results.

**Human oversight:** Predictions must not automatically reject reservations. Model performance may vary with missing data, class imbalance, historical patterns, and changing booking behavior.
""")
    st.code("""@startuml
actor "Reservation User" as User
actor "Facility Administrator" as Admin
rectangle "Campus Facility Reservation and Monitoring System" {
 usecase "Submit Reservation" as UC1
 usecase "Validate Booking Data" as UC2
 usecase "View Observed Dashboard" as UC3
 usecase "Estimate Reservation Outcome" as UC4
 usecase "Review Prediction" as UC5
 usecase "Approve / Manage Reservation" as UC6
 usecase "Record Actual Usage Outcome" as UC7
}
User --> UC1
UC1 --> UC2 : <<include>>
Admin --> UC3
Admin --> UC5
UC5 --> UC4 : <<include>>
Admin --> UC6
Admin --> UC7
@enduml""",language="text")
    st.markdown("Use the PlantUML text above to render the updated use-case diagram.")

st.sidebar.markdown("---")
st.sidebar.caption("Midterm notebook adapted to a multi-page Streamlit decision-support app.")
