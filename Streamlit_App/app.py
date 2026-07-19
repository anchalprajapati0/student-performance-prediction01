import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)
# ----------------------------------
# Custom CSS
# ----------------------------------

st.markdown("""
<style>

.main{
    background-color:#f8f9fa;
}

h1{
    color:#0d6efd;
    text-align:center;
}

h2,h3{
    color:#212529;
}

.stButton>button{
    width:100%;
    background-color:#0d6efd;
    color:white;
    font-size:18px;
    border-radius:10px;
    height:50px;
    border:none;
}

.stButton>button:hover{
    background-color:#084298;
    color:white;
}

div[data-testid="stMetric"]{
    background:#ffffff;
    border-radius:12px;
    padding:15px;
    box-shadow:0 0 10px rgba(0,0,0,0.1);
}

hr{
    border:1px solid #d9d9d9;
}

.footer{
    text-align:center;
    color:gray;
    font-size:16px;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)
# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("../Model/student_model.pkl")
scaler = joblib.load("../Model/scaler.pkl")
encoders = joblib.load("../Model/encoders.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Performance Prediction System")
st.markdown("### Predict a student's final performance using Machine Learning")

st.markdown("---")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 Project Information")

st.sidebar.info("""
### Student Performance Prediction

**Algorithm Used**
- XGBoost Regressor

**Features**
- Student Details
- Family Details
- Study Details
- Lifestyle Details

**Target**
- Final Grade (G3)

Developed using:
- Python
- Streamlit
- Scikit-Learn
- XGBoost
""")

st.sidebar.success("Model Loaded Successfully ✅")

# -----------------------------
# Student Information
# -----------------------------
st.header("👤 Student Information")

col1, col2, col3 = st.columns(3)

with col1:
    school = st.selectbox("School", ["GP", "MS"])

with col2:
    sex = st.selectbox("Gender", ["F", "M"])

with col3:
    age = st.slider("Age", 15, 22, 17)

col4, col5, col6 = st.columns(3)

with col4:
    address = st.selectbox("Address", ["U", "R"])

with col5:
    famsize = st.selectbox("Family Size", ["LE3", "GT3"])

with col6:
    pstatus = st.selectbox("Parent Status", ["T", "A"])
    # ----------------------------------
# Family Information
# ----------------------------------

st.markdown("---")
st.header("👨‍👩‍👧 Family Information")

col1, col2, col3 = st.columns(3)

with col1:
    Medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4])

with col2:
    Fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4])

with col3:
    guardian = st.selectbox("Guardian", ["mother", "father", "other"])

col4, col5 = st.columns(2)

with col4:
    Mjob = st.selectbox(
        "Mother's Job",
        ["at_home", "health", "services", "teacher", "other"]
    )

with col5:
    Fjob = st.selectbox(
        "Father's Job",
        ["at_home", "health", "services", "teacher", "other"]
    )

# ----------------------------------
# Study Information
# ----------------------------------

st.markdown("---")
st.header("📚 Study Information")

col1, col2, col3 = st.columns(3)

with col1:
    traveltime = st.selectbox("Travel Time", [1, 2, 3, 4])

with col2:
    studytime = st.selectbox("Study Time", [1, 2, 3, 4])

with col3:
    failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])

reason = st.selectbox(
    "Reason for Choosing School",
    ["course", "home", "reputation", "other"]
)
# ----------------------------------
# Student Support
# ----------------------------------

st.markdown("---")
st.header("🎓 Student Support")

col1, col2, col3 = st.columns(3)

with col1:
    schoolsup = st.selectbox("School Educational Support", ["yes", "no"])

with col2:
    famsup = st.selectbox("Family Educational Support", ["yes", "no"])

with col3:
    paid = st.selectbox("Extra Paid Classes", ["yes", "no"])

col4, col5 = st.columns(2)

with col4:
    activities = st.selectbox("Extra Curricular Activities", ["yes", "no"])

with col5:
    nursery = st.selectbox("Attended Nursery School", ["yes", "no"])

col6, col7 = st.columns(2)

with col6:
    higher = st.selectbox("Wants Higher Education", ["yes", "no"])

with col7:
    internet = st.selectbox("Internet Access at Home", ["yes", "no"])

romantic = st.selectbox("Romantic Relationship", ["yes", "no"])


# ----------------------------------
# Lifestyle Information
# ----------------------------------

st.markdown("---")
st.header("🌍 Lifestyle Information")

col1, col2, col3 = st.columns(3)

with col1:
    famrel = st.slider("Family Relationship", 1, 5, 3)

with col2:
    freetime = st.slider("Free Time", 1, 5, 3)

with col3:
    goout = st.slider("Going Out with Friends", 1, 5, 3)

col4, col5, col6 = st.columns(3)

with col4:
    Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1)

with col5:
    Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1)

with col6:
    health = st.slider("Health Status", 1, 5, 3)

absences = st.number_input(
    "Number of Absences",
    min_value=0,
    max_value=100,
    value=0
)
# ----------------------------------
# Previous Academic Performance
# ----------------------------------

st.markdown("---")
st.header("📊 Previous Academic Performance")

col1, col2 = st.columns(2)

with col1:
    G1 = st.number_input("First Period Grade (G1)", 0, 20, 10)

with col2:
    G2 = st.number_input("Second Period Grade (G2)", 0, 20, 10)

st.markdown("---")

# ----------------------------------
# Prediction
# ----------------------------------

if st.button("🎯 Predict Final Grade"):

    input_data = pd.DataFrame([{
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences,
        "G1": G1,
        "G2": G2
    }])

    # Encode categorical columns
    for col in encoders:
        input_data[col] = encoders[col].transform(input_data[col].astype(str))

    # Scale data
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction = round(float(prediction), 2)

    # -----------------------------
    # Gauge Chart
    # -----------------------------
    st.markdown("---")
    st.subheader("🎯 Predicted Grade Gauge")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={"text": "Predicted Final Grade (G3)"},
        gauge={
            "axis": {"range": [0, 20]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 10], "color": "#ff4d4d"},
                {"range": [10, 14], "color": "#ffd966"},
                {"range": [14, 17], "color": "#90ee90"},
                {"range": [17, 20], "color": "#00cc66"},
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.success(f"🎓 Predicted Final Grade (G3): {prediction}")

    # -----------------------------
    # Performance Level
    # -----------------------------
    if prediction >= 16:
        st.success("🏆 Performance Level: Excellent")
    elif prediction >= 14:
        st.info("😊 Performance Level: Good")
    elif prediction >= 10:
        st.warning("🙂 Performance Level: Average")
    else:
        st.error("⚠️ Performance Level: Needs Improvement")

    # -----------------------------
    # Progress Bar
    # -----------------------------
    progress = min(max(prediction / 20, 0), 1)
    st.progress(progress)

    st.metric("Predicted G3 Score", f"{prediction}/20")

    # -----------------------------
    # Recommendations
    # -----------------------------
    st.markdown("---")
    st.subheader("📘 Recommendations")

    if prediction >= 16:
        st.success("""
Excellent Performance!

✔ Keep practicing regularly.
✔ Continue solving advanced problems.
✔ Maintain consistency.
""")

    elif prediction >= 14:
        st.info("""
Good Performance!

✔ Increase study time.
✔ Revise weekly.
✔ Practice previous year papers.
""")

    elif prediction >= 10:
        st.warning("""
Average Performance

✔ Reduce absences.
✔ Improve study habits.
✔ Take help from teachers.
""")

    else:
        st.error("""
Needs Improvement

✔ Attend classes regularly.
✔ Focus on basics.
✔ Practice daily.
✔ Avoid distractions.
""")