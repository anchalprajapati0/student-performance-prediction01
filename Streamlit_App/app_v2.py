import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from xgboost import plot_importance
from datetime import datetime

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.caption(
    f"📅 Date: {datetime.now().strftime('%d-%m-%Y')} | "
    f"🕒 Time: {datetime.now().strftime('%I:%M %p')}"
)   
# ==========================
# Load Model Files
# ==========================

model = joblib.load("../Model/student_model.pkl")
scaler = joblib.load("../Model/scaler.pkl")
encoders = joblib.load("../Model/encoders.pkl")
st.markdown("""
<style>

/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

/* Main Background */
.stApp{
    background: linear-gradient(135deg,#eef4ff,#ffffff);
}

/* Main Container */
.main .block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1200px;
}

/* Header Card */
.header-card{
    background:linear-gradient(135deg,#2563eb,#4f46e5);
    color:white;
    padding:20px 25px;
    border-radius:18px;
    box-shadow:0 8px 20px rgba(0,0,0,.12);
}

.header-card h1{
    margin-bottom:6px;
    font-size:34px;
    font-weight:700;
}

.header-card p{
    font-size:15px;
    opacity:.9;
}

/* Glass Cards */
.glass-card{
    background:rgba(255,255,255,.85);
    backdrop-filter:blur(12px);
    border-radius:18px;
    padding:20px;
    border:1px solid rgba(255,255,255,.4);
    box-shadow:0 8px 20px rgba(0,0,0,.08);
    transition:.3s;
}

.glass-card:hover{
    transform:translateY(-5px);
    box-shadow:0 12px 30px rgba(37,99,235,.18);
}

/* Button */
.stButton>button{
    width:100%;
    background:linear-gradient(135deg,#2563eb,#4f46e5);
    color:white;
    border:none;
    border-radius:12px;
    height:55px;
    font-size:18px;
    font-weight:600;
    transition:.3s;
}

.stButton>button:hover{
    transform:scale(1.03);
    background:linear-gradient(135deg,#1d4ed8,#4338ca);
}

/* Metric Cards */
div[data-testid="stMetric"]{
    background:white;
    border-radius:15px;
    padding:15px;
    box-shadow:0 6px 18px rgba(0,0,0,.08);
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#1e293b;
}

section[data-testid="stSidebar"] *{
    color:white;
}

/* Animation */
@keyframes fadeIn{
    from{
        opacity:0;
        transform:translateY(20px);
    }
    to{
        opacity:1;
        transform:translateY(0);
    }
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="header-card">

<h1>🎓 Student Performance Prediction Dashboard</h1>

<p>Predict Student Academic Performance using Machine Learning & XGBoost</p>

</div>
""", unsafe_allow_html=True)
# ==========================
# Sidebar
# ==========================
# ==========================
# Sidebar
# ==========================

with st.sidebar:

    # Sidebar Image
    st.image(
        "https://img.icons8.com/color/96/graduation-cap.png",
        width=60
    )

    # Project Title
    st.markdown("""
    <h2 style='text-align:center; color:#2563EB;'>
    🎓 Student Performance Prediction
    </h2>

    <p style='text-align:center; font-size:15px;'>
    Predict a student's final grade using
    <b>Machine Learning (XGBoost)</b>
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    # Project Information
    st.subheader("📌 Project Information")

    st.write("**🤖 Model:** XGBoost Regressor")
    st.write("**📊 Features:** 32 Input Features")
    st.write("**🎯 Target:** Final Grade (G3)")
    st.write("**📂 Dataset:** Student Performance Dataset")
    st.write("**📈 Problem Type:** Regression")
    st.write("**⚡ Prediction:** AI-Based Regression")

    st.divider()

    # Technology Stack
    st.subheader("🛠️ Technology Stack")

    st.write("🐍 Python")
    st.write("🎈 Streamlit")
    st.write("🤖 XGBoost")
    st.write("📊 Scikit-Learn")
    st.write("🐼 Pandas")
    st.write("📉 NumPy")
    st.write("📈 Plotly")
    st.write("🎨 Matplotlib")

    st.divider()

    # Model Status
    st.success("✅ Model Loaded Successfully")

    st.divider()

    # Developer Information
    st.subheader("👩‍💻 Developer")

    st.write("**Name:** Anchal Prajapati")
    st.write("**College:** United College of Engineering and Research")
    st.write("**Branch:** B.Tech CSE")

    st.caption("© 2026 Student Performance Prediction")
st.title("📝 Student Details")

st.write("Enter the student information below to predict the final grade.")
left, right = st.columns(2)
with left:

    st.subheader("👤 Student Information")

    school = st.selectbox(
        "School",
        ["GP", "MS"]
    )

    sex = st.selectbox(
        "Gender",
        ["F", "M"]
    )

    age = st.slider(
        "Age",
        15,
        22,
        17
    )

    address = st.selectbox(
        "Address",
        ["U", "R"]
    )

    famsize = st.selectbox(
        "Family Size",
        ["LE3", "GT3"]
    )

    Pstatus = st.selectbox(
        "Parent Status",
        ["T", "A"]
    )

    Medu = st.slider(
        "Mother Education",
        0,
        4,
        2
    )

    Fedu = st.slider(
        "Father Education",
        0,
        4,
        2
    )
with right:

    st.subheader("📚 Study Information")

    traveltime = st.slider(
        "Travel Time",
        1,
        4,
        2
    )

    studytime = st.slider(
        "Study Time",
        1,
        4,
        2
    )

    failures = st.slider(
        "Past Class Failures",
        0,
        4,
        0
    )

    schoolsup = st.selectbox(
        "School Support",
        ["yes", "no"]
    )

    famsup = st.selectbox(
        "Family Support",
        ["yes", "no"]
    )

    paid = st.selectbox(
        "Extra Paid Classes",
        ["yes", "no"]
    )

    activities = st.selectbox(
        "Extra Activities",
        ["yes", "no"]
    )
family_col, personal_col = st.columns(2)

with family_col:

    st.subheader("🏠 Family Information")

    Mjob = st.selectbox(
        "Mother's Job",
        ["at_home", "health", "other", "services", "teacher"]
    )

    Fjob = st.selectbox(
        "Father's Job",
        ["at_home", "health", "other", "services", "teacher"]
    )

    reason = st.selectbox(
        "Reason for Choosing School",
        ["course", "home", "other", "reputation"]
    )

    guardian = st.selectbox(
        "Guardian",
        ["mother", "father", "other"]
    )

    nursery = st.selectbox(
        "Attended Nursery",
        ["yes", "no"]
    )

    higher = st.selectbox(
        "Wants Higher Education",
        ["yes", "no"]
    )

with personal_col:

    st.subheader("❤️ Personal & Lifestyle")

    internet = st.selectbox(
        "Internet Access",
        ["yes", "no"]
    )

    romantic = st.selectbox(
        "Romantic Relationship",
        ["yes", "no"]
    )

    famrel = st.slider(
        "Family Relationship",
        1,5,3
    )

    freetime = st.slider(
        "Free Time",
        1,5,3
    )

    goout = st.slider(
        "Going Out",
        1,5,3
    )

    Dalc = st.slider(
        "Workday Alcohol Consumption",
        1,5,1
    )

    Walc = st.slider(
        "Weekend Alcohol Consumption",
        1,5,2
    )
academic_col, grade_col = st.columns(2)

with academic_col:

    st.subheader("🩺 Health Information")

    health = st.slider(
        "Health Status",
        1,5,3
    )

    absences = st.slider(
        "Number of Absences",
        0,100,5
    )

with grade_col:

    st.subheader("📚 Previous Academic Performance")

    G1 = st.slider(
        "First Period Grade (G1)",
        0,20,10
    )

    G2 = st.slider(
        "Second Period Grade (G2)",
        0,20,10
    )
st.markdown("<br>", unsafe_allow_html=True)

predict_btn = st.button(
    "🎯 Predict Student Performance",
    use_container_width=True
)   
if predict_btn:

    input_data = {
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
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
    }

    input_df = pd.DataFrame([input_data])
    # Encode categorical values

    for col in encoders:
        input_df[col] = encoders[col].transform(input_df[col].astype(str))
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
# ==========================
# Prediction
# ==========================

prediction = model.predict(input_scaled)[0]
prediction = round(float(prediction), 2)

st.success(f"🎓 Predicted Final Grade (G3): {prediction}")

# ==========================
# Calculate Percentage
# ==========================

percentage = round((prediction / 20) * 100, 2)

# ==========================
# Prediction Summary
# ==========================


st.subheader("📋 Prediction Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎓 Predicted Grade", f"{prediction}/20")

with col2:
    st.metric("📊 Percentage", f"{percentage}%")

with col3:
    if prediction >= 10:
        st.metric("✅ Status", "PASS")
    else:
        st.metric("❌ Status", "FAIL")

# ==========================
# Download Prediction Report
# ==========================

report = pd.DataFrame({
    "Field": [
        "Predicted Grade",
        "Percentage",
        "Status"
    ],
    "Value": [
        prediction,
        f"{percentage}%",
        "PASS" if prediction >= 10 else "FAIL"
    ]
})

csv = report.to_csv(index=False)

st.download_button(
    label="📥 Download Prediction Report",
    data=csv,
    file_name="Student_Prediction_Report.csv",
    mime="text/csv"
)    




st.subheader("🎯 Performance Gauge")

fig = go.Figure(go.Indicator(

    mode="gauge+number",

    value=prediction,

    title={'text':"Predicted Final Grade"},

    gauge={

        'axis':{'range':[0,20]},

        'bar':{'color':'royalblue'},

        'steps':[

            {'range':[0,8],'color':'#ff4d4d'},

            {'range':[8,12],'color':'#ffa500'},

            {'range':[12,16],'color':'#ffd700'},

            {'range':[16,20],'color':'#00cc66'}

        ]

    }

))

fig.update_layout(
    width=450,
    height=320,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig, use_container_width=False)
st.markdown("---")
st.subheader("📊 Grade Comparison")

chart = pd.DataFrame({
    "Exam": ["G1", "G2", "Predicted G3"],
    "Marks": [G1, G2, prediction]
})

bar_fig = go.Figure(
    go.Bar(
        x=chart["Exam"],
        y=chart["Marks"],
        text=chart["Marks"],
        textposition="outside"
    )
)

bar_fig.update_layout(
    title="📊 Grade Comparison",
    width=450,
    height=300,
    margin=dict(l=20, r=20, t=50, b=20),
    xaxis_title="Exam",
    yaxis_title="Marks",
    yaxis=dict(range=[0, 20])
)

st.plotly_chart(bar_fig, use_container_width=False)
st.markdown("---")
st.subheader("📈 Top 10 Important Features")

fig2, ax = plt.subplots(figsize=(5,4))


plot_importance(
    model,
    ax=ax,
    max_num_features=10,
    height=0.5
)
st.markdown("---")
st.subheader("🏅 Performance Badge")

if prediction >= 16:
    st.success("🥇 Gold Badge - Excellent Student")
elif prediction >= 14:
    st.info("🥈 Silver Badge - Good Performer")
elif prediction >= 10:
    st.warning("🥉 Bronze Badge - Average Performer")
else:
    st.error("📘 Improvement Badge - Keep Learning")
st.pyplot(fig2)
st.markdown("---")
st.subheader("💡 Personalized Recommendations")

if prediction >= 16:

    st.success("""

🏆 Excellent Performance

✔ Continue your current study routine.

✔ Practice advanced questions.

✔ Help classmates.

✔ Prepare for competitive exams.

""")

elif prediction >= 14:

    st.info("""

😊 Good Performance

✔ Increase revision.

✔ Solve PYQs.

✔ Improve consistency.

""")

elif prediction >= 10:

    st.warning("""

🙂 Average Performance

✔ Increase study time.

✔ Reduce absences.

✔ Take regular mock tests.

""")

else:

    st.error("""

⚠ Needs Improvement

✔ Attend classes regularly.

✔ Focus on fundamentals.

✔ Practice daily.

✔ Seek guidance from teachers.

""")
st.snow()
