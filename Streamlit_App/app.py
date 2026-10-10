from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ----------------------------------
#
# ----------------------------------
# Custom CSS - Dark Black UI
# ----------------------------------

st.markdown("""
<style>
:root {
    --midnight: #111827;
    --panel: #1B2435;
    --panel-soft: #202B3F;
    --purple: #A78BFA;
    --purple-strong: #8B5CF6;
    --teal: #2DD4BF;
    --coral: #FB7185;
    --text: #F3F4F6;
    --muted: #B8C1D1;
    --border: #344155;
}

/* Main canvas */
.stApp {
    background: radial-gradient(circle at 80% 0%, #252044 0%, var(--midnight) 42%);
    color: var(--text);
}
.main .block-container {
    max-width: 1320px;
    padding-top: 1.8rem;
    padding-bottom: 2.5rem;
}

/* Typography */
h1, h2, h3, h4 {
    color: var(--text) !important;
    letter-spacing: -0.35px;
}
h1 { font-size: 2.25rem !important; font-weight: 800 !important; }
h2 { font-size: 1.65rem !important; font-weight: 750 !important; }
h3 { font-size: 1.2rem !important; font-weight: 700 !important; }
p, label, .stMarkdown, .stCaption, [data-testid="stWidgetLabel"] {
    color: var(--muted);
}
small { color: var(--muted) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #171D2D 0%, #131A28 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p {
    color: var(--text) !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--muted) !important;
}
section[data-testid="stSidebar"] hr { border-color: var(--border); }

/* Dashboard cards */
.hero-card {
    padding: 28px 32px;
    margin: 0 0 24px 0;
    border: 1px solid #51427B;
    border-radius: 22px;
    background: linear-gradient(120deg, #292344 0%, #1D2B40 62%, #183B42 100%);
    box-shadow: 0 12px 32px rgba(0,0,0,.16);
}
.hero-label {
    color: var(--teal);
    font-size: .76rem;
    font-weight: 800;
    letter-spacing: 1.7px;
    margin-bottom: 10px;
}
.hero-title {
    color: var(--text);
    font-size: clamp(1.65rem, 3vw, 2.35rem);
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 10px;
}
.hero-description {
    color: #D0D7E5;
    font-size: 1rem;
    line-height: 1.65;
    max-width: 850px;
}
.info-card, .section-card {
    height: 100%;
    padding: 22px;
    border: 1px solid var(--border);
    border-radius: 17px;
    background: linear-gradient(145deg, rgba(32,43,63,.98), rgba(25,34,50,.98));
    box-shadow: 0 8px 24px rgba(0,0,0,.12);
}
.info-label {
    color: #C4B5FD;
    font-size: .76rem;
    font-weight: 800;
    letter-spacing: 1.1px;
    margin-bottom: 10px;
}
.info-value {
    color: var(--text);
    font-size: 1.8rem;
    line-height: 1.25;
    font-weight: 800;
    margin-bottom: 7px;
}
.info-caption {
    color: var(--muted);
    font-size: .88rem;
    line-height: 1.5;
}
.section-card h3 { color: var(--teal) !important; }
.section-card p { color: #D0D7E5 !important; line-height: 1.65; }

/* Input widgets */
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
    background-color: #202B3F !important;
    color: var(--text) !important;
    border: 1px solid #45536A !important;
    border-radius: 10px !important;
}
.stSelectbox div[data-baseweb="select"] span { color: var(--text) !important; }
div[data-baseweb="popover"], div[data-baseweb="menu"],
div[role="option"] {
    background-color: #202B3F !important;
    color: var(--text) !important;
}
div[role="option"]:hover { background-color: #3A315B !important; }
.stNumberInput button {
    background-color: #303B50 !important;
    color: var(--text) !important;
    border-color: #45536A !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: var(--teal) !important;
    border-color: var(--teal) !important;
}
.stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] { background: var(--border); }

/* Form containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--border) !important;
    border-radius: 16px !important;
    background: rgba(27,36,53,.68);
    padding: 5px;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    min-height: 48px;
    border: 1px solid transparent;
    border-radius: 12px;
    color: #FFFFFF !important;
    background: linear-gradient(105deg, #8B5CF6 0%, #A78BFA 52%, #2DD4BF 130%);
    font-size: 1rem;
    font-weight: 750;
    transition: transform .18s ease, box-shadow .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #C4B5FD;
    box-shadow: 0 8px 22px rgba(139,92,246,.24);
    transform: translateY(-2px);
}

/* Streamlit metrics, alerts and progress */
div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
}
div[data-testid="stMetric"] label { color: var(--muted) !important; }
div[data-testid="stMetricValue"] { color: var(--text) !important; }
hr { border-color: var(--border) !important; }
div[data-testid="stAlert"] { border-radius: 12px; }
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--purple), var(--teal));
}
div[data-testid="stRadio"] label { color: var(--text) !important; }

/* Footer */
.footer {
    text-align: center;
    color: #A8B3C7;
    font-size: .88rem;
    padding: 26px 12px 10px;
    margin-top: 24px;
    border-top: 1px solid var(--border);
}
.footer strong { color: var(--purple); }

@media (max-width: 768px) {
    .main .block-container { padding: 1rem 1rem 2rem; }
    .hero-card { padding: 22px; }
    .info-card, .section-card { padding: 17px; }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD TRAINED MODEL
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "Model"

try:
    model = joblib.load(MODEL_DIR / "student_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    encoders = joblib.load(MODEL_DIR / "encoders.pkl")

except FileNotFoundError as error:
    st.error(f"Required model file was not found: {error}")
    st.stop()

except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


# =====================================================
# 4. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 12px 0 20px;">
            <div style="font-size: 34px;">🎓</div>
            <div style="font-size: 23px; font-weight: 800;
                        color: #A78BFA;">
                Student Performance Prediction
            </div>
            <div style="font-size: 12px; color: #858AA2;">
                STUDENT PERFORMANCE ANALYTICS
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🎯 Predict Performance",
            "ℹ️ About Project"
        ],
        label_visibility="visible"
    )

    st.markdown("---")

    st.markdown("### 🤖 ML Model")

    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Algorithm</div>
            <div class="info-value" style="font-size: 21px;">
                XGBoost
            </div>
            <div class="info-caption">Regression model</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.success("Model files loaded")

    st.caption("Built with Python, Streamlit and Machine Learning.")


# =====================================================
# 5. REUSABLE HERO SECTION
# =====================================================

def show_hero(title, description, label="STUDENT PERFORMANCE ANALYTICS"):

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-label">{label}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# 6. DASHBOARD PAGE
# =====================================================

if page == "🏠 Dashboard":

    show_hero(
        "Student Performance Dashboard 🎓",
        "Understand student performance and estimate final academic "
        "grades using a Machine Learning regression model."
    )

    st.markdown("## Welcome to Student Performance Prediction")

    st.write(
        "This application uses student information and academic "
        "features to estimate the final grade (G3)."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-label">Machine Learning</div>
                <div class="info-value">XGBoost</div>
                <div class="info-caption">Regression algorithm</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-label">Prediction Target</div>
                <div class="info-value">G3</div>
                <div class="info-caption">Final grade out of 20</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-label">Application</div>
                <div class="info-value">Interactive</div>
                <div class="info-caption">Student input and prediction</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## ✨ What can you do?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="section-card">
                <h3>🎯 Predict Performance</h3>
                <p>
                    Enter student, family, study and lifestyle
                    information to estimate the final grade.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="section-card">
                <h3>📊 Understand the Results</h3>
                <p>
                    View the predicted grade, performance category,
                    visual gauge and study recommendations.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "Select 'Predict Performance' from the sidebar to enter "
        "student details and generate a prediction."
    )


# =====================================================
# 7. PREDICTION PAGE
# =====================================================

elif page == "🎯 Predict Performance":

    show_hero(
        "Predict Student Performance",
        "Enter the student's details below. The trained model "
        "will estimate the final grade on a scale from 0 to 20.",
        label="AI-POWERED GRADE ESTIMATION"
    )

    # -------------------------------------------------
    # Student Information
    # -------------------------------------------------

    with st.container(border=True):

        st.markdown("### 👤 Student Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            school = st.selectbox("School", ["GP", "MS"])

        with col2:
            sex = st.selectbox("Gender", ["F", "M"])

        with col3:
            age = st.slider("Age", 15, 22, 17)

        col1, col2, col3 = st.columns(3)

        with col1:
            address = st.selectbox(
                "Home Address Type",
                ["U", "R"],
                format_func=lambda x: {
                    "U": "Urban",
                    "R": "Rural"
                }[x]
            )

        with col2:
            famsize = st.selectbox(
                "Family Size",
                ["LE3", "GT3"],
                format_func=lambda x: {
                    "LE3": "3 or fewer",
                    "GT3": "More than 3"
                }[x]
            )

        with col3:
            pstatus = st.selectbox(
                "Parent Status",
                ["T", "A"],
                format_func=lambda x: {
                    "T": "Living together",
                    "A": "Apart"
                }[x]
            )

    # -------------------------------------------------
    # Family Information
    # -------------------------------------------------

    with st.container(border=True):

        st.markdown("### 👨‍👩‍👧 Family Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            Medu = st.selectbox(
                "Mother's Education",
                [0, 1, 2, 3, 4],
                format_func=lambda x: {
                    0: "0 - None",
                    1: "1 - Primary education",
                    2: "2 - 5th to 9th grade",
                    3: "3 - Secondary education",
                    4: "4 - Higher education"
                }[x]
            )

        with col2:
            Fedu = st.selectbox(
                "Father's Education",
                [0, 1, 2, 3, 4],
                format_func=lambda x: {
                    0: "0 - None",
                    1: "1 - Primary education",
                    2: "2 - 5th to 9th grade",
                    3: "3 - Secondary education",
                    4: "4 - Higher education"
                }[x]
            )

        with col3:
            guardian = st.selectbox(
                "Guardian",
                ["mother", "father", "other"]
            )

        col1, col2 = st.columns(2)

        with col1:
            Mjob = st.selectbox(
                "Mother's Job",
                ["at_home", "health", "services", "teacher", "other"]
            )

        with col2:
            Fjob = st.selectbox(
                "Father's Job",
                ["at_home", "health", "services", "teacher", "other"]
            )

    # -------------------------------------------------
    # Study Information
    # -------------------------------------------------

    with st.container(border=True):

        st.markdown("### 📚 Study Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            traveltime = st.selectbox(
                "Travel Time",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1 - Less than 15 min",
                    2: "2 - 15 to 30 min",
                    3: "3 - 30 min to 1 hour",
                    4: "4 - More than 1 hour"
                }[x]
            )

        with col2:
            studytime = st.selectbox(
                "Weekly Study Time",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1 - Less than 2 hours",
                    2: "2 - 2 to 5 hours",
                    3: "3 - 5 to 10 hours",
                    4: "4 - More than 10 hours"
                }[x]
            )

        with col3:
            failures = st.selectbox(
                "Past Class Failures",
                [0, 1, 2, 3]
            )

        reason = st.selectbox(
            "Reason for Choosing School",
            ["course", "home", "reputation", "other"]
        )

    # -------------------------------------------------
    # Student Support
    # -------------------------------------------------

    with st.container(border=True):

        st.markdown("### 🎓 Student Support")

        col1, col2, col3 = st.columns(3)

        with col1:
            schoolsup = st.selectbox(
                "School Educational Support",
                ["yes", "no"]
            )

        with col2:
            famsup = st.selectbox(
                "Family Educational Support",
                ["yes", "no"]
            )

        with col3:
            paid = st.selectbox(
                "Extra Paid Classes",
                ["yes", "no"]
            )

        col1, col2 = st.columns(2)

        with col1:
            activities = st.selectbox(
                "Extra-Curricular Activities",
                ["yes", "no"]
            )

        with col2:
            nursery = st.selectbox(
                "Attended Nursery School",
                ["yes", "no"]
            )

        col1, col2 = st.columns(2)

        with col1:
            higher = st.selectbox(
                "Wants Higher Education",
                ["yes", "no"]
            )

        with col2:
            internet = st.selectbox(
                "Internet Access at Home",
                ["yes", "no"]
            )

        romantic = st.selectbox(
            "Romantic Relationship",
            ["yes", "no"]
        )

    # -------------------------------------------------
    # Lifestyle Information
    # -------------------------------------------------

    with st.container(border=True):

        st.markdown("### 🌿 Lifestyle Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            famrel = st.slider(
                "Family Relationship",
                1, 5, 3
            )

        with col2:
            freetime = st.slider(
                "Free Time",
                1, 5, 3
            )

        with col3:
            goout = st.slider(
                "Going Out with Friends",
                1, 5, 3
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            Dalc = st.slider(
                "Workday Alcohol Consumption",
                1, 5, 1
            )

        with col2:
            Walc = st.slider(
                "Weekend Alcohol Consumption",
                1, 5, 1
            )

        with col3:
            health = st.slider(
                "Self-Reported Health",
                1, 5, 3
            )

        absences = st.number_input(
            "Number of Absences",
            min_value=0,
            max_value=100,
            value=0
        )

    # -------------------------------------------------
    # Previous Academic Performance
    # -------------------------------------------------

    with st.container(border=True):

        st.markdown("### 📊 Previous Academic Performance")

        st.caption(
            "Enter previous period grades on the same 0–20 scale "
            "used by the dataset."
        )

        col1, col2 = st.columns(2)

        with col1:
            G1 = st.number_input(
                "First Period Grade (G1)",
                min_value=0,
                max_value=20,
                value=10
            )

        with col2:
            G2 = st.number_input(
                "Second Period Grade (G2)",
                min_value=0,
                max_value=20,
                value=10
            )

    # -------------------------------------------------
    # Prediction Button
    # -------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button(
        "✨ Predict Final Grade",
        use_container_width=True
    )

    if predict_clicked:

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

        try:
            # Validate expected features
            expected_features = list(scaler.feature_names_in_)

            if set(input_data.columns) != set(expected_features):
                st.error(
                    "Input features do not match the trained scaler. "
                    "Please check the training feature columns."
                )
                st.stop()

            # Match the exact feature order used during training
            input_data = input_data[expected_features]

            # Apply saved encoders
            for col in encoders:
                if col in input_data.columns:
                    input_data[col] = encoders[col].transform(
                        input_data[col].astype(str)
                    )

            # Apply saved scaler and predict
            input_scaled = scaler.transform(input_data)

            prediction = float(model.predict(input_scaled)[0])

            # G3 is normally in the 0–20 range
            prediction = max(0.0, min(20.0, prediction))
            prediction = round(prediction, 2)

        except Exception as error:
            st.error(
                "Prediction could not be generated. "
                "Please check the model's encoders and feature order."
            )
            st.exception(error)
            st.stop()

        # ---------------------------------------------
        # Prediction Results
        # ---------------------------------------------

        st.markdown("---")

        show_hero(
            "Your Prediction Results",
            "Here is the estimated final grade based on the information "
            "you provided.",
            label="PREDICTION COMPLETE"
        )

        if prediction >= 16:
            level = "Excellent"
            result_color = "#16865B"
            message = "Strong predicted performance. Keep up consistent study habits."

        elif prediction >= 14:
            level = "Good"
            result_color = "#38835C"
            message = "The prediction is positive. Regular revision may help."

        elif prediction >= 10:
            level = "Average"
            result_color = "#B7791F"
            message = "A structured study routine and practice may be helpful."

        else:
            level = "Needs Support"
            result_color = "#C24155"
            message = "Consider reviewing fundamentals and seeking academic support."

        # Result cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">PREDICTED FINAL GRADE</div>
                    <div class="info-value">{prediction}/20</div>
                    <div class="info-caption">Estimated G3 score</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">PERFORMANCE CATEGORY</div>
                    <div class="info-value"
                         style="font-size:22px;color:{result_color};">
                        {level}
                    </div>
                    <div class="info-caption">Based on application thresholds</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">GRADE SCALE</div>
                    <div class="info-value">
                        {round(prediction / 20 * 100)}%
                    </div>
                    <div class="info-caption">Of the 20-point grade scale</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge chart
        st.markdown("### 🎯 Predicted Grade")

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediction,
                number={
                    "suffix": " / 20",
                    "font": {
                        "size": 42,
                        "color": "#F3F4F6"
                    }
                },
                title={
                    "text": "Estimated Final Grade (G3)",
                    "font": {
                        "size": 18,
                        "color": "#B8C1D1"
                    }
                },
                gauge={
                    "axis": {
                        "range": [0, 20],
                        "tickwidth": 1,
                        "tickcolor": "#858AA2"
                    },
                    "bar": {
                        "color": "#A78BFA",
                        "thickness": 0.25
                    },
                    "bgcolor": "#1B2435",
                    "borderwidth": 1,
                    "bordercolor": "#344155",
                    "steps": [
                        {
                            "range": [0, 10],
                            "color": "#53303F"
                        },
                        {
                            "range": [10, 14],
                            "color": "#55442E"
                        },
                        {
                            "range": [14, 17],
                            "color": "#1C4B48"
                        },
                        {
                            "range": [17, 20],
                            "color": "#23665C"
                        }
                    ]
                }
            )
        )

        fig.update_layout(
            height=350,
            paper_bgcolor="#1B2435",
            plot_bgcolor="#1B2435",
            margin=dict(l=30, r=30, t=75, b=20),
            font=dict(
                family="Arial, sans-serif",
                color="#302D65"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        st.progress(
            min(max(prediction / 20, 0.0), 1.0),
            text=f"Predicted grade: {prediction} out of 20"
        )

        if prediction >= 16:
            st.success(f"🏆 {level}: {message}")

        elif prediction >= 14:
            st.success(f"😊 {level}: {message}")

        elif prediction >= 10:
            st.warning(f"📘 {level}: {message}")

        else:
            st.error(f"💡 {level}: {message}")

        # Recommendations
        st.markdown("### 📚 Suggested Next Steps")

        if prediction >= 16:
            recommendations = [
                "Maintain a consistent study routine.",
                "Practice challenging questions.",
                "Review previous topics regularly."
            ]

        elif prediction >= 14:
            recommendations = [
                "Revise important concepts every week.",
                "Practice previous examination questions.",
                "Identify topics where mistakes occur."
            ]

        elif prediction >= 10:
            recommendations = [
                "Create a realistic weekly study timetable.",
                "Practice core concepts regularly.",
                "Ask teachers for help with difficult topics."
            ]

        else:
            recommendations = [
                "Break difficult topics into smaller sections.",
                "Build a regular revision routine.",
                "Seek support from teachers or mentors.",
                "Track progress over time instead of relying on one score."
            ]

        for recommendation in recommendations:
            st.markdown(f"- {recommendation}")

        # Download result
        report = pd.DataFrame([{
            "Predicted Final Grade (G3)": prediction,
            "Performance Category": level,
            "Grade Scale": "0-20",
            "Model": "XGBoost Regressor"
        }])

        st.download_button(
            label="⬇️ Download Prediction Report (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="student_prediction_report.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "This is a model-based estimate, not a guaranteed exam result. "
            "The displayed performance categories are application-defined "
            "and should not be treated as official academic classifications."
        )


# =====================================================
# 8. ABOUT PROJECT PAGE
# =====================================================

elif page == "ℹ️ About Project":

    show_hero(
        "About Student Performance Prediction",
        "A Machine Learning project designed to estimate student "
        "academic performance from historical student attributes.",
        label="PROJECT INFORMATION"
    )

    st.markdown("## 🎯 Project Objective")

    st.write(
        "The project estimates a student's final grade (G3) using "
        "student information, family background, study habits, "
        "lifestyle attributes and previous academic grades."
    )

    st.markdown("## 🧠 Machine Learning Methodology")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="section-card">
                <h3>Supervised Learning</h3>
                <p>
                    The model learns from labelled historical data
                    containing input features and known final grades.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="section-card">
                <h3>Regression</h3>
                <p>
                    XGBoost Regressor estimates the numerical final
                    grade on the dataset's 0–20 scale.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## 🛠️ Technologies Used")

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas",
            "Scikit-learn",
            "XGBoost",
            "Plotly",
            "Streamlit",
            "Joblib"
        ],
        "Purpose": [
            "Core programming language",
            "Dataset handling",
            "Preprocessing and scaling",
            "Regression model",
            "Interactive visualizations",
            "Web application interface",
            "Saving and loading model files"
        ]
    })

    st.dataframe(
        technologies,
        hide_index=True,
        use_container_width=True
    )

    st.markdown("## 📌 Important Limitations")

    st.info(
        "Predictions depend on the dataset, training procedure and "
        "quality of the provided inputs. A prediction is an estimate, "
        "not a guarantee of future academic performance."
    )


# =====================================================
# 9. FOOTER
# =====================================================

st.markdown(
    """
    <div class="footer">
        <strong>Student Performance Prediction</strong><br>
        Student Performance Prediction System<br>
        Built with Python · Streamlit · XGBoost
    </div>
    """,
    unsafe_allow_html=True
)