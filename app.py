
import streamlit as st
import pandas as pd
import joblib
import math

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================
# 2. LOAD ML MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ==========================================
# 3. PREMIUM CSS DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #f5f8ff,
        #eef4ff,
        #f8fbff
    );
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 1rem;
}

.header-box {
    background: linear-gradient(
        110deg,
        #ffffff,
        #e8f1ff
    );
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #dce7fa;
    margin-bottom: 20px;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #10234b;
    margin-bottom: 5px;
}

.gradient-text {
    background: linear-gradient(
        90deg,
        #2563eb,
        #6842ef
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    font-size: 17px;
    color: #64748b;
}

.feature-row {
    display: flex;
    gap: 25px;
    flex-wrap: wrap;
    margin-top: 20px;
    color: #334155;
    font-size: 14px;
    font-weight: 600;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.95);
    border: 1px solid #dce7fa;
    border-radius: 18px;
    padding: 15px;
    box-shadow: 0 5px 20px rgba(30,64,175,0.05);
}

.section-title {
    font-size: 24px;
    font-weight: 750;
    color: #14264b;
    margin-bottom: 4px;
}

.section-subtitle {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 20px;
}

label {
    color: #172554 !important;
    font-weight: 600 !important;
}

div[data-baseweb="input"] input,
div[data-baseweb="select"] > div {
    border-radius: 9px !important;
}

.result-box {
    background: linear-gradient(
        135deg,
        #e3f9ef,
        #f0fff7
    );
    border: 1px solid #b9efd2;
    padding: 30px 15px;
    border-radius: 16px;
    text-align: center;
    margin-top: 15px;
}

.result-label {
    font-size: 17px;
    color: #166534;
    font-weight: 600;
}

.result-price {
    font-size: 43px;
    font-weight: 850;
    color: #047857;
    margin-top: 10px;
}

.success-box {
    background: #dcfce7;
    padding: 13px;
    border-radius: 10px;
    color: #166534;
    font-weight: 600;
    text-align: center;
    margin-top: 15px;
}

.summary-title {
    font-size: 21px;
    font-weight: 750;
    color: #14264b;
    margin-top: 20px;
    margin-bottom: 12px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 20px;
    font-size: 13px;
}

.stFormSubmitButton > button {
    background: linear-gradient(
        90deg,
        #2563eb,
        #4f46e5
    );
    color: white !important;
    border: none;
    border-radius: 10px;
    height: 52px;
    font-weight: 700;
    font-size: 16px;
    width: 100%;
}

.stFormSubmitButton > button:hover {
    background: linear-gradient(
        90deg,
        #1d4ed8,
        #4338ca
    );
    color: white !important;
}

div[data-testid="stMetric"] {
    background: #f8faff;
    border: 1px solid #dce7fa;
    border-radius: 12px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. HEADER SECTION
# ==========================================

st.html("""
<div class="header-box">

    <div class="main-title">
        🏠 House Price
        <span class="gradient-text">Prediction</span>
    </div>

    <div class="subtitle">
        AI-powered system to estimate house prices
        based on key property features.
    </div>

    <div class="feature-row">
        <span>🧠 Machine Learning</span>
        <span>🎯 Smart Predictions</span>
        <span>⚡ Fast Results</span>
        <span>🏡 Property Valuation</span>
    </div>

</div>
""")


# ==========================================
# 5. SESSION STATE
# ==========================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "house_details" not in st.session_state:
    st.session_state.house_details = None


# ==========================================
# 6. DASHBOARD LAYOUT
# ==========================================

left_col, right_col = st.columns(
    [1.05, 0.95],
    gap="large"
)


# ==========================================
# 7. HOUSE INPUT SECTION
# ==========================================

with left_col:

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">'
            '🎚️ Enter House Details</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Provide property details to estimate its market price.'
            '</div>',
            unsafe_allow_html=True
        )

        with st.form("house_prediction_form"):

            col1, col2 = st.columns(2)

            with col1:

                area = st.number_input(
                    "Area (sqft)",
                    min_value=100,
                    max_value=10000,
                    value=1500,
                    step=100
                )

                bedrooms = st.number_input(
                    "Bedrooms",
                    min_value=1,
                    max_value=10,
                    value=3
                )

                bathrooms = st.number_input(
                    "Bathrooms",
                    min_value=1,
                    max_value=10,
                    value=2
                )

                age = st.number_input(
                    "House Age (years)",
                    min_value=0,
                    max_value=100,
                    value=5
                )

            with col2:

                location = st.selectbox(
                    "📍 Location",
                    [
                        "Coimbatore",
                        "Chennai",
                        "Madurai",
                        "Salem",
                        "Trichy"
                    ]
                )

                parking = st.selectbox(
                    "🚗 Parking",
                    ["Yes", "No"]
                )

                furnishing = st.selectbox(
                    "🛋️ Furnishing",
                    [
                        "Fully-Furnished",
                        "Semi-Furnished",
                        "Unfurnished"
                    ]
                )

            predict_button = st.form_submit_button(
                "🔍 Predict House Price",
                use_container_width=True
            )


# ==========================================
# 8. PREDICTION LOGIC
# ==========================================

if predict_button:

    new_house = pd.DataFrame([{
        "Area_sqft": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age_years": age,
        "Location": location,
        "Parking": parking,
        "Furnishing": furnishing
    }])

    try:

        predicted_price = float(
            model.predict(new_house)[0]
        )

        if not math.isfinite(predicted_price):

            st.session_state.prediction = None
            st.session_state.house_details = None

            st.error("Invalid prediction returned by model.")

        elif predicted_price < 0:

            st.session_state.prediction = None
            st.session_state.house_details = None

            st.error("The model returned a negative price.")

        else:

            st.session_state.prediction = predicted_price

            st.session_state.house_details = {
                "Area": area,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "Age": age,
                "Location": location,
                "Parking": parking,
                "Furnishing": furnishing
            }

    except Exception as e:

        st.session_state.prediction = None
        st.session_state.house_details = None

        st.error(f"Prediction failed: {e}")


# ==========================================
# 9. PREDICTION RESULT SECTION
# ==========================================

with right_col:

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">'
            '📊 Prediction Result</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Estimated price based on the provided house details.'
            '</div>',
            unsafe_allow_html=True
        )

        if st.session_state.prediction is not None:

            price = st.session_state.prediction
            details = st.session_state.house_details

            # Prediction Result Card

            st.html(f"""
            <div class="result-box">

                <div style="font-size:45px;">
                    🏡
                </div>

                <div class="result-label">
                    Estimated House Price
                </div>

                <div class="result-price">
                    ₹ {price:,.2f} Lakhs
                </div>

                <div class="success-box">
                    ✓ Prediction completed successfully!
                </div>

            </div>
            """)

            # House Details Summary

            st.markdown(
                '<div class="summary-title">'
                '📋 House Details Summary</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "Summary of the property details you entered."
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Area",
                f"{details['Area']:,} sqft"
            )

            c2.metric(
                "Bedrooms",
                details["Bedrooms"]
            )

            c3.metric(
                "Bathrooms",
                details["Bathrooms"]
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "House Age",
                f"{details['Age']} years"
            )

            c2.metric(
                "Location",
                details["Location"]
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Parking",
                details["Parking"]
            )

            c2.metric(
                "Furnishing",
                details["Furnishing"]
            )

        else:

            # Initial Placeholder

            st.html("""
            <div class="result-box">

                <div style="font-size:55px;">
                    🏠
                </div>

                <div class="result-label">
                    Your Prediction Will Appear Here
                </div>

                <p style="color:#64748b;">
                    Enter the house details and click
                    Predict House Price to see the result.
                </p>

            </div>
            """)

            st.markdown(
                '<div class="summary-title">'
                '✨ How It Works</div>',
                unsafe_allow_html=True
            )

            st.info(
                "1. Enter house details.\n\n"
                "2. Select property features.\n\n"
                "3. Click Predict House Price.\n\n"
                "4. View the estimated price."
            )


# ==========================================
# 10. FEATURE HIGHLIGHTS
# ==========================================

st.divider()

f1, f2, f3, f4 = st.columns(4)

with f1:
    st.markdown("### 🎯 AI Powered")
    st.caption("Machine Learning model")

with f2:
    st.markdown("### ⚡ Fast Results")
    st.caption("Prediction in seconds")

with f3:
    st.markdown("### 🏡 Easy to Use")
    st.caption("Simple property inputs")

with f4:
    st.markdown("### 🔒 Reliable")
    st.caption("Saved model integration")


# ==========================================
# 11. FOOTER
# ==========================================

st.html("""
<div class="footer">

    🏠 <b>House Price Prediction</b>
    &nbsp; | &nbsp;
    Machine Learning Project
    &nbsp; | &nbsp;
    Python & Streamlit

    <br><br>

    Built with ❤️ using Streamlit

</div>
""")
