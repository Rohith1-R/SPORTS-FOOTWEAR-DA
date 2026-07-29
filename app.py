import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Sports Footwear Revenue Prediction",
    page_icon="👟",
    layout="centered"
)

# --------------------------------------------------
# LOAD MODEL AND SCALER
# --------------------------------------------------

model = joblib.load("sports_footwear_revenue_model.pkl")
scaler = joblib.load("sports_footwear_scaler.pkl")

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("👟 Sports Footwear Revenue Prediction")
st.write("Predict the expected revenue of a sports footwear product.")

st.divider()

# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------

brand = st.selectbox(
    "Brand",
    ["Nike", "Adidas", "Puma", "New Balance", "Under Armour"]
)

product_category = st.selectbox(
    "Product Category",
    ["Running Shoes", "Football Shoes", "Basketball Shoes",
     "Training Shoes", "Tennis Shoes", "Casual Sports Shoes"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Unisex"]
)

country = st.selectbox(
    "Country",
    ["India", "USA", "UK", "Germany", "France",
     "Spain", "Brazil", "Italy", "Japan", "Australia"]
)

sales_channel = st.selectbox(
    "Sales Channel",
    ["Online", "Retail Store"]
)

base_price = st.number_input(
    "Base Price (USD)",
    min_value=1.0,
    value=100.0,
    step=1.0
)

discount = st.number_input(
    "Discount Percentage",
    min_value=0.0,
    max_value=90.0,
    value=10.0,
    step=1.0
)

final_price = base_price * (1 - discount / 100)

st.info(f"Final Price: ${final_price:,.2f}")

units_sold = st.number_input(
    "Units Sold",
    min_value=1,
    value=10,
    step=1
)

customer_rating = st.slider(
    "Customer Rating",
    min_value=1.0,
    max_value=5.0,
    value=4.0,
    step=0.1
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔮 Predict Revenue", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame({
        "brand": [brand],
        "product_category": [product_category],
        "gender": [gender],
        "country": [country],
        "sales_channel": [sales_channel],
        "base_price_usd": [base_price],
        "discount_percent": [discount],
        "final_price_usd": [final_price],
        "units_sold": [units_sold],
        "customer_rating": [customer_rating]
    })

    try:

        # Convert categorical variables
        input_encoded = pd.get_dummies(input_data)

        # Match training columns
        input_encoded = input_encoded.reindex(
            columns=scaler.feature_names_in_,
            fill_value=0
        )

        # Scale input
        input_scaled = scaler.transform(input_encoded)

        # Prediction
        prediction = model.predict(input_scaled)[0]

        # Display result
        st.success(
            f"### Predicted Revenue: ${prediction:,.2f}"
        )

        st.metric(
            label="Expected Revenue",
            value=f"${prediction:,.2f}"
        )

    except Exception as e:

        st.error("Prediction failed.")

        st.write(
            "Make sure the preprocessing used in Streamlit "
            "matches the preprocessing used during model training."
        )

        st.exception(e)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Sports Footwear Sales Analytics | Machine Learning Revenue Prediction"
)

