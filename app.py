import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="E-Comm Return AI", page_icon="📦", layout="wide")

# ==========================================
# 2. LOAD AI ENGINE (WITH CACHING FOR SPEED)
# ==========================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('xgboost_returns_model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, model_columns
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure .pkl files are uploaded to GitHub.")
        st.stop()

xgb_model, model_columns = load_model()

# ==========================================
# 3. SIDEBAR: DATA ENTRY (OPERATIONS MANAGER)
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/679/679821.png", width=80)
st.sidebar.header("🛒 Order Parameters")

# Demographic Data
user_age = st.sidebar.slider("Customer Age", 18, 80, 35)
user_gender = st.sidebar.selectbox("Customer Gender", ["Female", "Male", "Other"])
user_location = st.sidebar.text_input("Customer Location (e.g., City5)", "City1")

# Product Data
product_category = st.sidebar.selectbox("Product Category", ["Clothing", "Toys", "Home Appliances", "Books", "Electronics"])
product_price = st.sidebar.number_input("Product Price (₹)", min_value=10.0, value=1500.0)
order_qty = st.sidebar.number_input("Order Quantity", min_value=1, max_value=20, value=1)
discount = st.sidebar.number_input("Discount Applied (₹)", min_value=0.0, value=50.0)

# Logistics Data
shipping = st.sidebar.selectbox("Shipping Method", ["Next-Day", "Express", "Standard"])
payment = st.sidebar.selectbox("Payment Method", ["Wallet", "COD", "Credit Card", "UPI"])
co2_emissions = st.sidebar.slider("Estimated CO2 Emissions (kg)", 0.0, 10.0, 1.5)
pkg_waste = st.sidebar.slider("Packaging Waste (kg)", 0.0, 5.0, 0.4)

# ==========================================
# 4. MAIN DASHBOARD UI
# ==========================================
st.title("📦 E-Commerce Product Return Predictor")
st.markdown("### Operations & Logistics Risk Assessment Dashboard")
st.markdown("---")

# Auto-Calculate Order Value based on user inputs
order_value = (product_price * order_qty) - discount

col1, col2, col3 = st.columns(3)
col1.metric("Total Order Value", f"₹{order_value:,.2f}")
col2.metric("Discount Rate", f"{(discount/order_value)*100 if order_value > 0 else 0:.1f}%")
col3.metric("Logistics Profile", f"{shipping} | {payment}")

st.markdown("---")

# ==========================================
# 5. PREDICTION ENGINE
# ==========================================
if st.button("🔮 Run AI Return Prediction", type="primary", use_container_width=True):
    
    # 1. Create Raw Data Dictionary
    input_data = {
        'Product_Category': product_category,
        'Product_Price': product_price,
        'Order_Quantity': order_qty,
        'Discount_Applied': discount,
        'Shipping_Method': shipping,
        'Payment_Method': payment,
        'User_Age': user_age,
        'User_Gender': user_gender,
        'User_Location': user_location,
        'Order_Value': order_value,
        'CO2_Emissions': co2_emissions,
        'Packaging_Waste': pkg_waste
    }
    
    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # 2. Apply the SAME Feature Engineering used in Colab
    df_input['Discount_Impact'] = df_input['Discount_Applied'] / (df_input['Order_Value'] + 1)
    df_input['Avg_Item_Value'] = df_input['Order_Value'] / df_input['Order_Quantity']
    df_input['Waste_to_CO2'] = df_input['Packaging_Waste'] / (df_input['CO2_Emissions'] + 0.1)
    
    # 3. Dummy Encoding
    categorical_cols = ['Product_Category', 'Shipping_Method', 'Payment_Method', 'User_Gender', 'User_Location']
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols)
    
    # 4. Match Skeleton (CRITICAL STEP)
    # This aligns our 1 row of input with the 120 columns the model expects. Missing columns get filled with 0.
    df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
    
   # 5. Make Prediction
    prediction = xgb_model.predict(df_final)[0]
    probabilities = xgb_model.predict_proba(df_final)[0]
    
    # SAFETY CLAMP: Force values to be strictly between 0.0 and 1.0 
    prob_return = float(max(0.0, min(1.0, probabilities[1])))
    prob_keep = float(max(0.0, min(1.0, probabilities[0])))
    
    # 6. Display Results
    st.markdown("### AI Assessment Results")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 1:
            st.error("🚨 HIGH RISK OF RETURN")
            st.write("The XGBoost model predicts this item will likely be returned.")
        else:
            st.success("✅ LOW RISK (Likely to Keep)")
            st.write("The XGBoost model predicts the customer will keep this item.")
            
    with res_col2:
        st.write("**Probability Breakdown:**")
        # Use the clamped floats directly
        st.progress(prob_return, text=f"Probability of Return: {prob_return * 100:.1f}%")
        st.progress(prob_keep, text=f"Probability of Keeping: {prob_keep * 100:.1f}%")
        
    st.info("💡 **Business Recommendation:** If the return probability exceeds 60% on high-value COD orders, consider enforcing prepaid shipping or routing the order to manual customer verification.")
