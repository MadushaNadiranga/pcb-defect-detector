import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
import subprocess
import requests

FIREBASE_API_KEY = "AIzaSyA0i2j4DDVEI7eyrbqtzKI0XQB2I-pVoMk" 
def verify_password(email, password): 
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}" 
    payload = { 
    "email": email, 
    "password": password, 
    "returnSecureToken": True 
    } 
    response = requests.post(url, json=payload) 
    return response.json()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="PCB Defect Detection System",
    page_icon="img/pcb_icon.png",
    layout="wide"
)

# -----------------------------
# LOAD CSS
# -----------------------------
def load_css():
    with open("main_style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -----------------------------
# TWO COLUMNS LAYOUT
# -----------------------------
col1, col2 = st.columns([1, 1])  # Adjust ratios as needed

# Left Column: Image
with col1:
    st.image(
            "img/pcb_img.png",
            use_container_width=True
    )
# Right Column: Login/Register Form
with col2:
    st.markdown('<h1 class="main-header">PCB Defect Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Secure Access for Quality Control Professionals</p>', unsafe_allow_html=True)

    choice = st.radio("Select Action", ["Login", "Register"], horizontal=True)

    email = st.text_input("📧 Email Address", placeholder="your.email@company.com")
    password = st.text_input("🔒 Password", type="password", placeholder="Enter your secure password")

    if choice == "Register":
        if st.button("Create Account"):
            if not email or not password:
                st.error("⚠️ Please fill in all fields")
            else:
                try:
                    user = auth.create_user(email=email, password=password)
                    st.success("✅ Account created successfully! Please login.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    if choice == "Login":
        if st.button("Access System"):
            if not email or not password:
                st.error("⚠️ Please fill in all fields")
            else:
                result = verify_password(email, password)
                if "idToken" in result:
                    st.success("✅ Login successful!")
                    subprocess.run(["streamlit", "run", "app.py"])
                    st.stop()
                else:
                    st.error(f"❌ Login failed: {result.get('error', {}).get('message', 'Invalid credentials')}")

    st.markdown('<div class="card-footer">Protected by advanced encryption • ISO 27001 compliant</div>', unsafe_allow_html=True)
