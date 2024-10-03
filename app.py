import streamlit as st
import pickle

# Custom CSS for background, buttons, text input, title, and animations
st.markdown(
    """
    <style>
    /* Background image */
    .stApp {
        background-image: url("https://www.fau.eu/files/2022/06/stress_colourbox50410411.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Gradient background for text input */
    .stTextInput > div > input {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: black;
        border-radius: 10px;
        padding: 10px;
    }

    /* Animated Predict button */
    div.stButton > button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        transition: transform 0.3s ease, background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #ff7373;
        transform: scale(1.05);
    }

    /* Center the content */
    .stApp > div:first-child {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }

    /* White Title with Shadow and Gradient Border */
    .stTitle {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3rem;
        color: #fff;
        text-align: center;
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.8);
        padding: 10px;
        border-radius: 8px;
        # background: linear-gradient(135deg, rgba(255,255,255,0.3) 0%, rgba(255,255,255,0.1) 100%);
        border: 2px solid transparent;
        # background-clip: padding-box, border-box;
        # box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
    }

    /* Font styles */
    .stMarkdown, .stTextInput, .stButton {
        font-family: 'Arial', sans-serif;
        color: #fff;
    }

    </style>
    """, unsafe_allow_html=True
)

# Load the model and vectorizer
with open('stress_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('cv_transform.pkl', 'rb') as cv_file:
    cv = pickle.load(cv_file)

# Title and Description with enhanced styling
st.markdown("<h1 class='stTitle'>🧠 Stress Detection App</h1>", unsafe_allow_html=True)
st.write("**This app predicts if a person is in stress condition based on their input text.**")

# Text input from the user with enhanced styling
user_input = st.text_input("Enter text to analyze:", "")

# Predict button with animation and interaction
if st.button("🔍 Predict"):
    if user_input:
        data = cv.transform([user_input]).toarray()
        output = model.predict(data)

        if output == 1:
            st.markdown("<h2 style='color: red;'>The person is in a stress condition.</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color: green;'>The person is in no stress condition.</h2>",
                        unsafe_allow_html=True)
    else:
        st.write("⚠️ Please enter some text for prediction.")

# To run: streamlit run app.py
