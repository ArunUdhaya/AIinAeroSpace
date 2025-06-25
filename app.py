# Save this file as app.py
import streamlit as st # type: ignore

st.title("🚀 My First Streamlit App")
st.write("Hello, welcome to my world by Arun Udayasuriyan!")
st.write("Here's a simple slider:")

x = st.slider("Select a value", 0, 100, 25)
st.write("You selected:", x)