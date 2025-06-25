# Save this as engine_health_app.py
!pip install scikit-learn
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

# Title
st.title("✈️ Aircraft Engine Health Visualizer")
st.write("A simple aerospace AI mini-project simulating Rolls-Royce style engine monitoring.")

# Function to simulate engine sensor data
def simulate_engine_data():
    temperature = np.random.normal(750, 5)  # Kelvin
    vibration = np.random.normal(0.3, 0.05) # G
    rpm = np.random.normal(5000, 50)        # RPM
    fuel_flow = np.random.normal(480, 10)   # kg/hr
    return [temperature, vibration, rpm, fuel_flow]

# Generate dummy training data
np.random.seed(42)
data_size = 500
X_train = np.random.normal(loc=[750, 0.3, 5000, 480], scale=[5, 0.05, 50, 10], size=(data_size, 4))
y_train = np.random.choice([0, 1, 2], size=data_size, p=[0.85, 0.1, 0.05])  # 0: Healthy, 1: Warning, 2: Critical

# Train Random Forest Classifier
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y_train)

# Placeholder for real-time chart
placeholder = st.empty()

# Simulate Real-Time Monitoring
if st.button("Start Engine Monitoring"):
    data_history = []

    for i in range(30):  # 30 cycles
        new_data = simulate_engine_data()
        data_history.append(new_data)
        
        scaled_data = scaler.transform([new_data])
        prediction = model.predict(scaled_data)[0]

        status_map = {0: 'Healthy ✅', 1: 'Warning ⚠️', 2: 'Critical ❌'}
        status = status_map[prediction]

        df = pd.DataFrame(data_history, columns=['Temperature (K)', 'Vibration (G)', 'RPM', 'Fuel Flow (kg/hr)'])

        with placeholder.container():
            st.subheader(f"Cycle {i+1}: Engine Status - {status}")
            st.line_chart(df)
            st.dataframe(df.tail(5))

        time.sleep(1)

else:
    st.write("🔵 Click the button to start engine simulation.")

