import streamlit as st
import pandas as pd
from housing_utils import apply_clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import numpy as np
import requests

# --- LOAD DATA ---
df = pd.read_csv("data/AI_Housing_Lisbon_NOVA.csv")

# --- CLUSTERING STEP ---
features = ['monthly_rent', 'commute_to_NOVA', 'estimated_grocery_cost', 'size_sqm', 'bedrooms']
df = apply_clustering(df, features)

# Assign human-readable cluster labels
cluster_labels = {
    0: "Moderate Budget",
    1: "High-Cost, Low Convenience",
    2: "Student Friendly"
}
df['cluster_label'] = df['cluster'].map(cluster_labels)

# --- ADD FAKE COORDINATES FOR MAPPING ---
np.random.seed(42)
df['latitude'] = 38.72 + np.random.normal(0, 0.02, len(df))
df['longitude'] = -9.14 + np.random.normal(0, 0.02, len(df))

# --- SIDEBAR FILTERS ---
st.sidebar.title("🎛️ Filter Preferences")
max_rent = st.sidebar.slider("Maximum Rent (€)", 500, 1500, 1000)
max_commute = st.sidebar.slider("Maximum Commute to NOVA (min)", 10, 60, 30)
min_beds = st.sidebar.selectbox("Minimum Bedrooms", [1, 2, 3])
cluster_type = st.sidebar.selectbox("Preferred Cluster", ["Any"] + list(cluster_labels.values()))

# --- FILTER DATA ---
filtered_df = df[
    (df['monthly_rent'] <= max_rent) &
    (df['commute_to_NOVA'] <= max_commute) &
    (df['bedrooms'] >= min_beds)
]
if cluster_type != "Any":
    filtered_df = filtered_df[filtered_df['cluster_label'] == cluster_type]
filtered_df = filtered_df.dropna(subset=['latitude', 'longitude'])

# --- MAIN TITLE & TABLE ---
st.title("🏡 AI Housing Assistant for NOVA Students")
st.write(f"🎯 **{len(filtered_df)} listings found** matching your criteria:")
st.dataframe(filtered_df[['neighborhood', 'monthly_rent', 'commute_to_NOVA', 'bedrooms', 'cluster_label']])

# --- MAP VISUALIZATION ---
st.subheader("📍 Apartment Locations")
m = folium.Map(location=[38.72, -9.14], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)

color_map = {
    'Student Friendly': 'green',
    'Moderate Budget': 'blue',
    'High-Cost, Low Convenience': 'red'
}

for _, row in filtered_df.iterrows():
    popup_info = f"""
    <b>{row['neighborhood']}</b><br>
    €{row['monthly_rent']} / {row['bedrooms']} bed<br>
    Commute: {row['commute_to_NOVA']} min<br>
    Cluster: {row['cluster_label']}
    """
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=popup_info,
        icon=folium.Icon(color=color_map.get(row['cluster_label'], 'gray'))
    ).add_to(marker_cluster)

st_data = st_folium(m, width=700, height=500)

# --- AI CHAT ASSISTANT (OLLAMA) ---
st.subheader("💬 Ask the AI Housing Assistant")

if "ollama_chat" not in st.session_state:
    st.session_state.ollama_chat = []

user_prompt = st.chat_input("Ask something like: 'Find me a 1-bedroom under €800 near NOVA'")

if user_prompt:
    st.chat_message("user").write(user_prompt)

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": user_prompt},
            stream=True
        )

        result = ""
        for chunk in response.iter_lines():
            if chunk:
                try:
                    part = eval(chunk.decode())["response"]
                    result += part
                except:
                    continue

        st.chat_message("assistant").write(result)
        st.session_state.ollama_chat.append((user_prompt, result))

    except Exception as e:
        st.error("⚠️ Could not reach Ollama. Make sure it’s running.")
        st.caption(str(e))
