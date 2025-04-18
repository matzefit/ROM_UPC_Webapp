import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

# --- Load models and PCA ---
@st.cache_resource
def load_models():
    models = {}
    pcas = {}
    for name in ["MRT", "MagVel", "AT"]:
        models[name] = joblib.load(f"{name}_rf_model.pkl")
        pcas[name] = joblib.load(f"{name}_pca.pkl")
    return models, pcas

rf_models, PCA_models = load_models()

# --- Sidebar for inputs ---
with st.sidebar:
    st.markdown("### 🌤 Input Parameters")
    hour = st.slider("Hour of Day", 1, 24, 12)
    windspeed = st.slider("Windspeed (m/s)", 0.11, 6.31, 2.0)
    winddir = st.slider("Wind Direction (°)", 3, 353, 180)
    solar = st.slider("Solar Irradiance (W/m²)", 0.0, 930.0, 500.0)
    airtemp = st.slider("Air Temperature (K)", 279.95, 310.05, 295.0)

input_array = np.array([[hour, windspeed, winddir, solar, airtemp]])

# --- Main Title ---
st.title("🌆 Urban Microclimate Predictor")
st.markdown("This tool uses trained ML models to predict high-resolution spatial fields for:")
st.markdown("""
- **MRT** (Mean Radiant Temperature)  
- **MagVel** (Velocity Magnitude)  
- **AT** (Air Temperature)
""")

st.markdown("---")
st.markdown("### 🎯 Predicted Output Maps")

# --- Predict and Plot each output (1 per row) ---
for target in ["MRT", "MagVel", "AT"]:
    model = rf_models[target]
    pca = PCA_models[target]

    # Predict + reshape
    pca_pred = model.predict(input_array)
    flat_img = pca.inverse_transform(pca_pred).reshape(-1)
    img = flat_img.reshape(1000, 1000)
    img[img == 0] = np.nan  # Mask buildings

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(img, cmap="jet")
    ax.set_title(f"**{target}** Prediction", fontsize=18)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    st.pyplot(fig)