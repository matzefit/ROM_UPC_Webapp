import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle

# --- Load models and PCA ---
@st.cache_resource
def load_models():
    models = {}
    pcas = {}
    for name in ["MRT", "MagVel", "AT"]:
        with open(f"{name}_model.pkl", "rb") as f:
            models[name] = pickle.load(f)
        with open(f"{name}_pca.pkl", "rb") as f:
            pcas[name] = pickle.load(f)
    return models, pcas

rf_models, PCA_models = load_models()

# --- Streamlit UI ---
st.title("Urban Microclimate Predictor 🌆")
st.markdown("Adjust the inputs to predict **MRT**, **Velocity**, and **Air Temperature** maps.")

hour = st.slider("Hour of Day", 1, 24, 12)
windspeed = st.slider("Windspeed (m/s)", 0.0, 20.0, 2.0)
winddir = st.slider("Wind Direction (°)", 0, 360, 180)
solar = st.slider("Solar Irradiance (W/m²)", 0, 1000, 500)
airtemp = st.slider("Air Temperature (K)", 285.0, 315.0, 300.0)

input_array = np.array([[hour, windspeed, winddir, solar, airtemp]])

# --- Prediction ---
st.subheader("Predicted Output Maps")

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for i, target in enumerate(["MRT", "MagVel", "AT"]):
    model = rf_models[target]
    pca = PCA_models[target]

    pca_pred = model.predict(input_array)
    flat_img = pca.inverse_transform(pca_pred).reshape(-1)
    img = flat_img.reshape(1000, 1000)

    img[img == 0] = np.nan  # Mask buildings

    im = axs[i].imshow(img, cmap="plasma" if target != "Error" else "coolwarm")
    axs[i].set_title(f"{target} Prediction")
    axs[i].axis("off")
    fig.colorbar(im, ax=axs[i], fraction=0.025, pad=0.02)

st.pyplot(fig)