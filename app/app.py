import base64
from pathlib import Path

import requests
import streamlit as st

API = "http://localhost:8000"
ROOT = Path(__file__).resolve().parents[1]

st.title("Waste Sorting — Class 7")


def call(method, route, **kwargs):
    try:
        response = requests.request(
            method,
            API + route,
            timeout=60,
            **kwargs
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        st.error(f"API request failed: {error}")
        st.stop()


task = next(
    t for t in call("GET", "/tasks")["tasks"]
    if t["name"] == "trash"
)

st.caption(task["model_name"])

threshold = st.slider(
    "Not-sure threshold",
    0.0,
    1.0,
    float(task["not_sure_threshold"]),
    0.01
)

source = st.radio(
    "Image source",
    ["Saved sample", "Upload"],
    horizontal=True
)

if source == "Saved sample":
    folder = ROOT / "data/trash/samples"
    samples = sorted(folder.glob("*.jpg"))

    if not samples:
        st.info("Notebook export must finish first.")
        st.stop()

    selected = st.selectbox(
        "Choose a sample",
        samples,
        format_func=lambda p: p.name
    )

    data = selected.read_bytes()
    st.write("Actual label:", selected.name.split("__")[0])

else:
    photo = st.file_uploader(
        "Upload one object photo",
        type=["jpg", "jpeg", "png"]
    )

    if photo is None:
        st.info("Choose a photo.")
        st.stop()

    data = photo.getvalue()

st.image(data, width=320)

if st.button("Predict"):
    out = call(
        "POST",
        "/tasks/trash/predict",
        params={"threshold": threshold},
        files={"file": ("photo.jpg", data)}
    )

    if out["decision"] == "not_sure":
        st.warning(
            f"Not sure — top guess: {out['top_class']} "
            f"({out['probability']:.1%})"
        )
    else:
        st.success(
            f"{out['top_class']} ({out['probability']:.1%})"
        )

    st.bar_chart({"Probability": out["probabilities"]})

    st.subheader("Grad-CAM")

    explanation = call(
        "POST",
        "/tasks/trash/explain",
        files={"file": ("photo.jpg", data)}
    )

    st.image(
        base64.b64decode(explanation["overlay_png"]),
        width=320,
        caption="Class-score attribution — not proof of reasoning."
    )