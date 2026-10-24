import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd

st.set_page_config(
    page_title="Everyday Object Recognition",
    page_icon="🔎",
    layout="centered"
)

st.title("🔎 Everyday Object Recognition AI")
st.write(
    "Upload a picture and the model will suggest what objects it can recognize. "
    "This project uses a pre-trained image-classification model."
)

st.sidebar.header("About the project")
st.sidebar.write("Topic: Image Classification")
st.sidebar.write("Interface: Streamlit")
st.sidebar.write("Model: Vision Transformer (ImageNet pre-trained)")
st.sidebar.write(
    "The model was trained on a large general image dataset, so results are predictions "
    "and may not always be correct."
)

@st.cache_resource
def load_model():
    return pipeline(
        task="image-classification",
        model="google/vit-base-patch16-224"
    )

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Analyze Image", type="primary"):
            with st.spinner("Loading the AI model and analyzing the image..."):
                classifier = load_model()
                predictions = classifier(image, top_k=5)

            st.subheader("Prediction results")
            best = predictions[0]
            st.success(
                f"Most likely result: **{best['label'].replace('_', ' ').title()}**"
            )
            st.caption(
                f"Model confidence for this prediction: {best['score'] * 100:.1f}%"
            )

            rows = []
            for item in predictions:
                label = item["label"].replace("_", " ").title()
                confidence = item["score"] * 100
                rows.append({"Possible label": label, "Confidence (%)": round(confidence, 2)})
                st.write(f"**{label}** — {confidence:.1f}%")
                st.progress(float(item["score"]))

            st.subheader("Results table")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.info(
                "Tip: Use a clear picture with the main object easy to see. "
                "This is a student demonstration, not a professional decision-making tool."
            )

    except Exception as error:
        st.error("The image could not be analyzed. Please try another image.")
        st.caption("If this is the first run, wait for the model to download and try again.")

else:
    st.info("Upload an image above to begin.")
    st.write("**How it works**")
    st.write("1. Select an image from your device.")
    st.write("2. Click **Analyze Image**.")
    st.write("3. Review the five most likely labels and their confidence scores.")
