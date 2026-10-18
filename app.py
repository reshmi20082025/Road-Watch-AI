
import os
import tempfile
import streamlit as st
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

st.set_page_config(page_title="RoadWatch AI", layout="wide")
st.title("RoadWatch AI")
st.subheader("Road Surface and Hazard Detection")
st.write("Upload a road image to detect possible potholes and cracks.")

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    st.error("Roboflow API key not found. Please set it again.")
    st.stop()

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
).configure(
    InferenceConfiguration(api_key_transport="header")
)

uploaded_file = st.file_uploader(
    "Choose a road image", type=["jpg", "jpeg", "png"]
)
confidence = st.slider(
    "Minimum confidence",
    0.01, 0.90, 0.10, 0.01
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Road Image", width="stretch")

    if st.button("Detect Road Hazards"):
        temp_path = None
        try:
            with st.spinner("Analysing road image..."):
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as temp:
                    temp_path = temp.name
                    image.save(temp_path, format="JPEG")

                result = client.infer(
                    temp_path,
                    model_id="crack-and-pothole-bftyl/1"
                )

            all_predictions = result.get("predictions", [])
            detected = [
                p for p in all_predictions
                if float(p.get("confidence", 0)) >= confidence
            ]
            rejected = len(all_predictions) - len(detected)

            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)

            for p in detected:
                x, y = float(p["x"]), float(p["y"])
                w, h = float(p["width"]), float(p["height"])
                left, top = x - w / 2, y - h / 2
                right, bottom = x + w / 2, y + h / 2
                label = f'{p["class"]} {float(p["confidence"]):.1%}'

                draw.rectangle(
                    [left, top, right, bottom],
                    outline="red", width=4
                )
                draw.text(
                    (max(0, left), max(0, top - 20)),
                    label, fill="red"
                )

            st.subheader("Detection Results")
            st.image(
                annotated,
                caption="Detected road hazards",
                width="stretch"
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Detected Objects", len(detected))
            col2.metric("Rejected by Confidence", rejected)
            col3.metric("Total Predictions", len(all_predictions))

            if detected:
                st.dataframe([
                    {
                        "Hazard": p["class"],
                        "Confidence": f'{float(p["confidence"]):.1%}',
                        "Status": "Detected"
                    }
                    for p in detected
                ], use_container_width=True)
            else:
                st.warning(
                    "No predictions passed the confidence threshold."
                )

            st.caption(
                "AI predictions need human verification; "
                "they are not official road-safety assessments."
            )

        except Exception as e:
            st.error(f"Detection failed: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
