import streamlit as st
from model import load_saved_model
from data_loader import get_transforms
from config import model_save_path, device
from PIL import Image
import torch


# Load model (cached)
@st.cache_resource
def load_app_model():
    return load_saved_model(model_save_path)


# Streamlit app
def main():
    st.set_page_config(page_title="Gender Classification", layout="wide")
    st.title("Gender Classification using ResNet18")

    model = load_app_model()
    transform = get_transforms()[1]  # Use validation transform

    DEFAULT_IMAGE_PATH = "Aretha_Franklin_0001.jpg"
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Classify'):
            # Preprocess and predict
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

            # Display results
            st.success(f"Prediction: {'Male' if predicted.item() else 'Female'}")
            st.write("Confidence:")
            st.write(f"- Female: {probabilities[0]:.2f}%")
            st.write(f"- Male: {probabilities[1]:.2f}%")


if __name__ == '__main__':
    main()