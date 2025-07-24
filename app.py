import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load the trained model
MODEL_PATH = 'cnn_image_classification_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Load evaluation results
def load_evaluation_results():
    with open('evaluation_results.pkl', 'rb') as f:
        return pickle.load(f)
evaluation_results = load_evaluation_results()

# Function to preprocess uploaded image
def preprocess_image(image, img_size=(128, 128)):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert to BGR
    image = cv2.resize(image, img_size)  # Resize
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Function to make prediction
def predict_image(image):
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_label]
    return predicted_label, confidence

# Set custom theme colors
COLOR_BACKGROUND = "#57375D"
COLOR_TEXT = "#FFC8C8"
COLOR_PRIMARY = "#FF3FA4"
COLOR_SECONDARY = "#FF9B82"

st.markdown(f"""
    <style>
    .stApp {{
        background-color: {COLOR_BACKGROUND};
        color: {COLOR_TEXT};
    }}
    .stSidebar {{
        background-color: {COLOR_PRIMARY};
    }}
    .stButton>button {{
        background-color: {COLOR_SECONDARY};
        color: {COLOR_TEXT};
    }}
    </style>  
""", unsafe_allow_html=True)

# Streamlit App
st.title("Deepfake Detection App")
st.write("Upload an image to detect if it is a deepfake or not.")

# Sidebar: Display model performance metrics
st.sidebar.header("Model Performance Metrics")
if 'classification_report' in evaluation_results and isinstance(evaluation_results['classification_report'], dict):
    st.sidebar.write(f"**Accuracy**: {evaluation_results.get('accuracy', 0.0):.2f}")
    for label, metrics in evaluation_results['classification_report'].items():
        if label in ['0', '1']:
            st.sidebar.write(f"**Label {label}**:")
            st.sidebar.write(f"- Precision: {metrics['precision']:.2f}")
            st.sidebar.write(f"- Recall: {metrics['recall']:.2f}")
            st.sidebar.write(f"- F1-Score: {metrics['f1-score']:.2f}")
else:
    st.sidebar.write("Model performance metrics are unavailable or improperly formatted.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read and display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Uploaded Image", use_column_width=True, channels="BGR")

    # Preprocess and predict
    preprocessed_image = preprocess_image(image)
    predicted_label, confidence = predict_image(preprocessed_image)

    # Display prediction
    label_map = {0: "Fake", 1: "Real"}
    st.write(f"**Prediction:** {label_map[predicted_label]} ({confidence * 100:.2f}% confidence)")

    # Show confusion matrix (optional)
    st.subheader("Confusion Matrix")
    if 'confusion_matrix' in evaluation_results:
        fig, ax = plt.subplots()
        conf_matrix = evaluation_results['confusion_matrix']
        ax.matshow(conf_matrix, cmap='coolwarm', alpha=0.7)
        for (i, j), val in np.ndenumerate(conf_matrix):
            ax.text(j, i, f"{val}", ha='center', va='center')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        st.pyplot(fig)
    else:
        st.write("Confusion matrix is unavailable.")
