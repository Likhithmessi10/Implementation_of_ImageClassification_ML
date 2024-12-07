import streamlit as st
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
from PIL import Image

# Function to load the CIFAR-10 model from .h5 file
def load_cifar10_model():
    try:
        # Load CIFAR-10 model from the .h5 file
        st.write("Attempting to load cifar10_model.h5")
        model = tf.keras.models.load_model('cifar10_model.h5')
        st.success("Successfully loaded cifar10_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading CIFAR-10 model: {str(e)}")
        return None

# Function to load the MobileNetV2 model
def load_mobilenet_model():
    # Load MobileNetV2 pre-trained on ImageNet
    return tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to preprocess image for MobileNetV2
def preprocess_mobilenet_image(image):
    # Resize the image to 224x224 for MobileNetV2
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to preprocess image for CIFAR-10 model
def preprocess_cifar_image(image):
    # Resize the image to 32x32 for CIFAR-10 model
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to get prediction using MobileNetV2
def predict_mobilenet(image):
    model = load_mobilenet_model()
    image_array = preprocess_mobilenet_image(image)
    predictions = model.predict(image_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
    # Get top 5 predictions
    top_predictions = decoded_predictions[0][:5]  # Top 5 predictions
    result = []
    for pred in top_predictions:
        class_name = pred[1]
        confidence = pred[2] * 100  # Convert to percentage
        result.append(f"{class_name}: {confidence:.2f}%")
    return result

# Function to get prediction using CIFAR-10 model
def predict_cifar10(image):
    model = load_cifar10_model()
    if model is None:
        return ["Error loading CIFAR-10 model"]
    image_array = preprocess_cifar_image(image)
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)
    class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    class_name = class_labels[predicted_class[0]]
    confidence = np.max(predictions) * 100  # Convert to percentage
    return [f"{class_name}: {confidence:.2f}%"]

# Streamlit app UI
st.title("Image Classification with MobileNetV2 and CIFAR-10")
st.write("Select a model to classify your image:")

# Dropdown to select model
model_choice = st.selectbox("Select Model", ["MobileNetV2", "CIFAR-10"])

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated for deprecated warning

    if st.button("Classify Image"):
        if model_choice == "MobileNetV2":
            # Get predictions from MobileNetV2
            predictions = predict_mobilenet(image)
            st.write("Predictions:")
            for pred in predictions:
                st.write(pred)  # Display each prediction on a new line
        elif model_choice == "CIFAR-10":
            # Get prediction from CIFAR-10 model
            predictions = predict_cifar10(image)
            st.write("Prediction:")
            for pred in predictions:
                st.write(pred)  # Display prediction on a new line
