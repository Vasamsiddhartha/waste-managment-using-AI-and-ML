from pathlib import Path
import PIL
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Local Modules
import settings
import helper

# Caching the model loading to optimize performance
@st.cache_resource
def load_model_once(model_path):
    return helper.load_model(model_path)

# Setting page layout
st.set_page_config(
    page_title="Waste Classification using AL AND ML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Waste Classification using AI AND ML")

# Sidebar for configuration
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio("Select Task", ['Detection', 'Segmentation'])

# Confidence Threshold
confidence_threshold = float(st.sidebar.slider(
    "Select Model Confidence Threshold", 0.0, 1.0, 0.4))

# Selecting Detection or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = load_model_once(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Config")
source_radio = st.sidebar.radio("Select Source", ['Image', 'Webcam'])

# Recyclable and Non-Recyclable Classes
recyclable_classes = ['plastic', 'glass', 'metal', "paper", "organic"]
non_recyclable_classes = ['batteries', 'e-waste', 'clothes']

# Define a mapping from class indices to class names
class_names = {
    0: 'organic',
    1: 'batteries',
    2: 'glass',
    3: 'metal',
    4: 'paper',
    5: 'plastic',
    6: 'clothes',
    7: 'e-waste',
}

# Initialize detected_classes as an empty dictionary
detected_classes = {}

# Function to display detected objects with bounding boxes and labels
def plot_boxes(image, boxes):
    for box in boxes:
        class_index = int(box.cls.item())
        class_name = class_names.get(class_index, "Unknown Category")
        
        # Determine if it's recyclable or non-recyclable and set color
        if class_name in recyclable_classes:
            color = (0, 255, 0)  # Green for recyclable
        elif class_name in non_recyclable_classes:
            color = (255, 0, 0)  # Red for non-recyclable
        else:
            color = (255, 255, 255)  # White for unknown

        # Draw bounding box on the image using OpenCV
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(image, f'{class_name}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Image input handling
source_img = None
if source_radio == 'Image':
    source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image", use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image", use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image', use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image, conf=confidence_threshold)
                boxes = res[0].boxes
                img = np.array(uploaded_image)
                img_with_boxes = plot_boxes(img, boxes)
                st.image(img_with_boxes, caption='Detected Image', use_column_width=True)

                try:
                    with st.expander("Detection Results"):
                        detected_classes = {}

                        for box in boxes:
                            class_index = int(box.cls.item())
                            class_name = class_names.get(class_index, "Unknown Category")
                            confidence_score = box.conf.item()

                            st.write(f"Detected Class: {class_name}, Confidence: {confidence_score:.2f}")

                            if confidence_score >= confidence_threshold:
                                detected_classes[class_name] = max(detected_classes.get(class_name, 0), confidence_score)

                        # Display results for detected classes
                        for class_name, confidence in detected_classes.items():
                            if class_name in recyclable_classes:
                                st.write(f"{class_name}: Recycle (Confidence: {confidence:.2f})")
                            elif class_name in non_recyclable_classes:
                                st.write(f"{class_name}: Non-Recycle (Confidence: {confidence:.2f})")
                            else:
                                st.write(f"{class_name}: Unknown Category (Confidence: {confidence:.2f})")

                except Exception as ex:
                    st.error("Error processing detection results.")
                    st.error(ex)

elif source_radio == 'Webcam':
    helper.play_webcam(confidence_threshold, model)

else:
    st.error("Please select a valid source type!")

# Adding feedback feature
st.sidebar.header("Feedback")
if st.sidebar.button('Give Feedback'):
    st.sidebar.text_input('Correct class label for the detected object:')
    st.sidebar.button('Submit Feedback')

# Visualizing detection results as a Pie Chart
with st.expander("Confidence Levels Pie Chart"):
    if detected_classes:  # Ensure detected_classes is not empty
        class_names = list(detected_classes.keys())
        confidence_values = list(detected_classes.values())

        # Create a pie chart using matplotlib
        fig, ax = plt.subplots()
        ax.pie(confidence_values, labels=class_names, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.

        st.pyplot(fig)
    else:
        st.write("No detection results to display.")
