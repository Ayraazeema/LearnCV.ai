import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(layout="wide")

# Sidebar: Upload image
st.sidebar.title("Image Operations")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

# Sidebar: Select operation
operation = st.sidebar.selectbox("Choose Operation", [
    "None (Show original)",
    "Convert to Grayscale",
    "Change Color Space: RGB ↔ HSV",
    "Rotate Image",
    "Edge Detection"
])

if operation == "Rotate Image":
    angle = st.sidebar.slider("Rotation Angle", min_value=-180, max_value=180, value=0, step=1)

# Add a toggle for color space display in sidebar only when needed
if operation == "Change Color Space: RGB ↔ HSV":
    show_hsv = st.sidebar.checkbox("Show HSV image (for analysis)", value=False)

if uploaded_file is not None:
    # Load image and convert to OpenCV format
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    processed_img = img_cv.copy()

    # Processing branches
    if operation == "Convert to Grayscale":
        processed_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # For display, don't convert color
        disp_processed = Image.fromarray(processed_img)
    elif operation == "Change Color Space: RGB ↔ HSV":
        # Convert BGR to HSV
        hsv_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        if show_hsv:
            # For analysis: Show HSV channels directly (will look odd to humans)
            disp_processed = Image.fromarray(hsv_img)
        else:
            # Convert to RGB for visual display (otherwise odd colors!)
            disp_processed = Image.fromarray(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB))
        processed_img = hsv_img  # For clarity
    elif operation == "Rotate Image":
        (h, w) = img_cv.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_cv, matrix, (w, h))
        processed_img = rotated
        disp_processed = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    elif operation == "Edge Detection":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        processed_img = edges
        disp_processed = Image.fromarray(processed_img)
    else:
        disp_processed = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # Use columns for side-by-side display
    col1, col2 = st.columns(2)
    col1.header("Original Image")
    col1.image(image, use_column_width=True)
    col2.header("Processed Image")
    col2.image(disp_processed, use_column_width=True)

else:
    st.info("Upload an image to get started.")

# Requirements: pip install streamlit opencv-python pillow numpy
