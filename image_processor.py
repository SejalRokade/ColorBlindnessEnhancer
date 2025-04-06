import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import filters, transform
import io

def load_image():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        return np.array(image)
    return None

def display_image(image, title="Image"):
    st.image(image, caption=title, use_column_width=True)

def point_processing(image, operation, threshold=128):
    if operation == "Negation":
        return 255 - image
    elif operation == "Thresholding":
        return np.where(image > threshold, 255, 0)
    elif operation == "Darken":
        return np.clip(image * 0.5, 0, 255).astype(np.uint8)
    elif operation == "Lighten":
        return np.clip(image * 1.5, 0, 255).astype(np.uint8)

def histogram_equalization(image):
    if len(image.shape) == 3:
        # Convert to YUV color space
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    else:
        return cv2.equalizeHist(image)

def apply_filter(image, filter_type):
    if filter_type == "Gaussian Blur":
        return cv2.GaussianBlur(image, (5,5), 0)
    elif filter_type == "Sobel":
        return cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)
    elif filter_type == "Prewitt":
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        prewittx = cv2.filter2D(image, -1, kernelx)
        prewitty = cv2.filter2D(image, -1, kernely)
        return np.sqrt(prewittx**2 + prewitty**2)
    elif filter_type == "High Boost":
        blurred = cv2.GaussianBlur(image, (5,5), 0)
        return cv2.addWeighted(image, 2.0, blurred, -1.0, 0)

def simulate_color_blindness(image, cvd_type):
    # Convert to float for processing
    img_float = image.astype(np.float32) / 255.0
    
    # Ensure image is in RGB format
    if len(img_float.shape) == 3:
        # Convert BGR to RGB if needed
        if img_float.shape[2] == 3:
            img_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
    else:
        # If grayscale, convert to RGB
        img_float = cv2.cvtColor(img_float, cv2.COLOR_GRAY2RGB)
    
    # Get image dimensions
    h, w = img_float.shape[:2]
    
    if cvd_type == "Protanopia":
        # Red deficiency
        transform_matrix = np.array([
            [0.567, 0.433, 0],
            [0.558, 0.442, 0],
            [0, 0.242, 0.758]
        ])
    elif cvd_type == "Deuteranopia":
        # Green deficiency
        transform_matrix = np.array([
            [0.625, 0.375, 0],
            [0.7, 0.3, 0],
            [0, 0.3, 0.7]
        ])
    elif cvd_type == "Tritanopia":
        # Blue deficiency
        transform_matrix = np.array([
            [0.95, 0.05, 0],
            [0, 0.433, 0.567],
            [0, 0.475, 0.525]
        ])
    
    # Apply color transformation using matrix multiplication
    simulated = np.zeros_like(img_float)
    for i in range(h):
        for j in range(w):
            simulated[i, j] = np.dot(img_float[i, j], transform_matrix.T)
    
    # Clip values to valid range
    simulated = np.clip(simulated, 0, 1)
    
    return (simulated * 255).astype(np.uint8)

def enhance_for_color_blindness(image, cvd_type, enhancement_strength=1.0):
    # Convert to float for processing
    img_float = image.astype(np.float32) / 255.0
    
    # Apply histogram equalization for better contrast
    enhanced = histogram_equalization(image) / 255.0
    
    # Simulate color blindness
    simulated = simulate_color_blindness(image, cvd_type) / 255.0
    
    # Enhance the difference between original and simulated
    difference = enhanced - simulated
    enhanced = enhanced + (difference * enhancement_strength)
    
    # Clip values to valid range
    enhanced = np.clip(enhanced, 0, 1)
    
    return (enhanced * 255).astype(np.uint8)

def main():
    st.title("Color Blindness Image Processor")
    
    # Sidebar for operations
    st.sidebar.title("Operations")
    operation_type = st.sidebar.selectbox(
        "Select Operation Type",
        ["Color Blindness Simulation", "Color Blindness Enhancement", "Basic Image Processing"]
    )
    
    # Load image
    image = load_image()
    if image is not None:
        # Display original image
        st.subheader("Original Image")
        display_image(image)
        
        if operation_type == "Color Blindness Simulation":
            st.sidebar.subheader("Color Vision Deficiency Options")
            cvd_type = st.sidebar.selectbox(
                "Select Color Vision Deficiency Type",
                ["Protanopia", "Deuteranopia", "Tritanopia"]
            )
            
            simulated_image = simulate_color_blindness(image, cvd_type)
            st.subheader(f"Simulated {cvd_type} View")
            display_image(simulated_image)
            
        elif operation_type == "Color Blindness Enhancement":
            st.sidebar.subheader("Enhancement Options")
            cvd_type = st.sidebar.selectbox(
                "Select Color Vision Deficiency Type",
                ["Protanopia", "Deuteranopia", "Tritanopia"]
            )
            
            enhancement_strength = st.sidebar.slider(
                "Enhancement Strength",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
            
            enhanced_image = enhance_for_color_blindness(image, cvd_type, enhancement_strength)
            st.subheader(f"Enhanced for {cvd_type}")
            display_image(enhanced_image)
            
        elif operation_type == "Basic Image Processing":
            st.sidebar.subheader("Basic Processing Options")
            basic_operation = st.sidebar.selectbox(
                "Select Operation",
                ["Point Processing", "Histogram Processing", "Spatial Filtering"]
            )
            
            if basic_operation == "Point Processing":
                st.sidebar.subheader("Point Processing Options")
                point_operation = st.sidebar.selectbox(
                    "Select Operation",
                    ["Negation", "Thresholding", "Darken", "Lighten"]
                )
                
                if point_operation == "Thresholding":
                    threshold = st.sidebar.slider("Threshold Value", 0, 255, 128)
                    processed_image = point_processing(image, point_operation, threshold)
                else:
                    processed_image = point_processing(image, point_operation)
                
                st.subheader("Processed Image")
                display_image(processed_image)
                
            elif basic_operation == "Histogram Processing":
                st.sidebar.subheader("Histogram Options")
                if st.sidebar.button("Apply Histogram Equalization"):
                    processed_image = histogram_equalization(image)
                    st.subheader("Histogram Equalized Image")
                    display_image(processed_image)
                    
                    # Display histograms
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.hist(image.ravel(), 256, [0, 256])
                    ax1.set_title('Original Histogram')
                    ax2.hist(processed_image.ravel(), 256, [0, 256])
                    ax2.set_title('Equalized Histogram')
                    st.pyplot(fig)
                    
            elif basic_operation == "Spatial Filtering":
                st.sidebar.subheader("Filter Options")
                filter_type = st.sidebar.selectbox(
                    "Select Filter",
                    ["Gaussian Blur", "Sobel", "Prewitt", "High Boost"]
                )
                
                processed_image = apply_filter(image, filter_type)
                st.subheader(f"{filter_type} Filtered Image")
                display_image(processed_image)

if __name__ == "__main__":
    main() 