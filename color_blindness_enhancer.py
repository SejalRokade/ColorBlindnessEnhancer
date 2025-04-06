import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

class ColorBlindnessEnhancer:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Blindness Image Enhancer")
        self.root.geometry("1200x800")
        
        # Variables
        self.current_image = None
        self.enhanced_image = None
        self.original_photo = None
        self.enhanced_photo = None
        self.cvd_type = tk.StringVar(value="protanopia")
        self.enhancement_strength = tk.DoubleVar(value=1.0)
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nw")
        
        # Load Image Button
        ttk.Button(control_frame, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=5, pady=5)
        
        # CVD Type Selection
        ttk.Label(control_frame, text="Color Vision Deficiency Type:").grid(row=1, column=0, padx=5, pady=5)
        cvd_combo = ttk.Combobox(control_frame, textvariable=self.cvd_type)
        cvd_combo['values'] = ('protanopia', 'deuteranopia', 'tritanopia')
        cvd_combo.grid(row=1, column=1, padx=5, pady=5)
        cvd_combo.bind('<<ComboboxSelected>>', lambda e: self.process_image())
        
        # Enhancement Strength Slider
        ttk.Label(control_frame, text="Enhancement Strength:").grid(row=2, column=0, padx=5, pady=5)
        strength_slider = ttk.Scale(control_frame, from_=0.1, to=2.0, orient="horizontal",
                                  variable=self.enhancement_strength, command=lambda e: self.process_image())
        strength_slider.grid(row=2, column=1, padx=5, pady=5)
        
        # Save Button
        ttk.Button(control_frame, text="Save Enhanced Image", command=self.save_image).grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        # Image Display Area
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=0, column=1, padx=5, pady=5)
        
        # Original Image
        self.original_label = ttk.Label(display_frame, text="Original Image")
        self.original_label.grid(row=0, column=0, padx=5)
        self.original_display = ttk.Label(display_frame)
        self.original_display.grid(row=1, column=0, padx=5)
        
        # Enhanced Image
        self.enhanced_label = ttk.Label(display_frame, text="Enhanced Image")
        self.enhanced_label.grid(row=0, column=1, padx=5)
        self.enhanced_display = ttk.Label(display_frame)
        self.enhanced_display.grid(row=1, column=1, padx=5)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path:
            try:
                self.current_image = cv2.imread(file_path)
                self.display_image(self.current_image, self.original_display)
                self.process_image()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def process_image(self, *args):
        if self.current_image is None:
            return
            
        try:
            # Process image
            enhanced = self.enhance_for_colorblind(
                self.current_image,
                self.cvd_type.get(),
                self.enhancement_strength.get()
            )
            self.enhanced_image = enhanced
            self.display_image(enhanced, self.enhanced_display)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def save_image(self):
        if self.enhanced_image is None:
            messagebox.showwarning("Warning", "No enhanced image to save!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            try:
                cv2.imwrite(file_path, self.enhanced_image)
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def display_image(self, image, label, max_size=500):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate new dimensions while maintaining aspect ratio
        height, width = image_rgb.shape[:2]
        scale = min(max_size/width, max_size/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image_rgb, (new_width, new_height))
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(Image.fromarray(resized))
        
        # Update label
        label.configure(image=photo)
        label.image = photo  # Keep a reference
    
    def apply_histogram_equalization(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = cv2.equalizeHist(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    
    def simulate_protanopia(self, image):
        protanopia = np.array([
            [0.567, 0.433, 0],
            [0.558, 0.442, 0],
            [0, 0.242, 0.758]
        ])
        return cv2.transform(image, protanopia)
    
    def simulate_deuteranopia(self, image):
        deuteranopia = np.array([
            [0.625, 0.375, 0],
            [0.7, 0.3, 0],
            [0, 0.3, 0.7]
        ])
        return cv2.transform(image, deuteranopia)
    
    def simulate_tritanopia(self, image):
        tritanopia = np.array([
            [0.95, 0.05, 0],
            [0, 0.433, 0.567],
            [0, 0.475, 0.525]
        ])
        return cv2.transform(image, tritanopia)
    
    def enhance_for_colorblind(self, image, cvd_type, enhancement_strength=1.0):
        # Convert to float32 for processing
        image = image.astype(np.float32) / 255.0
        
        # Apply histogram equalization
        enhanced = self.apply_histogram_equalization(image)
        
        # Apply color transformation based on CVD type
        if cvd_type == 'protanopia':
            simulated = self.simulate_protanopia(enhanced)
        elif cvd_type == 'deuteranopia':
            simulated = self.simulate_deuteranopia(enhanced)
        elif cvd_type == 'tritanopia':
            simulated = self.simulate_tritanopia(enhanced)
        else:
            return enhanced
        
        # Enhance contrast between original and simulated
        difference = enhanced - simulated
        enhanced = enhanced + (difference * enhancement_strength)
        
        # Clip values to valid range
        enhanced = np.clip(enhanced, 0, 1)
        
        # Convert back to uint8
        return (enhanced * 255).astype(np.uint8)

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorBlindnessEnhancer(root)
    root.mainloop() 