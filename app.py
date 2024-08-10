import gradio as gr
import numpy as np
from PIL import Image
from landingai.pipeline.image_source import Webcam
from landingai.predict import Predictor
import os
import cv2
from hidden import *
# Initialize the Predictor with your API key and endpoint ID
predictor = Predictor(
    endpoint_id=my_new_endpoint,
    api_key=my_api,
)

# Create the directory if it doesn't exist
save_dir = "output_images"
os.makedirs(save_dir, exist_ok=True)

# Function to process and display webcam frames
def process_frame():
    with Webcam(fps=2.0) as webcam:
        for frame in webcam:
            # Resize the frame to the desired width
            frame.resize(width=512)
            
            # Run prediction on the current frame
            frame.run_predict(predictor=predictor)
            
            # Overlay predictions on the frame
            frame.overlay_predictions()
            
            # Convert the frame to a NumPy array
            image_array = np.array(frame.image)
            
            # Ensure the image is in BGR format for OpenCV
            if image_array.shape[-1] == 4:  # RGBA to BGR
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            elif image_array.shape[-1] == 3:  # RGB to BGR
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            # Convert to PIL Image for Gradio
            img_pil = Image.fromarray(image_array)
            
            # Save the image with predictions
            if any(cls in frame.predictions for cls in ["front", "left", "right"]):
                save_path = os.path.join(save_dir, "latest-webcam-image.png")
                frame.save_image(save_path, include_predictions=True)
                print(f"Captured and saved image: {save_path}")

            return img_pil

# Create a Gradio interface
iface = gr.Interface(
    fn=process_frame, 
    inputs=[],  # No inputs needed, just captures from webcam
    outputs="image",  # Output is an image
    live=True,  # Set live to True for real-time updates
    title="Webcam Capture",
    description="Capture webcam images and display predictions"
)

# Launch the Gradio interface
iface.launch(share=True)
