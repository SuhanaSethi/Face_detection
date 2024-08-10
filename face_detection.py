# from PIL import Image
import landingai
# from landingai.predict import Predictor
from hidden import *
# # Enter your API Key

# # # Load your image
# # image = Image.open("image.png")
# # # Run inference
# # predictor = Predictor(endpoint_id, api_key=api_key)
# # predictions = predictor.predict(image)

# from PIL import Image
# from landingai.pipeline.image_source import Webcam
# from landingai.predict import Predictor
# import numpy as np

# from landingai.pipeline.image_source import Webcam
# from landingai.predict import Predictor
# import os

# # Initialize the Predictor with your API key and endpoint ID
# predictor = Predictor(  
#     endpoint_id=my_new_endpoint, 
#     api_key=my_api, 
# )


# # Create the directory if it doesn't exist
# save_dir = "output_images"
# os.makedirs(save_dir, exist_ok=True)

# # Initialize the webcam and start capturing frames
# with Webcam(fps=2.0) as webcam:
#     for frame in webcam:
#         # Resize the frame to the desired width
#         frame.resize(width=512)
        
#         # Run prediction on the current frame
#         frame.run_predict(predictor=predictor)
        
#         # Overlay predictions on the frame
#         frame.overlay_predictions()
        
#         # Check for each class ("front", "left", "right") in the predictions
#         if any(cls in frame.predictions for cls in ["front", "left", "right"]):
#             # Save the image with predictions overlaid in the specified directory
#             save_path = os.path.join(save_dir, "latest-webcam-image.png")
#             frame.save_image(save_path, include_predictions=True)


from landingai.pipeline.image_source import Webcam
from landingai.predict import Predictor
import os
import cv2  

# Initialize the Predictor with your API key and endpoint ID
predictor = Predictor(  
    endpoint_id=my_new_endpoint, 
    api_key=my_api, 
)

# Create the directory if it doesn't exist
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

# Initialize the webcam and start capturing frames
with Webcam(fps=0.5) as webcam:
    for frame in webcam:
        # Resize the frame to the desired width
        frame.resize(width=512)
        
        # Run prediction on the current frame
        frame.run_predict(predictor=predictor)
        
        # Overlay predictions on the frame
        frame.overlay_predictions()
        
        # Show the frame with predictions
        cv2.imshow("Webcam Feed", frame.image)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF

        # If 'c' is pressed, capture the frame
        if key == ord('c'):
            # Get the predictions (assuming the first prediction is the desired one)
            if frame.predictions:
                prediction = frame.predictions[0]['label']  # Get the label of the first prediction
                save_path = os.path.join(save_dir, f"{prediction}.png")
                
                # Save the image with the prediction as the file name
                frame.save_image(save_path, include_predictions=True)
                print(f"Captured and saved image: {save_path}")

        # If 'q' is pressed, quit the loop
        elif key == ord('q'):
            print("Quitting...")
            break

# Release the webcam and close all OpenCV windows
cv2.destroyAllWindows()
