import os
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from flask import Flask, Response
import mss
from roop.core import live_swap, ProcessOptions, release_resources, limit_resources
from roop.globals import INPUT_FACESETS, TARGET_FACES, blend_ratio, subsample_size

app = Flask(__name__)

# Ensure the 'static' folder exists for saving processed frames
if not os.path.exists('static'):
    os.makedirs('static')

# Capture screenshot using mss and crop the unwanted part (e.g., top menu bar)
def capture_screenshot():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Capture the primary monitor (index 1)
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR
        
        # Crop the image (adjust the coordinates as necessary)
        cropped_img = img[350:, :]  # Crop top 350 pixels

        return cropped_img

# Function to process frame (perform face swap) and return the processed image
def process_frame():
    start_time = time.time()  # Start measuring time

    # Capture a screenshot using mss
    frame = capture_screenshot()

    # Prepare options for face swapping
    options = ProcessOptions(
        processordefines={},
        face_distance=0.6,  # Example value, replace with actual
        blend_ratio=blend_ratio,
        swap_mode=1,  # Example value, replace with actual
        selected_index=0,
        masking_text=None,
        imagemask=None,
        num_steps=1,
        subsample_size=subsample_size,
        show_face_area=False,
        restore_original_mouth=False
    )

    # Perform face swap
    processed_frame = live_swap(frame, options)

    process_duration = time.time() - start_time  # End measuring time
    print(f"Process duration: {process_duration:.4f} seconds")

    return processed_frame

# Generator function for streaming video frames
def generate_frames():
    # Initialize Selenium WebDriver with headless mode
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")  # Ensure full screen on start
    driver = webdriver.Chrome(options=chrome_options)

    # Maximize the browser window
    driver.maximize_window()

    # Open the website
    driver.get('http://localhost:3000')
    time.sleep(2)  # Give the page time to load

    # Variables for calculating average time
    frame_count = 0
    total_time = 0

    try:
        while True:
            frame_start_time = time.time()  # Start measuring time for this frame

            # Process the frame and yield the result
            processed_frame = process_frame()

            # Encode the frame in JPEG format for streaming
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()

            # Yield the frame as a byte stream (MIME type multipart/x-mixed-replace)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # Measure time for this frame
            frame_duration = time.time() - frame_start_time
            total_time += frame_duration
            frame_count += 1

            # Print average processing time every 10 frames
            if frame_count % 10 == 0:
                avg_time = total_time / frame_count
                print(f"Average processing time per frame: {avg_time:.4f} seconds")

    finally:
        driver.quit()

# Flask route for video streaming
@app.route('/')
def video_feed():
    # Streaming the frames
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    # Start the Flask server
    app.run(debug=True)

if __name__ == "__main__":
    main()