import os
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from flask import Flask, Response
import mss
import roop.globals
from roop.core import batch_process_regular
from roop.face_util import get_face_analyser, extract_face_images
from roop.ProcessEntry import ProcessEntry
from roop.ProcessOptions import ProcessOptions
from roop.FaceSet import FaceSet
from settings import Settings

app = Flask(__name__)

def capture_screenshot():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cropped_img = img[350:, :]
        return cropped_img

def initialize_roop_globals():
    roop.globals.CFG = Settings('config.yaml')
    roop.globals.execution_threads = roop.globals.CFG.max_threads
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    roop.globals.max_memory = roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None
    roop.globals.face_swap_mode = 'first'
    roop.globals.INPUT_FACESETS = []
    roop.globals.TARGET_FACES = []
    roop.globals.output_path = os.path.join('output', 'frame')
    output_dir = os.path.dirname(roop.globals.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

def prepare_faces(source_image, target_image):
    face_analyser = get_face_analyser()
    source_faces = face_analyser.get(source_image)
    target_faces = face_analyser.get(target_image)
    
    if source_faces:
        face_set = FaceSet()
        face = source_faces[0]
        if not hasattr(face, 'mask_offsets') or face.mask_offsets is None:
            face.mask_offsets = (0, 0, 0, 0, 1, 20)
        face_set.faces.append(face)
        roop.globals.INPUT_FACESETS.append(face_set)
    else:
        print("Warning: No source face detected!")
    
    if target_faces:
        roop.globals.TARGET_FACES.append(target_faces[0])
    else:
        print("Warning: No target face detected!")

def get_latest_file(directory):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        if not files:
            return None
        files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
        return os.path.join(directory, files[0])
    except Exception as e:
        print(f"Error while fetching latest file: {str(e)}")
        return None

def perform_face_swap(source_image_path, target_image_path):
    # Read images
    source_image = cv2.imread(source_image_path)
    target_image = cv2.imread(target_image_path)
    
    if source_image is None:
        print(f"Error: Unable to read source image from {source_image_path}")
        return None
    if target_image is None:
        print(f"Error: Unable to read target image from {target_image_path}")
        return None
    
    prepare_faces(source_image, target_image)
    
    if not roop.globals.INPUT_FACESETS or not roop.globals.TARGET_FACES:
        print("Error: Unable to proceed with face swap. Missing source or target faces.")
        return None

    process_entry = ProcessEntry(target_image_path, 0, 0, 0)
    options = ProcessOptions(
        {"faceswap": {}},
        0.2,  # distance_threshold
        1,    # blend_ratio
        'first',  # face_swap_mode
        0,    # selected_index
        None, # clip_text
        None, # imagemask
        1,    # num_swap_steps
        128,  # subsample_size
        False,# show_face_area
        False # restore_original_mouth
    )
    
    try:
        batch_process_regular([process_entry], None, None, True, None, False, 1, None, 0)
        
        # Get the most recent file from the output directory
        result_path = get_latest_file(roop.globals.output_path)
        
        if result_path:
            result = cv2.imread(result_path)
            print(f"Face swap successful, output image shape: {result.shape}")

            # Clean up the output directory
            for file in os.listdir(roop.globals.output_path):
                os.remove(os.path.join(roop.globals.output_path, file))

            return result
        else:
            print("No result file found.")
            return None
    except Exception as e:
        print(f"Error during face swap process: {str(e)}")
        return None

def process_frame_async():
    start_time = time.time()

    frame = capture_screenshot()
    cv2.imwrite('temp_frame.png', frame)

    processed_frame = perform_face_swap(roop.globals.source_path, 'temp_frame.png')

    if processed_frame is None:
        processed_frame = frame

    process_duration = time.time() - start_time
    print(f"Process duration: {process_duration:.4f} seconds")

    return processed_frame

def generate_frames():
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=chrome_options)

    driver.maximize_window()
    driver.get('http://localhost:3000')
    time.sleep(2)

    frame_count = 0
    total_time = 0

    try:
        while True:
            frame_start_time = time.time()

            processed_frame = process_frame_async()

            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            frame_duration = time.time() - frame_start_time
            total_time += frame_duration
            frame_count += 1

            if frame_count % 10 == 0:
                avg_time = total_time / frame_count
                print(f"Average processing time per frame: {avg_time:.4f} seconds")

    finally:
        driver.quit()

@app.route('/')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    initialize_roop_globals()

    roop.globals.execution_providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']

    roop.globals.source_path = 'face-simon-robben.jpg'

    # Initialize face analyser with the correct det_size
    face_analyser = get_face_analyser()
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))

    app.run(debug=True)

if __name__ == "__main__":
    main()