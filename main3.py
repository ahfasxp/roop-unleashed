import os
import cv2
import numpy as np
import insightface
from settings import Settings
import roop.globals
from roop.core import batch_process_regular
from roop.face_util import extract_face_images
from roop.ProcessEntry import ProcessEntry
from roop.ProcessOptions import ProcessOptions
from roop.FaceSet import FaceSet

print(f"InsightFace version: {insightface.__version__}")

def initialize_roop_globals():
    roop.globals.CFG = Settings('config.yaml')
    roop.globals.execution_threads = roop.globals.CFG.max_threads
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    roop.globals.max_memory = roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None
    roop.globals.face_swap_mode = 'first'
    roop.globals.INPUT_FACESETS = []
    roop.globals.TARGET_FACES = []
    roop.globals.output_path = 'output'
    if not os.path.exists(roop.globals.output_path):
        os.makedirs(roop.globals.output_path)

def prepare_faces(source_image, target_image):
    source_faces = extract_face_images(source_image, (False, 0))
    target_faces = extract_face_images(target_image, (False, 0))
    
    if source_faces:
        face_set = FaceSet()
        face = source_faces[0][0]
        if not hasattr(face, 'mask_offsets') or face.mask_offsets is None:
            face.mask_offsets = (0, 0, 0, 0, 1, 20)  # Default values
        face_set.faces.append(face)
        roop.globals.INPUT_FACESETS.append(face_set)
        print(f"Source face added with mask_offsets: {face.mask_offsets}")
    else:
        print("Warning: No source face detected!")
    
    if target_faces:
        roop.globals.TARGET_FACES.append(target_faces[0][0])
        print("Target face added successfully")
    else:
        print("Warning: No target face detected!")

def perform_face_swap(source_image, target_image):
    initialize_roop_globals()
    prepare_faces(source_image, target_image)
    
    if not roop.globals.INPUT_FACESETS or not roop.globals.TARGET_FACES:
        print("Error: Unable to proceed with face swap. Missing source or target faces.")
        return

    process_entry = ProcessEntry(target_image, 0, 0, 0)
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
    except Exception as e:
        print(f"Error during face swap process: {str(e)}")

def check_model_files(model_path):
    required_files = ['genderage.onnx', '2d106det.onnx', 'det_10g.onnx', '1k3d68.onnx', 'w600k_r50.onnx']
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            print(f"Warning: Required model file {file} not found in {model_path}")
            return False
    return True

def main():
    # Initialize face analyser
    if not hasattr(roop.globals, 'face_analyser'):
        model_path = os.path.abspath('models/buffalo_l')
        if not check_model_files(model_path):
            print("Error: Some required model files are missing. Please ensure all model files are present.")
            return

        roop.globals.face_analyser = insightface.app.FaceAnalysis(
            name="buffalo_l", 
            root=os.path.abspath('models'),
            providers=['CPUExecutionProvider'],
            allowed_modules=['detection', 'recognition'],
            download=False  # Prevent download attempt
        )
        roop.globals.face_analyser.prepare(ctx_id=0, det_size=(640, 640))

    # Source and target images
    face_source = 'face-simon-robben.jpg'
    target_image = 'avatar.png'

    # Verify source image
    if not os.path.exists(face_source):
        print(f"Error: Source image not found: {face_source}")
        return
    
    # Verify target image
    if not os.path.exists(target_image):
        print(f"Error: Target image not found: {target_image}")
        return

    print(f"Source image: {face_source}")
    print(f"Target image: {target_image}")

    # Perform face swap
    perform_face_swap(face_source, target_image)

    print("Face swap process completed. Please check the output directory for results.")

if __name__ == "__main__":
    main()