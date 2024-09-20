from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import cv2
import numpy as np
import os
import roop.globals
import roop.metadata
from roop.face_util import extract_face_images, get_all_faces
from roop.ProcessOptions import ProcessOptions
from roop.ProcessMgr import ProcessMgr
import matplotlib.pyplot as plt
import insightface
from settings import Settings

print(f"InsightFace version: {insightface.__version__}")

def init_face_analyser():
    print("Initializing face analyser...")
    if roop.globals.CFG is None:
        print("Initializing global configuration...")
        roop.globals.CFG = Settings('config.yaml')
    
    if not hasattr(roop.globals.CFG, 'models_dir'):
        print("Warning: 'models_dir' not found in configuration. Setting default value.")
        setattr(roop.globals.CFG, 'models_dir', 'models')

    model_path = os.path.abspath(roop.globals.CFG.models_dir)
    print(f"Model path: {model_path}")

    # Check if all required model files exist
    required_files = ['det_10g.onnx', 'w600k_r50.onnx']
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, 'buffalo_l', file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"Warning: The following model files are missing: {', '.join(missing_files)}")
        print("You may need to manually download and extract the model files.")
        return None

    if not hasattr(roop.globals, 'g_desired_face_analysis'):
        roop.globals.g_desired_face_analysis = ["detection", "landmark_3d_68", "landmark_2d_106", "recognition"]
    allowed_modules = roop.globals.g_desired_face_analysis
    print(f"Allowed modules: {allowed_modules}")

    if not hasattr(roop.globals, 'execution_providers'):
        roop.globals.execution_providers = ['CPUExecutionProvider']
    providers = roop.globals.execution_providers
    print(f"Execution providers: {providers}")
    
    try:
        # Disable auto-download by setting download=False
        face_analyser = insightface.app.FaceAnalysis(
            name="buffalo_l", 
            root=model_path, 
            providers=providers, 
            allowed_modules=allowed_modules,
            download=False  # This should prevent auto-download
        )
        face_analyser.prepare(ctx_id=0, det_size=(640, 640))
        print("Face analyser initialized successfully")
        return face_analyser
    except Exception as e:
        print(f"Error initializing face analyser: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def capture_screenshot(driver):
    screenshot = driver.get_screenshot_as_png()
    img = np.frombuffer(screenshot, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img

def verify_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File gambar sumber tidak ditemukan: {image_path}")
        return False
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Tidak dapat membaca file gambar: {image_path}")
        return False
    
    print(f"Gambar sumber berhasil dibaca. Ukuran: {img.shape}")
    return True

def display_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def display_image_with_faces(image_path, faces):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if faces:
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(img_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def main():
    # Initialize global configuration
    if roop.globals.CFG is None:
        print("Initializing global configuration...")
        roop.globals.CFG = Settings('config.yaml')

    # Set default values for required globals
    if not hasattr(roop.globals, 'g_desired_face_analysis'):
        roop.globals.g_desired_face_analysis = ["detection", "landmark_3d_68", "landmark_2d_106", "recognition"]
    
    if not hasattr(roop.globals, 'execution_providers'):
        roop.globals.execution_providers = ['CPUExecutionProvider']

    # Initialize face analyser
    face_analyser = init_face_analyser()
    if face_analyser is None:
        print("Failed to initialize face analyser. Exiting.")
        return

    # Gambar sumber
    face_source = 'face-simon-robben.jpg'
    
    # Verifikasi dan tampilkan gambar sumber
    if not verify_image(face_source):
        return
    print("Menampilkan gambar sumber. Silakan tutup jendela gambar untuk melanjutkan.")
    display_image(face_source)

    # Deteksi dan tampilkan wajah pada gambar sumber
    source_image = cv2.imread(face_source)
    print(f"Ukuran gambar sumber: {source_image.shape}")
    
    faces = get_all_faces(source_image)
    if faces is None:
        print("Peringatan: Tidak ada wajah yang terdeteksi pada gambar sumber!")
        return
    
    print(f"Jumlah wajah yang terdeteksi pada gambar sumber: {len(faces)}")
    print("Menampilkan gambar sumber dengan deteksi wajah. Silakan tutup jendela gambar untuk melanjutkan.")
    display_image_with_faces(face_source, faces)

    # Inisialisasi Selenium WebDriver dengan mode headless
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)

    # Buka website hanya sekali dalam mode headless
    driver.get('http://localhost:3000')
    time.sleep(2)

    # Buat direktori untuk menyimpan frame
    frames_dir = 'frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Pengaturan output video
    video_filename = 'output_video.mp4'
    fps = 10
    duration = 10
    total_frames = fps * duration

    # Ambil dan tampilkan screenshot pertama
    first_frame = capture_screenshot(driver)
    cv2.imwrite('first_screenshot.jpg', first_frame)
    print("Menampilkan screenshot pertama. Silakan tutup jendela gambar untuk melanjutkan.")
    display_image('first_screenshot.jpg')

    # Deteksi dan tampilkan wajah pada screenshot pertama
    target_faces = get_all_faces(first_frame)
    if target_faces is None:
        print("Peringatan: Tidak ada wajah yang terdeteksi pada screenshot!")
    else:
        print(f"Jumlah wajah yang terdeteksi pada screenshot: {len(target_faces)}")
        print("Menampilkan screenshot dengan deteksi wajah. Silakan tutup jendela gambar untuk melanjutkan.")
        display_image_with_faces('first_screenshot.jpg', target_faces)

    height, width, _ = first_frame.shape

    # Pengaturan penulis video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    # Inisialisasi face swapping
    roop.globals.source_path = face_source
    roop.globals.target_path = 'temp_target.jpg'
    roop.globals.output_path = 'temp_output.jpg'
    roop.globals.frame_processors = ['face_swapper']
    roop.globals.headless = True
    roop.globals.execution_threads = 1
    roop.globals.face_swap_mode = 'selected'

    # Ekstrak wajah sumber
    source_face_data = extract_face_images(face_source, (False, 0))
    if not source_face_data:
        print("Peringatan: Tidak ada wajah yang dapat diekstrak dari gambar sumber!")
        driver.quit()
        return
    print(f"Jumlah wajah yang berhasil diekstrak dari gambar sumber: {len(source_face_data)}")

    # Inisialisasi ProcessMgr
    process_options = ProcessOptions({"faceswap": {}}, 0.2, 1, 'selected', 0, None, None, 1, 0, False, False)
    process_mgr = ProcessMgr(None)
    process_mgr.initialize([face[0] for face in source_face_data], [], process_options)

    frame_count = 0
    while frame_count < total_frames:
        # Ambil screenshot tanpa memuat ulang halaman
        frame = capture_screenshot(driver)

        # Simpan frame sebagai gambar temporary
        cv2.imwrite(roop.globals.target_path, frame)

        # Ekstrak wajah target
        target_faces = get_all_faces(frame)
        if target_faces is None:
            print(f"Peringatan: Tidak ada wajah yang terdeteksi pada frame {frame_count}")
            swapped_frame = frame  # Gunakan frame asli jika tidak ada wajah terdeteksi
        else:
            print(f"Jumlah wajah yang terdeteksi pada frame {frame_count}: {len(target_faces)}")
            roop.globals.TARGET_FACES = target_faces

            # Proses frame (face swap)
            swapped_frame = process_mgr.process_frame(frame)
            
            if np.array_equal(frame, swapped_frame):
                print(f"Peringatan: Frame {frame_count} tidak berubah setelah face swap")

        # Simpan setiap frame sebagai gambar
        frame_filename = os.path.join(frames_dir, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frame_filename, swapped_frame)
        print(f"Menyimpan frame {frame_filename} (Frame {frame_count+1}/{total_frames})")

        # Tulis frame ke dalam video
        out.write(swapped_frame)

        # Tunggu untuk mempertahankan frame rate yang diinginkan
        time.sleep(1 / fps)
        frame_count += 1

    # Lepaskan penulis video dan tutup browser
    out.release()
    driver.quit()
    print(f"Video disimpan sebagai {video_filename}, dengan {total_frames} frame")

    # Bersihkan file temporary
    if os.path.exists(roop.globals.target_path):
        os.remove(roop.globals.target_path)
    if os.path.exists(roop.globals.output_path):
        os.remove(roop.globals.output_path)
    if os.path.exists('first_screenshot.jpg'):
        os.remove('first_screenshot.jpg')

if __name__ == "__main__":
    main()