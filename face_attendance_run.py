import cv2
import os
import pandas as pd
import time
import numpy as np
from datetime import datetime
from deepface import DeepFace
import pickle

DATASET = "dataset"
ATT_FILE = "attendance/attendance.xlsx"
SIMILARITY_THRESHOLD = 0.55
MODEL_NAME = "Facenet"
CACHE_FILE = "embeddings_cache.pkl"

def is_student_folder(name):
    if not name or name.startswith("."):
        return False
    if name.endswith(".pkl") or "ds_model" in name:
        return False
    return True

def get_dataset_hash():
    """Get hash of dataset folder to detect changes"""
    import hashlib
    hash_str = ""
    for user in sorted(os.listdir(DATASET)):
        user_path = os.path.join(DATASET, user)
        if not os.path.isdir(user_path) or not is_student_folder(user):
            continue
        for img in sorted(os.listdir(user_path)):
            if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                hash_str += f"{user}/{img}{os.path.getmtime(os.path.join(user_path, img))}"
    return hashlib.md5(hash_str.encode()).hexdigest()

def load_embeddings():
    """Load embeddings from cache or generate new"""
    # Always load model first to avoid delay during recognition
    print(f"Loading {MODEL_NAME} model...")
    DeepFace.build_model(MODEL_NAME)
    
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
            current_hash = get_dataset_hash()
            if cache.get('hash') == current_hash:
                print(f"Loaded {len(cache['embeddings'])} embeddings from cache")
                return np.array(cache['embeddings']), cache['names']
        except:
            pass
    
    print("Encoding dataset faces...")
    
    known_embeddings = []
    known_names = []
    
    if not os.path.exists(DATASET):
        print("Dataset folder missing")
        exit()
    
    for user in os.listdir(DATASET):
        user_path = os.path.join(DATASET, user)
        if not os.path.isdir(user_path) or not is_student_folder(user):
            continue
        for img in os.listdir(user_path):
            img_path = os.path.join(user_path, img)
            if not img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            try:
                rep = DeepFace.represent(
                    img_path=img_path,
                    model_name=MODEL_NAME,
                    enforce_detection=False,
                    align=True
                )[0]["embedding"]
                known_embeddings.append(rep)
                known_names.append(user)
            except:
                continue
    
    # Save to cache
    cache = {
        'hash': get_dataset_hash(),
        'embeddings': known_embeddings,
        'names': known_names
    }
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    
    return np.array(known_embeddings), known_names

known_embeddings, known_names = load_embeddings()

if len(known_embeddings) == 0:
    print("No faces in dataset")
    exit()

known_embeddings = np.array(known_embeddings)
print(f"Total embeddings loaded: {len(known_embeddings)}")

proto_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"

dnn_detector = None
try:
    if os.path.exists(proto_path) and os.path.exists(model_path):
        dnn_detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        print("Using DNN Face Detector (High Accuracy)")
except:
    pass

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Camera failed")
    exit()

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Camera Started (SMART MULTI FACE MODE - Optimized)")
print("Resolution: 640x480 | Target FPS: 30")

detection_window = 30
required_presence = 10
start_time = time.time()
presence_timer = {}
detected_students = set()
unknown_face_found = False
frame_count = 0
process_every_n = 3
label_cache = {}
cache_timeout = 2.0

fps_counter = 0
fps_time = time.time()
current_fps = 0

today_str = datetime.now().strftime("%Y-%m-%d")
info_line = f"Date: {today_str}  |  ESC to finish"
while time.time() - start_time < detection_window:

    ret, frame = cam.read()
    if not ret:
        continue

    # Calculate FPS
    fps_counter += 1
    if time.time() - fps_time >= 1.0:
        current_fps = fps_counter
        fps_counter = 0
        fps_time = time.time()

    frame_count += 1
    process_frame = (frame_count % process_every_n == 0)

    # Clear old cache entries
    current_time = time.time()
    label_cache = {k: v for k, v in label_cache.items() 
                   if current_time - v[2] < cache_timeout}

    # Low-light enhancement
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 100:
        # Strong enhancement for low light
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Stronger CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        # Additional brightness and contrast boost
        l = cv2.convertScaleAbs(l, alpha=1.5, beta=30)
        enhanced_frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        # Gamma correction
        gamma = 0.6
        lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced_frame = cv2.LUT(enhanced_frame, lookup_table)
    else:
        enhanced_frame = frame
    
    faces = []
    if dnn_detector is not None:
        h, w = enhanced_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(enhanced_frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        dnn_detector.setInput(blob)
        detections = dnn_detector.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # Lower threshold for low light
            threshold = 0.3 if mean_brightness < 100 else 0.5
            if confidence > threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                faces.append((x1, y1, x2 - x1, y2 - y1))
    else:
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        # More sensitive detection in low light
        scale = 1.05 if mean_brightness < 100 else 1.1
        neighbors = 2 if mean_brightness < 100 else 3
        faces = face_detector.detectMultiScale(gray, scaleFactor=scale,
                                               minNeighbors=neighbors, minSize=(50, 50))

    for (x,y,w,h) in faces:
        cx, cy = x + w // 2, y + h // 2
        cache_key = (cx // 50, cy // 50)
        name = "Unknown"
        color = (0,0,255)

        if cache_key in label_cache and not process_frame:
            cached = label_cache[cache_key]
            name = cached[0]
            color = cached[1]
        else:
            # Use enhanced frame for recognition in low light
            face_img = enhanced_frame[y:y+h, x:x+w]
            
            # Additional enhancement for face recognition in low light
            if mean_brightness < 100:
                face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                face_gray = cv2.equalizeHist(face_gray)
                face_img = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
            
            try:
                if process_frame:
                    rep = DeepFace.represent(
                        img_path=face_img,
                        model_name=MODEL_NAME,
                        enforce_detection=False,
                        align=True
                    )[0]["embedding"]
                    emb = np.array(rep)
                    dots = np.dot(known_embeddings, emb)
                    norms = np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(emb)
                    similarities = dots / norms
                    best_index = np.argmax(similarities)
                    best_score = similarities[best_index]
                    
                    # Lower threshold for low light recognition
                    threshold = SIMILARITY_THRESHOLD - 0.15 if mean_brightness < 100 else SIMILARITY_THRESHOLD
                    if best_score > threshold:
                        name = known_names[best_index]
                        color = (0,255,0)
                        name = f"{name} ({best_score:.2f})"
                        if known_names[best_index] not in presence_timer:
                            presence_timer[known_names[best_index]] = time.time()
                        stay_time = time.time() - presence_timer[known_names[best_index]]
                        if stay_time >= required_presence:
                            detected_students.add(known_names[best_index])
                    else:
                        name = "Unknown"
                        unknown_face_found = True
                    label_cache[cache_key] = (name, color, current_time)
                else:
                    if cache_key in label_cache:
                        cached = label_cache[cache_key]
                        name = cached[0]
                        color = cached[1]
            except Exception as e:
                if cache_key in label_cache:
                    cached = label_cache[cache_key]
                    name = cached[0]
                    color = cached[1]

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,name,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    fps_text = f"FPS: {current_fps}"
    cv2.putText(frame, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, info_line, (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Smart Face Attendance - Optimized", frame)
    if cv2.waitKey(1)==27:
        break

cam.release()
cv2.destroyAllWindows()

if len(detected_students) == 0:
    print("No Presents")

if unknown_face_found:
    print("New face detected")

today = datetime.now().strftime("%Y-%m-%d")
os.makedirs("attendance", exist_ok=True)

if os.path.exists(ATT_FILE):
    df = pd.read_excel(ATT_FILE)
else:
    df = pd.DataFrame(columns=["Reg_No", "Gmail", "Phone"])

for base_col in ["Reg_No", "Gmail", "Phone"]:
    if base_col not in df.columns:
        df[base_col] = ""

if today not in df.columns:
    df[today] = "Absent"

df = df[df["Reg_No"].apply(lambda x: is_student_folder(str(x)) if pd.notna(x) else True)]

for user in os.listdir(DATASET):
    user_path = os.path.join(DATASET, user)
    if not os.path.isdir(user_path) or not is_student_folder(user):
        continue
    if user not in df["Reg_No"].values:
        row = {"Reg_No": user, "Gmail": "", "Phone": ""}
        for col in df.columns:
            if col not in row:
                row[col] = "Absent"
        df.loc[len(df)] = row

for user in detected_students:
    if df.loc[df["Reg_No"]==user, today].values[0] == "Present":
        print(user, "already marked")
    else:
        print(user, "attendance marked")
        df.loc[df["Reg_No"]==user, today] = "Present"

df.to_excel(ATT_FILE,index=False)
print("Process Completed Successfully")
