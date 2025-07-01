import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from datetime import datetime, timedelta
from retinaface import RetinaFace

# Paths to the folders
main_save_path = r"E:\DLC project paper\Webcam\pythonProject\dataset"
unknown_save_path = r"E:\DLC project paper\Webcam\pythonProject\unknown_faces"

# Create the main directory if it doesn't exist
if not os.path.exists(main_save_path):
    os.makedirs(main_save_path)

# Create the unknown faces directory if it doesn't exist
if not os.path.exists(unknown_save_path):
    os.makedirs(unknown_save_path)

# Function to load and encode known faces from a specified folder
def load_known_faces(folder_path):
    known_face_encodings = []
    known_face_names = []
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    for dir_name in os.listdir(folder_path):
        dir_path = os.path.join(folder_path, dir_name)
        if os.path.isdir(dir_path):
            for file_name in os.listdir(dir_path):
                if file_name.endswith(".jpg") or file_name.endswith(".png"):
                    image_path = os.path.join(dir_path, file_name)
                    image = cv2.imread(image_path)
                    face_encoding = get_face_encoding(image, resnet)
                    if face_encoding is not None:
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(dir_name)
                        print(f"Loaded encoding for {dir_name} from {file_name}")
    return known_face_encodings, known_face_names

def get_face_encoding(image, resnet):
    faces = RetinaFace.detect_faces(image)
    if isinstance(faces, dict) and 'face_1' in faces:
        face = faces['face_1']
        bbox = face['facial_area']
        aligned_face = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        aligned_face = cv2.resize(aligned_face, (160, 160))
        aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        face_tensor = torch.tensor(aligned_face, dtype=torch.float32).permute(2, 0, 1)
        face_tensor = (face_tensor / 255.0 - 0.5) / 0.5
        face_tensor = face_tensor.unsqueeze(0)
        with torch.no_grad():
            face_encoding = resnet(face_tensor).numpy().flatten()
        return face_encoding
    return None

# Function to delete images older than 60 days
def delete_old_images(folder_path, days=60):
    cutoff_date = datetime.today() - timedelta(days=days)
    for dir_name in os.listdir(folder_path):
        dir_path = os.path.join(folder_path, dir_name)
        if os.path.isdir(dir_path):
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                file_date = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_date < cutoff_date:
                    os.remove(file_path)
                    print(f"Deleted {file_path}")

# Function to recognize faces in a frame
def recognize_faces(frame, known_face_encodings, known_face_names, unknown_faces_captured):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    today_str = datetime.today().strftime('%Y-%m-%d')
    faces = RetinaFace.detect_faces(frame)
    if isinstance(faces, dict):
        for face_key in faces.keys():
            face = faces[face_key]
            bbox = face['facial_area']
            aligned_face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            aligned_face = cv2.resize(aligned_face, (160, 160))
            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            face_tensor = torch.tensor(aligned_face, dtype=torch.float32).permute(2, 0, 1)
            face_tensor = (face_tensor / 255.0 - 0.5) / 0.5
            face_tensor = face_tensor.unsqueeze(0)
            with torch.no_grad():
                face_encoding = resnet(face_tensor).numpy().flatten()
            name = "Unknown"
            color = (0, 0, 255)
            if known_face_encodings:
                similarities = [np.dot(face_encoding, known_encoding) for known_encoding in known_face_encodings]
                best_match_index = np.argmax(similarities)
                best_match_similarity = similarities[best_match_index]
                if best_match_similarity > 0.5:
                    name = known_face_names[best_match_index]
                    color = (0, 255, 0)
                    user_folder_path = os.path.join(main_save_path, name)
                    if not any(fname.startswith(today_str) for fname in os.listdir(user_folder_path)):
                        image_name = os.path.join(user_folder_path, f"{today_str}_{name}.jpg")
                        cv2.imwrite(image_name, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
                else:
                    if name not in unknown_faces_captured:
                        image_counter = len(os.listdir(unknown_save_path)) + 1
                        image_name = os.path.join(unknown_save_path, f"{today_str}_Unknown_{image_counter}.jpg")
                        cv2.imwrite(image_name, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
                        unknown_faces_captured.add(name)
            else:
                if name not in unknown_faces_captured:
                    image_counter = len(os.listdir(unknown_save_path)) + 1
                    image_name = os.path.join(unknown_save_path, f"{today_str}_Unknown_{image_counter}.jpg")
                    cv2.imwrite(image_name, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
                    unknown_faces_captured.add(name)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return frame

if __name__ == "__main__":
    known_face_encodings, known_face_names = load_known_faces(main_save_path)
    print(f"Loaded {len(known_face_encodings)} known faces.")
    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Video', cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow('Video', 450, 150)
    print("Face recognition started. Press 'q' to quit")
    unknown_faces_captured = set()
    while True:
        delete_old_images(main_save_path)
        delete_old_images(unknown_save_path)
        ret, frame = video_capture.read()
        frame = recognize_faces(frame, known_face_encodings, known_face_names, unknown_faces_captured)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
