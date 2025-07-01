import cv2
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime, timedelta

# Paths to the folders
main_save_path = r"E:\DLC project paper\Webcam\pythonProject\captured_images"
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
    mtcnn = MTCNN()
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    for dir_name in os.listdir(folder_path):
        dir_path = os.path.join(folder_path, dir_name)
        if os.path.isdir(dir_path):
            for file_name in os.listdir(dir_path):
                if file_name.endswith(".jpg") or file_name.endswith(".png"):
                    image_path = os.path.join(dir_path, file_name)
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face = mtcnn(image)
                    if face is not None:
                        face_encoding = resnet(face.unsqueeze(0))
                        known_face_encodings.append(face_encoding.detach().numpy().flatten())
                        known_face_names.append(dir_name)
    return known_face_encodings, known_face_names

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    mtcnn = MTCNN()
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    today_str = datetime.today().strftime('%Y-%m-%d')

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (160, 160))  # Resize the face to match FaceNet input size
        face_resized_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        detected_face = mtcnn(face_resized_rgb)

        name = "Unknown"
        color = (0, 0, 255)  # Red for unrecognized faces

        if detected_face is not None:
            face_encoding = resnet(detected_face.unsqueeze(0))

            if known_face_encodings:  # Ensure there are known face encodings to compare with
                # Calculate similarities
                similarities = [np.inner(face_encoding.detach().numpy().flatten(), known_encoding) for known_encoding in known_face_encodings]
                best_match_index = np.argmax(similarities)
                best_match_similarity = similarities[best_match_index]

                # Set a threshold for recognizing a match
                if best_match_similarity > 0.5:  # Adjust the threshold as needed
                    name = known_face_names[best_match_index]
                    color = (0, 255, 0)  # Green for recognized faces

                    # Check if an image for today already exists
                    user_folder_path = os.path.join(main_save_path, name)
                    if not any(fname.startswith(today_str) for fname in os.listdir(user_folder_path)):
                        image_name = os.path.join(user_folder_path, f"{today_str}_{name}.jpg")
                        cv2.imwrite(image_name, frame)
                else:
                    # Save the unknown user's image if not already captured in this session
                    if name not in unknown_faces_captured:
                        if not os.path.exists(unknown_save_path):
                            os.makedirs(unknown_save_path)
                        image_counter = len(os.listdir(unknown_save_path)) + 1
                        image_name = os.path.join(unknown_save_path, f"{today_str}_Unknown_{image_counter}.jpg")
                        cv2.imwrite(image_name, frame)
                        unknown_faces_captured.add(name)
            else:
                # Save the unknown user's image if not already captured in this session
                if name not in unknown_faces_captured:
                    if not os.path.exists(unknown_save_path):
                        os.makedirs(unknown_save_path)
                    image_counter = len(os.listdir(unknown_save_path)) + 1
                    image_name = os.path.join(unknown_save_path, f"{today_str}_Unknown_{image_counter}.jpg")
                    cv2.imwrite(image_name, frame)
                    unknown_faces_captured.add(name)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    return frame

if __name__ == "__main__":
    # Load known faces
    known_face_encodings, known_face_names = load_known_faces(main_save_path)
    known_face_encodings_unknown, known_face_names_unknown = load_known_faces(unknown_save_path)

    known_face_encodings.extend(known_face_encodings_unknown)
    known_face_names.extend(known_face_names_unknown)

    # Initialize the webcam
    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Video', cv2.WND_PROP_TOPMOST, 1)  # Make the window topmost
    cv2.moveWindow('Video', 450, 150)  # Move the window to a specific position
    print("Face recognition started. Press 'q' to quit")

    # Set to keep track of unknown faces captured in this session
    unknown_faces_captured = set()

    while True:
        # Delete old images before capturing new ones
        delete_old_images(main_save_path)
        delete_old_images(unknown_save_path)

        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Recognize faces in the frame
        frame = recognize_faces(frame, known_face_encodings, known_face_names, unknown_faces_captured)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()
