import os
import pickle
import torch
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
import dlib
from datetime import datetime

# Set up face recognition model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Set up image transforms
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create a dictionary to store face embeddings and names
face_database = {}

# Set the dataset directory
dataset_dir = 'dataset'

# Check if the dataset directory exists
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
    print("The dataset directory did not exist, so it was created.")
else:
    print("The dataset directory already exists.")

# Load existing face database if available
if os.path.exists('face_database.pkl'):
    with open('face_database.pkl', 'rb') as f:
        face_database = pickle.load(f)
    print("Loaded existing face database.")

# Set up face detection function using dlib
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)


def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces


def align_face(frame, landmarks):
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    M = cv2.getRotationMatrix2D(eye_center, angle, 1)
    aligned_face = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    return aligned_face


def is_user_already_registered(face_embedding):
    threshold = 0.5  # Tighter threshold to handle different expressions
    if face_embedding is None:
        return False

    for name, db_embedding in face_database.items():
        if db_embedding is not None:
            distance = np.linalg.norm(face_embedding - db_embedding)
            print(f"Distance to {name}: {distance}")
            if distance < threshold:
                return True
    return False


def save_user_image(name):
    cap = cv2.VideoCapture(0)
    saved = False

    while not saved:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        faces = detect_faces(frame)

        # Check if only one face is detected
        if len(faces) == 1:
            face = faces[0]
            # Get the facial landmarks
            landmarks = predictor(frame, face)

            # Align the face
            aligned_face = align_face(frame, landmarks)

            # Draw landmarks on the face for visualization (optional)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(aligned_face, (x, y), 1, (255, 0, 0), -1)

            # Extract the face ROI
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_roi = aligned_face[y:y + h, x:x + w]

            # Preprocess the face ROI
            face_pil = Image.fromarray(face_roi)
            face_tensor = transform(face_pil).unsqueeze(0)
            with torch.no_grad():
                face_embedding = resnet(face_tensor).detach().numpy()

            # Check if the face is already registered
            if is_user_already_registered(face_embedding):
                print("This face is already registered. Registration not allowed.")
                cv2.putText(frame, "This face is already registered. Registration not allowed.", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Register User', frame)
                cv2.waitKey(2000)  # Display message for 2 seconds
                break
            else:
                # Display the output with a countdown
                for i in range(3, 0, -1):
                    frame_copy = frame.copy()
                    cv2.putText(frame_copy, f"Capturing in {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Register User', frame_copy)
                    cv2.waitKey(1000)

                # Save the face ROI in the user's folder with the date
                folder_name = os.path.join(dataset_dir, name)
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                img_name = os.path.join(folder_name, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(img_name, face_roi)
                print(f"Image saved as {img_name}")

                # Save the face embedding to the database
                face_database[name] = face_embedding

                # Save the updated face database
                with open('face_database.pkl', 'wb') as f:
                    pickle.dump(face_database, f)
                print(f"Face database updated with {name}'s embedding.")

                saved = True

        elif len(faces) > 1:
            cv2.putText(frame, "Multiple faces detected. Please ensure only one face is in the frame.", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            cv2.putText(frame, "No face detected. Please ensure your face is clearly visible.", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the output
        cv2.imshow('Register User', frame)

        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# Input name for registration
user_name = input("Enter your name for registration: ")

# Check if the user is already registered
if is_user_already_registered(face_database.get(user_name, None)):
    print("This face is already registered. Registration not allowed.")
else:
    # Capture and save user image
    save_user_image(user_name)
