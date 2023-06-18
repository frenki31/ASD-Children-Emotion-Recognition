import face_recognition
import os
import cv2
import numpy as np
import math
from keras.models import load_model

model = load_model(r'C:\Users\user\PycharmProjects\emotionRecognition\emotion_detection_model.h5')
emotions = ['Angry','Disgusted','Fearful','Happy','Neutral', 'Sad', 'Surprised']


def face_matching(face_distance, face_match=0.6):
    range = (1.0 - face_match)
    linear_value = (1.0-face_distance) / (range * 2.0)

    if face_distance > face_match:
        return str(round(linear_value*100, 2)) + '%'
    else:
        value = (linear_value+((1.0-linear_value) * math.pow((linear_value - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        folders = os.listdir('images')
        for folder in folders:
            folder_path = os.path.join('images', folder)

            if os.path.isdir(folder_path):
                # If this path is correct then find the directory of that path
                folder_items = os.listdir(folder_path)

                for image in folder_items:
                    image_path = os.path.join(folder_path, image)
                    img = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(img)
                    if len(encodings) > 0:
                        # if there is more than one face
                        encoding = encodings[0]
                        self.known_face_encodings.append(encoding)
                        self.known_face_names.append(folder)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print('Video source not found...')

        while True:
            _, frame = video_capture.read()
            frame = cv2.flip(frame, 1)

            if self.current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_matching(face_distances[best_match_index])
                    self.face_names.append(f'{name} ({confidence})')
            self.current_frame = not self.current_frame

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 3
                right *= 4
                bottom *= 4
                left *= 3

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.rectangle(frame, (left, top), (right, top + 35), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                gray_small_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2GRAY)
                input_image = cv2.resize(gray_small_frame, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([input_image]) != 0:
                    interest = input_image.astype('float')/255.0
                    interest = np.asarray(interest)
                    interest = np.expand_dims(interest, axis=0)

                    prediction = model.predict(interest)
                    label = emotions[np.argmax(prediction)]
                    cv2.putText(frame, label, (left+6, top+20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, "No face", (left+6, top+20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    fr = FaceRecognition()
    fr.run_recognition()