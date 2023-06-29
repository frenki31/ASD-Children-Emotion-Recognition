import face_recognition
import os
import cv2
import numpy as np
import math
from keras.models import load_model
import tkinter as tk
from tkinter import simpledialog
from datetime import datetime

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
    people_emotions = {}
    current_frame = True
    frame_list = []

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
        window = tk.Tk()
        window.withdraw()
        teacher = simpledialog.askstring("Teacher", "Teacher, can you please introduce yourself?")
        activity = simpledialog.askstring("Activity", f'{teacher}, what is the activity for today?')
        window.destroy()
        video_capture = cv2.VideoCapture('video.mp4')
        video_capture.set(cv2.CAP_PROP_FPS, 60)

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
                self.face_emotions = []
                for encoding, location in zip(self.face_encodings, self.face_locations):
                    matches = face_recognition.compare_faces(self.known_face_encodings, encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_matching(face_distances[best_match_index])
                    self.face_names.append(f'{name} ({confidence})')

                    top, right, bottom, left = location
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    face_region = frame[top:bottom, left:right]
                    gray_face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                    input_image = cv2.resize(gray_face_region, (48, 48), interpolation=cv2.INTER_AREA)

                    if np.sum([input_image]) != 0:
                        interest = input_image.astype('float') / 255.0
                        interest = np.asarray(interest)
                        interest = np.expand_dims(interest, axis=0)

                        prediction = model.predict(interest)
                        label = emotions[np.argmax(prediction)]
                    else:
                        label = 'No face'
                    self.face_emotions.append(label)

            self.current_frame = not self.current_frame

            for name, emotion in zip(self.face_names, self.face_emotions):
                # get only the name
                name_without_confidence = " ".join(name.split(" ")[:-1])
                # count the occurrences of people's emotions
                if name_without_confidence not in self.people_emotions:
                    self.people_emotions[name_without_confidence] = {}
                if emotion not in self.people_emotions[name_without_confidence]:
                    self.people_emotions[name_without_confidence][emotion] = 1
                self.people_emotions[name_without_confidence][emotion] += 1

            for (top, right, bottom, left), name, emotion in zip(self.face_locations, self.face_names, self.face_emotions):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.rectangle(frame, (left, top), (right, top + 35), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                cv2.putText(frame, emotion, (left + 6, top + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                self.frame_list.append(frame)
                height, width, colors = frame.shape
                size = (width, height)

            cv2.imshow('Face recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

        # save a video record
        output_path = 'emotions_record.mp4'
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, size)

        for frame in self.frame_list:
            output.write(frame)
        output.release()

        # Save people's emotions to a file
        file_path = 'people_emotions.txt'
        emotion_with_max_count = {}
        with open(file_path, 'w') as file:
            now = datetime.now()
            file.write(f'Today on {now:%d/%m/%Y} teacher {teacher} taught the children: "{activity}"\n\n')
            for person, person_emotions in self.people_emotions.items():
                emotions_count = ', '.join([f'{emotion} {count} times' for emotion, count in person_emotions.items()])
                line = f'{person} was {emotions_count}\n'
                if person_emotions:
                    max_emotion_count = max(person_emotions.values())
                    max_emotions = [emotion for emotion, count in person_emotions.items() if count == max_emotion_count]
                    emotion_with_max_count[person] = ', '.join(max_emotions)
                    if len(max_emotions) == 1:
                        if emotion_with_max_count[person] in ["Happy","Surprised"]:
                            line2 = f'{person} was {emotion_with_max_count[person]} most of the time. {person} likes "{activity}".\n'
                        elif emotion_with_max_count[person] in ["Sad", "Angry", "Disgusted", "Fearful"]:
                            line2 = f'{person} was {emotion_with_max_count[person]} most of the time. {person} does not like "{activity}".\n'
                        elif emotion_with_max_count[person] == "Neutral":
                            line2 = f'{person} was {emotion_with_max_count[person]} most of the time. {person} might like "{activity}".\n'
                    else:
                        line2 = f'{person} had a mix of emotions while doing: {activity}\n'
                else:
                    line2 = f'{person} did not display any emotions while doing: {activity}\n'
                file.write(f'{line}{line2}\n')

        print(f"Video saved to {output_path}")
        print(f"Emotion information saved to {file_path}")


if __name__ == "__main__":
    fr = FaceRecognition()
    fr.run_recognition()
