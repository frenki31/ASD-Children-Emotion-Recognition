import pickle
import threading
import face_recognition
import cv2
import numpy as np
from keras.models import load_model
import tkinter as tk
from tkinter import simpledialog
from datetime import datetime
import database
from file import reports
from analysis import face_matching, analyze_faces

model = load_model(r'../model/emotion_detection_model.h5')
emotions = ['Angry','Disgusted','Fearful','Happy','Neutral', 'Sad', 'Surprised']

class Recognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    people_emotions = {}
    current_frame = True
    frame_list = []
    size = None
    face_emotions = []

    def __init__(self):
        with open('../model/encoding.pickle', 'rb') as f:
            self.known_face_names = pickle.load(f)
            self.known_face_encodings = pickle.load(f)

    def recognition_loop(self, video_capture):
        while True:
            _, frame = video_capture.read()
            frame = cv2.flip(frame, 1)

            if self.current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                self.face_emotions = analyze_faces(self.face_encodings, self.face_locations, frame, cv2, np, model, emotions)

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
                self.size = (width, height)

            cv2.imshow('Face recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


    def run_recognition(self):
        window = tk.Tk()
        window.withdraw()
        teacher = simpledialog.askstring("Teacher", "Teacher, can you please introduce yourself?")
        activity = simpledialog.askstring("Activity", f'{teacher.split()[0]}, what is the activity for today?')
        window.destroy()
        database.enter_teacher(teacher)
        database.enter_activity(activity)
        start_time = datetime.now()
        video_capture = cv2.VideoCapture("../videos/ASD.mp4")

        if not video_capture.isOpened():
            print('Video source not found...')

        recognition_thread = threading.Thread(target=self.recognition_loop, args=(video_capture, ))
        recognition_thread.start()
        recognition_thread.join()

        end_time = datetime.now()

        reports(cv2, self.size, self.frame_list, start_time, teacher, activity, end_time, self.people_emotions, simpledialog)

if __name__ == "__main__":
    rec = Recognition()
    rec.run_recognition()