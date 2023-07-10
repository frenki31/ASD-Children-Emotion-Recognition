import face_recognition
import os
import cv2
import numpy as np
import math
from keras.models import load_model
import tkinter as tk
from tkinter import simpledialog
from datetime import datetime
import pyodbc
import pandas as pd

server = 'DESKTOP-QT7MTFJ\SQLEXPRESS'
database = 'ACTIVITY'
username = 'sa'
password = '1234'

conn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')
if conn:
    print('Connection okay')
cursor = conn.cursor()
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


class Recognition:
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
        activity = simpledialog.askstring("Activity", f'{teacher.split()[0]}, what is the activity for today?')
        window.destroy()
        cursor.execute("SELECT COUNT(*) FROM TEACHER WHERE TEACH_FNAME = ? AND TEACH_LNAME = ?", teacher.split()[0], teacher.split()[1])
        if cursor.fetchone()[0] == 0:
            cursor.execute('INSERT INTO TEACHER(TEACH_FNAME, TEACH_LNAME) VALUES (?, ?)', (teacher.split()[0], teacher.split()[1]))
            # conn.commit()
        cursor.execute("SELECT COUNT(*) FROM ACTIVITY WHERE ACT_NAME = ?", activity)
        if cursor.fetchone()[0] == 0:
            cursor.execute('INSERT INTO ACTIVITY(ACT_NAME) VALUES (?)', activity)
            # conn.commit()
        start_time = datetime.now()
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
        end_time = datetime.now()

        # save a video record
        output_path = 'emotions_record.mp4'
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 60, size)

        for frame in self.frame_list:
            output.write(frame)
        output.release()

        # Save people's emotions to a file
        file_path = 'people_emotions.txt'
        emotion_with_max_count = {}
        with open(file_path, 'w') as file:

            line0 = f'Today on {start_time:%d/%m/%Y} teacher {teacher} taught the children: "{activity}".\n' \
                    f'The session began at {start_time:%H:%M:%S} and ended at {end_time:%H:%M:%S}.\n'
            file.write(f'{line0}\n')
            for person, person_emotions in self.people_emotions.items():
                if person == "Unknown":
                    count_unknown = simpledialog.askinteger("Unknown Counter","How many children are not known by the system?")
                    if count_unknown == 1:
                        person = simpledialog.askstring('Unknown Child',"Who is the unknown person?")
                    else:
                        person = "Many children"
                # Check if person exists in the table
                cursor.execute("SELECT COUNT(*) FROM CHILD WHERE CHILD_FNAME=? AND CHILD_LNAME=?", person.split()[0], person.split()[1])
                if cursor.fetchone()[0] == 0:
                    # Person doesn't exist, perform INSERT
                    insert_query = "INSERT INTO CHILD (CHILD_FNAME, CHILD_LNAME) VALUES (?, ?)"
                    cursor.execute(insert_query, (person.split()[0], person.split()[1]))
                    # conn.commit()
                emotions_count = ', '.join([f'{emotion} {count} times' for emotion, count in person_emotions.items()])
                if person == "Many children":
                    line = f'{person} were {emotions_count}.'
                else:
                    line = f'{person} was {emotions_count}.'
                if person_emotions:
                    max_emotion_count = max(person_emotions.values())
                    max_emotions = [emotion for emotion, count in person_emotions.items() if count == max_emotion_count]
                    emotion_with_max_count[person] = ', '.join(max_emotions)
                else:
                    file.write(f'{person} did not display any emotions while doing: {activity}\n')
                file.write(f'{line}\n')
            like_activity = [person for person, emotion in emotion_with_max_count.items() if emotion in ["Happy", "Surprised"]]
            do_not_like_activity = [person for person, emotion in emotion_with_max_count.items() if emotion in ["Sad", "Fearful", "Angry", "Disgusted"]]
            might_like_activity = [person for person, emotion in emotion_with_max_count.items() if emotion in ["Neutral"]]
            many_emotions = [person for person, emotion in emotion_with_max_count.items() if len(emotion.split(',')) > 1]
            stored_procedure = 'EXEC SP_INSERT_INTO_ACT_TEACH_CHILD @act_name=?, @teach_fname=?, @teach_lname=?, @child_fname=?, @child_lname=?, @act_st=?, @act_et=?, @like=?'
            if like_activity:
                file.write(f'\nThe children who like to do "{activity}": {", ".join(like_activity)}.\n')
                cursor.execute(stored_procedure, activity, teacher.split()[0],teacher.split()[1], person.split()[0], person.split()[1], start_time, end_time, "Likes")
            if do_not_like_activity:
                file.write(f'\nThe children who do not like to do "{activity}": {", ".join(do_not_like_activity)}.\n')
                cursor.execute(stored_procedure, activity, teacher.split()[0],teacher.split()[1], person.split()[0], person.split()[1], start_time, end_time, "Does not like")
            if might_like_activity:
                file.write(f'\nThe children who might like to do "{activity}": {", ".join(might_like_activity)}.\n')
                cursor.execute(stored_procedure, activity, teacher.split()[0],teacher.split()[1], person.split()[0], person.split()[1], start_time, end_time, "Might like")
            if many_emotions:
                file.write(f'\nThe children who had many different emotions while doing "{activity}": {", ".join(many_emotions)}.\n')
                cursor.execute(stored_procedure, activity, teacher.split()[0],teacher.split()[1], person.split()[0], person.split()[1], start_time, end_time, "Not sure")
            conn.commit()
            cursor.close()

            question = simpledialog.askstring('Question', 'Do you want to have your all-time report? (y/n)')
            if question in ['y','Y','Yes','yes']:
                query = f"EXEC SP_TEACHER_REPORT @teach_fname = '{teacher.split()[0]}', @teach_lname='{teacher.split()[1]}'"
                df = pd.read_sql(query, conn)
                table = f'{df.to_string(index=False)}\n'
                with open('report.txt', 'w') as file:
                    file.write(table)
                    question1 = simpledialog.askstring('Question', 'Do you want the children names in your all-time report? (y/n)')
                    if question1 in ['y','Y','Yes','yes']:
                        query1 = f"EXEC SP_CHILDREN_REPORT @teach_fname = '{teacher.split()[0]}', @teach_lname = '{teacher.split()[1]}'"
                        df1 = pd.read_sql(query1, conn)
                        table1 = f'\n{df1.to_string(index=False)}'
                        file.write(table1)
            conn.close()
        print(f"Video saved to {output_path}")
        print(f"Emotion information saved to {file_path}")


if __name__ == "__main__":
    rec = Recognition()
    rec.run_recognition()