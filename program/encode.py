import pickle
import os
import face_recognition

known_face_names = []
known_face_encodings = []

folders = os.listdir('../images')
for folder in folders:
    folder_path = os.path.join('../images', folder)

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
                known_face_encodings.append(encoding)
                known_face_names.append(folder)


with open('../model/encoding.pickle','wb') as f:
    pickle.dump(known_face_names, f)
    pickle.dump(known_face_encodings, f)