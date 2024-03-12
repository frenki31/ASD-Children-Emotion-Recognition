import math

def face_matching(face_distance, face_match=0.6):
    match_range = (1.0 - face_match)
    linear_value = (1.0-face_distance) / (match_range * 2.0)

    if face_distance > face_match:
        return str(round(linear_value*100, 2)) + '%'
    else:
        value = (linear_value+((1.0-linear_value) * math.pow((linear_value - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

def emotion_analysis(face_region, cv2, np, model, emotions):
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
    return label

def analyze_faces(face_encodings, face_locations, frame, cv2, np, model, emotions):
    face_emotions = []

    for encoding, location in zip(face_encodings, face_locations):
        top, right, bottom, left = location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        face_region = frame[top:bottom, left:right]
        emotion = emotion_analysis(face_region, cv2, np, model, emotions)
        face_emotions.append(emotion)

    return face_emotions