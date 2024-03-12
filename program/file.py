import os
import database

def reports(cv2, size, frame_list, start_time, teacher, activity, end_time, people_emotions, simpledialog):
    path = '../report/'
    # save a video record
    output_path = 'emotions_record.mp4'
    output = cv2.VideoWriter(os.path.join(path, output_path), cv2.VideoWriter_fourcc(*"mp4v"), 60, size)

    for frame in frame_list:
        output.write(frame)
    output.release()

    # Save people's emotions to a file
    file_path = 'people_emotions.txt'
    emotion_with_max_count = {}
    with open(os.path.join(path, file_path), 'w') as file:

        line0 = f'Today on {start_time:%d/%m/%Y} teacher {teacher} taught the children: "{activity}".\n' \
                f'The session began at {start_time:%H:%M:%S} and ended at {end_time:%H:%M:%S}.\n'
        file.write(f'{line0}\n')
        for person, person_emotions in people_emotions.items():
            if person == "Unknown":
                count_unknown = simpledialog.askinteger("Unknown Counter",
                                                        "How many children are not known by the system?")
                for i in range(count_unknown):
                    person = simpledialog.askstring('Unknown Child', "Who is the unknown person?")
                # else:
                #     person = "Many children"
            # Check if person exists in the table
            database.enter_person(person)
            emotions_count = ', '.join([f'{emotion} {count} times' for emotion, count in person_emotions.items()])
            # if person == "Many children":
            #     line = f'{person} were {emotions_count}.'
            # else:
            line = f'{person} was {emotions_count}.'
            if person_emotions:
                max_emotion_count = max(person_emotions.values())
                max_emotions = [emotion for emotion, count in person_emotions.items() if count == max_emotion_count]
                emotion_with_max_count[person] = ', '.join(max_emotions)
            else:
                file.write(f'{person} did not display any emotions while doing: {activity}\n')
            file.write(f'{line}\n')
        like_activity = [person for person, emotion in emotion_with_max_count.items() if
                         emotion in ["Happy", "Surprised"]]
        do_not_like_activity = [person for person, emotion in emotion_with_max_count.items() if
                                emotion in ["Sad", "Fearful", "Angry", "Disgusted"]]
        might_like_activity = [person for person, emotion in emotion_with_max_count.items() if emotion in ["Neutral"]]
        many_emotions = [person for person, emotion in emotion_with_max_count.items() if len(emotion.split(',')) > 1]
        if like_activity:
            file.write(f'\nThe children who like to do "{activity}": {", ".join(like_activity)}.\n')
            database.insert_emotions(activity, teacher, person, start_time, end_time, "Likes")
        if do_not_like_activity:
            file.write(f'\nThe children who do not like to do "{activity}": {", ".join(do_not_like_activity)}.\n')
            database.insert_emotions(activity, teacher, person, start_time, end_time, "Does not like")
        if might_like_activity:
            file.write(f'\nThe children who might like to do "{activity}": {", ".join(might_like_activity)}.\n')
            database.insert_emotions(activity, teacher, person, start_time, end_time, "Might like")
        if many_emotions:
            file.write(
                f'\nThe children who had many different emotions while doing "{activity}": {", ".join(many_emotions)}.\n')
            database.insert_emotions(activity, teacher, person, start_time, end_time, "Not sure")
        database.conn.commit()
        database.cursor.close()

        question = simpledialog.askstring('Question', 'Do you want to have your all-time report? (y/n)')
        if question in ['y', 'Y', 'Yes', 'yes']:
            df = database.teacher_dataframe(teacher)
            table = f'{df.to_string(index=False)}\n'
            with open('../report/report.txt', 'w') as file:
                file.write(table)
                question1 = simpledialog.askstring('Question',
                                                   'Do you want the children names in your all-time report? (y/n)')
                if question1 in ['y', 'Y', 'Yes', 'yes']:
                    df1 = database.children_dataframe(teacher)
                    table1 = f'\n{df1.to_string(index=False)}'
                    file.write(table1)
        database.conn.close()
    print(f"Video saved to {output_path}")
    print(f"Emotion information saved to {file_path}")