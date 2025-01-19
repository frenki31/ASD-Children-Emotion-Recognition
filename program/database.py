'''
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

def enter_teacher(teacher):
    cursor.execute("SELECT COUNT(*) FROM TEACHER WHERE TEACH_FNAME = ? AND TEACH_LNAME = ?", teacher.split()[0],
                   teacher.split()[1])
    if cursor.fetchone()[0] == 0:
        cursor.execute('INSERT INTO TEACHER(TEACH_FNAME, TEACH_LNAME) VALUES (?, ?)',
                       (teacher.split()[0], teacher.split()[1]))
        # conn.commit()

def enter_activity(activity):
    cursor.execute("SELECT COUNT(*) FROM ACTIVITY WHERE ACT_NAME = ?", activity)
    if cursor.fetchone()[0] == 0:
        cursor.execute('INSERT INTO ACTIVITY(ACT_NAME) VALUES (?)', activity)
        # conn.commit()

def enter_person(person):
    cursor.execute("SELECT COUNT(*) FROM CHILD WHERE CHILD_FNAME=? AND CHILD_LNAME=?", person.split()[0],
                   person.split()[1])
    if cursor.fetchone()[0] == 0:
        # Person doesn't exist, perform INSERT
        insert_query = "INSERT INTO CHILD (CHILD_FNAME, CHILD_LNAME) VALUES (?, ?)"
        cursor.execute(insert_query, (person.split()[0], person.split()[1]))
        # conn.commit()

def insert_emotions(activity, teacher, person, start_time, end_time, like):
    stored_procedure = 'EXEC SP_INSERT_INTO_ACT_TEACH_CHILD @act_name=?, @teach_fname=?, @teach_lname=?, @child_fname=?, @child_lname=?, @act_st=?, @act_et=?, @like=?'
    cursor.execute(stored_procedure, activity, teacher.split()[0], teacher.split()[1], person.split()[0],
                   person.split()[1], start_time, end_time, like)

def teacher_dataframe(teacher):
    query = f"EXEC SP_TEACHER_REPORT @teach_fname = '{teacher.split()[0]}', @teach_lname='{teacher.split()[1]}'"
    return pd.read_sql(query, conn)

def children_dataframe(teacher):
    query1 = f"EXEC SP_CHILDREN_REPORT @teach_fname = '{teacher.split()[0]}', @teach_lname = '{teacher.split()[1]}'"
    return pd.read_sql(query1, conn)
'''