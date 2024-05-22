# %% [markdown]
# ### Imports and Constants

# %%
# Uncomment, then use ctrl-shift-p and "Developer:Reload Window" if static analysis does not recognize any of your imports
%pip install opencv-python numpy tensorflow scikit-learn face-recognition pillow


# %%
#Unused imports


    # Maybe later.
#from contextlib import closing
    # Sometimes needed for SQLite3 but not so far.
#from sklearn.preprocessing import LabelEncoder
    # For old model
#import string
    # Not necessary

# %%

import cv2
import numpy as np
import os
import tensorflow as tf
import pickle
import face_recognition
import tkinter as tk
import tkinter.messagebox
from PIL import Image,ImageTk 
import sqlite3
import constants
import signals
import threading
from threading import Thread,current_thread ,RLock
import pandas as pd
import typing
import logging, sys
from io import BytesIO 

from datetime import datetime,date
from time import sleep
from pathlib import Path
from my_net_utils import *
from socket import *
import tkinter.filedialog 


# STUDENTS_LIST_FILENAME = 'students_list.pkl'


if 'Environment_Server' not in os.getcwd():
    os.chdir('Environment_Server')
    

# %% [markdown]
# ###Constants 

# %%
LABELS_TO_FACES_DICT_PICKLED_FILENAME = 'labels_to_face_encodings_dict.pkl'
LAST_RUN_PICKLE_FILENAME = 'last_run_exited_correctly.pkl'

STUDENTS_TABLE_NAME = 'students'
TIMESHEET_TABLE_NAME = 'timesheet'
ATTENDANCE_TABLE_NAME = 'attendance'

MAX_CONN = 5


# %% [markdown]
# ###Error Logging

# %%
logging.basicConfig(stream=sys.stderr,encoding='utf-8',level=logging.DEBUG,format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
handler = logging.FileHandler('ServerErrorLog.log', 'a', 'utf-8')
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
logger.addHandler(handler)
logging.getLogger('PIL').setLevel(logging.WARNING)

def warn_user_and_log_event(error : Exception, 
                            * ,
                 level:int = logging.ERROR, heading="Error" , 
                 logfile_msg : str = None, user_msg : str = None,
                 suppress : bool = False, suppress_condition = '' ) ->None:
    """
    Shows user a message about a warning (unless suppress is set to True), 
    afterwards, write  to a log file with a timestamp for later inspection

    Args:
        error (Exception): Exception raised, if any
        level (str): Which level from a predefined set of levels (in logging) is to be used
        logfile_msg (str): Message to add to log. If none set, use exception text instead.
        user_msg (str): Message to show to user. If none set, use str(error) instead.
        suppress (bool): Choose whether or not to show error to user. False by default.
        suppress_condition (str): Suppress only if the substring specifies appears in the exception raised. If not set, suppress unconditionally.

    """
    log_message = logfile_msg if logfile_msg else error
    
    if not error and not user_msg:
        user_msg = logfile_msg    

    level_is_valid = level in range(logging.DEBUG,logging.CRITICAL + 1, 10) 
    
    if not level_is_valid:
        tkinter.messagebox.showerror(title = "Invalid logger call" , message=f'You tried logging an invalid level!')
        level = logging.ERROR #keep a default value  
    
    should_show_info = level > logging.WARNING

    logger.log(level, msg = log_message , exc_info = should_show_info)
    print('\n',file=handler.stream)
    handler.stream.flush()

    display_message = user_msg if user_msg else str(error)

    if not suppress or suppress_condition not in str(error):
        if level >= logging.WARNING:
            tkinter.messagebox.showerror(title = heading , message=f'An error has occurred: {display_message}' )
        elif level == logging.WARNING:
            tkinter.messagebox.showwarning(title = heading , message=f'Warning: {display_message}' )
        else:
            tkinter.messagebox.showinfo(title = heading , message=f'Alert: {display_message}' )


    return


# %% [markdown]
# ### Utility Functions

# %%
#TODO: Consider removing

# def rename_file(old_filename : str,new_filename : str,directory_old='',directory_new = '') -> None:
#     if directory_old:
#         old_filename = os.path.join(directory_old,old_filename)
#     if directory_new:
#         new_filename = os.path.join(directory_new,new_filename)
    
#     os.rename(old_filename,new_filename)
#     return
    
# def get_day_of_week_as_string():
#     return datetime.today().strftime('%A')



# %% [markdown]
# ### Existing Embeddings/Mappings Retrieval

# %%
def load_face_encodings_from_photos(directory='Known_Faces') -> dict:
    if os.path.isfile(LABELS_TO_FACES_DICT_PICKLED_FILENAME):
        try:
            labels_to_encodings = pickle.load(open(LABELS_TO_FACES_DICT_PICKLED_FILENAME,'rb'))
        except Exception as e:
            warn_user_and_log_event(e,level=logging.INFO,logfile_msg="No encodings detected. Rebuilding.",suppress=True)
            labels_to_encodings = {}
    else:
        labels_to_encodings = {}

    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'): 
                label = ' '.join(os.path.splitext(filename)[0].upper().split('_'))
                if not label.isnumeric():
                     continue
                
                label = int(label)
                if label not in labels_to_encodings.keys() : #Do not overwrite existing encodings
                    image_path = os.path.join(directory, filename)

                    # Load image and generate face encoding 
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)
                    
                    #if face has been identified. 
                    if len(encoding) > 0:
                            labels_to_encodings[label] = encoding[0]
                                    

    pickle.dump(labels_to_encodings,open(LABELS_TO_FACES_DICT_PICKLED_FILENAME,'wb'))

    return labels_to_encodings



# %% [markdown]
# ### Wipe all existing data

# %%
#%run wipe.py --dbpath SQL_db/class.db --pickle $LABELS_TO_FACES_DICT_PICKLED_FILENAME

# %% [markdown]
# ### MAIN LOOP

# %%
#TODO: Replace all the packs with proper grids. Maybe even move to using canvases.

class ServerApp:

    exited_correctly, time_of_last_exit = pickle.load(open(LAST_RUN_PICKLE_FILENAME,'rb')) if os.path.isfile(LAST_RUN_PICKLE_FILENAME) else True, datetime.now()
    #Assume the best since adding absences for no reason is worse than missing an absence

    

    def ungraceful_exit(self,exc : Exception):
        if self.db_connection:
            self.db_connection.close()
        if self.server_sock:
            self.server_sock.close()
        if self.root:
            self.root.destroy()
        raise exc
        

    def __init__(self) -> None:
        self.initialize_values()
        self.make_root()

    def make_root(self):
        self.root = tk.Tk()
        self.root.title('Start Page')
        self.root.protocol('WM_DELETE_WINDOW',self.quit_app)

        self.root_widgets = []

        self.root_widgets.extend(
            [
                tk.Button(self.root, text="Import Excel file with all sheets (Not implemented)", command = None),
                tk.Button(self.root, text="Import Students List", command = lambda: self.populate_table_from_spreadsheet(STUDENTS_TABLE_NAME)),
                tk.Button(self.root,text='Import Time Sheet',command = lambda: self.populate_table_from_spreadsheet(TIMESHEET_TABLE_NAME)),
                tk.Button(self.root,text='Import Attendance Sheet',command = lambda: self.populate_table_from_spreadsheet(ATTENDANCE_TABLE_NAME)),
                
                tk.Button(self.root, text="Start Attendance", command=self.start_attendance),
                tk.Button(self.root, text="Exit", command = self.quit_app)
             ]
        ) 

    def initialize_values(self): 
        self.exception = None
        self.server_sock = socket()
        self.server_sock.setblocking(False)
        self.server_sock.setsockopt(SOL_SOCKET,SO_KEEPALIVE,1)
        self.connections = [] 
        self.connection_handler_threads = []
        self.lock = RLock()
        self.keep_threads_running = threading.Event()
        self.keep_threads_running.clear()
        self.error_in_thread = None

        self.path_photos = 'Photos_In'
        self.path_db = 'SQL_db/class.db'

        self.client_connected = False

        self.last_predicted_name = None
        
    def launch(self):
        self.server_sock.bind(('',PORT))
        
        self.load_model()

        self.db_connection = sqlite3.connect(self.path_db , detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES )
        self.db_cursor = self.db_connection.cursor()
        print('DB connection created')
        
        try:
            self.create_tables()
        except sqlite3.IntegrityError as integrity_violation:
            warn_user_and_log_event(integrity_violation,level=logging.CRITICAL)
            self.ungraceful_exit(integrity_violation)
        print('Tables created')
        hours_since_last_exit = ((datetime.now() - ServerApp.time_of_last_exit).seconds) // (60**2)
        
        #Check that last run was not a crash, and that it was last exited at the end of the day
        #Let's just assume that the longest day has at least 10 hours between its and the start of the next
        if ServerApp.exited_correctly and hours_since_last_exit >= 10:
            
            #TRUE/FALSE in SQLite are just aliases for 1 and 0, so using it in integer arithmetic works.
            self.db_cursor.execute(f'UPDATE {ATTENDANCE_TABLE_NAME} SET "{ServerApp.__NUM_ABSENT}" = ?',
                                    (f'"{ServerApp.__NUM_ABSENT}" + "{ServerApp.__BOOL_ABSENT}" ',))

            self.db_cursor.execute(f'UPDATE {ATTENDANCE_TABLE_NAME} SET {ServerApp.__BOOL_ABSENT}  = ?',('FALSE',))
            
            #If program did not crash, it must mean that it exited at the end of the previous day
            #If so, reset everyone to absent and restart from there
            

        self.db_connection.commit()

        print('Attendance updated')

        for widget in self.root_widgets:
            widget.pack()

        self.root.mainloop()

    def populate_table_from_spreadsheet(self,table_name):
        excel_extensions = ['.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt']

        file_name = tkinter.filedialog.askopenfilename(filetypes= ( ('Excel File',[f'*{extension}' for extension in excel_extensions]) ,
                                                                    ('CSV File', '*.csv' ) )
                                                        )
        if not file_name: 
            return
        
        file_extension = os.path.splitext(file_name)[-1]
        
        is_excel = False

        if file_extension in excel_extensions:
            is_excel = True

        elif file_extension == '.csv':
            is_csv = True
        else:
            #Happens when a file is not filtered out by filetypes, such as internet shortcuts
            tkinter.messagebox.showwarning('Invalid choice',
                                           'File chosen cannot be read for data. Please use a valid excel or .csv file.')
            return
        
        try:
            if is_excel:
                df_table = pd.read_excel(file_name)
            elif is_csv:
                _table = pd.read_csv(file_name)

        except Exception as e:
            warn_user_and_log_event(e,level=logging.INFO,
                                    user_msg = "The file you selected was of a valid format, but did not contain readable tables.",
                                    suppress=False)
            return

        try:
            if table_name == ATTENDANCE_TABLE_NAME:
                table_headings = df_table.columns

                if ServerApp.__BOOL_ABSENT not in table_headings:
                    df_table[ServerApp.__BOOL_ABSENT] = 'TRUE'
                
                if ServerApp.__NUM_ABSENT not in table_headings:
                    df_table[ServerApp.__NUM_ABSENT] = 0

            
            #Overwrite on conflict
            #There's a way to do this with if_exists='append' by combining two dfs and removing duplicates, but let's not worry about that yet
            df_table.to_sql(name=table_name,con=self.db_connection,if_exists='append',index=False) 

        except sqlite3.IntegrityError as duplicates_found:
            #Should not be possible with if_exists set to 'replace'
            warn_user_and_log_event(error=duplicates_found,
                                    level=logging.WARNING,
                                    suppress=False,
                                    suppress_condition='UNIQUE constraint')
            # df_table.to_sql(name=table_name,con=self.db_connection,if_exists='replace',index=False) 
        
        except sqlite3.OperationalError as shape_mismatch:
            warn_user_and_log_event(shape_mismatch,
                                    user_msg=f'Invalid format for table {table_name}',
                                    level=logging.INFO,
                                    suppress=False)


        except Exception as e:
            warn_user_and_log_event(e,
                                    level=logging.ERROR,
                                    suppress=False)
            return
        
        else: #On insert success
            self.db_connection.commit()
            if table_name == STUDENTS_TABLE_NAME:
                self.check_for_discrepancies()

    #Check for discrepancies between existing encodings and table entries           
    def check_for_discrepancies(self):
        students = self.db_cursor.execute(f'SELECT "student_id" FROM {STUDENTS_TABLE_NAME}')
        student_ids = [id_ for (id_,) in students]
        
        print(student_ids)
        print(self.known_face_labels)

        for student_id in student_ids:
            if student_id not in self.known_face_labels:
                warn_user_and_log_event(error=None,
                                        level=logging.INFO,
                                        logfile_msg=f'User {student_id} present in database but not in encodings',
                                        heading='Missing student face image',
                                        user_msg=(f'Student with id {student_id} is in the database, but their face is not in the system.  ',
                                            'They will not be recognized by the system if seen.')
                                        )
            
        for label in self.known_face_labels:
            if label not in student_ids:
                warn_user_and_log_event(error=None,
                                        level=logging.INFO,
                                        logfile_msg=f'User {label} present in encodings but not in database',
                                        heading='Student not in database',
                                        user_msg=(f'Student with id {label} has a photo, but their name is not in the system.  '
                                            "They cannot be logged as present as a result.")
                                        )
                

    def quit_app(self):
        self.keep_threads_running.clear() #Redunant?
        self.db_connection.close()
        self.server_sock.close()
        ServerApp.exited_correctly = True
        self.root.quit()
        self.root.destroy()

    
    __STU_ID = "student_id"
    __STU_NAME = "student_name"
    __SBJ_ID = "subject_id"
    __SEC = "section"
    __ROOM = "room"
    __DAYS = "days"
    __TIME_ST = "start_time"
    __TIME_END = "end_time"
    __BOOL_ABSENT = "is_absent"
    __NUM_ABSENT = "num_absences"

    def create_tables(self): #always ensure tables are present 
        create_students_table = f'''CREATE TABLE IF NOT EXISTS "{STUDENTS_TABLE_NAME}"(
            "{ServerApp.__STU_ID}"	INTEGER,
            "{ServerApp.__STU_NAME}"	TEXT COLLATE NOCASE,
            PRIMARY KEY("{ServerApp.__STU_ID}") ON CONFLICT ABORT
            )
        ''' 

        create_timesheet_table = f'''CREATE TABLE IF NOT EXISTS "{TIMESHEET_TABLE_NAME}"(
            "{ServerApp.__SBJ_ID}"    INTEGER,
            "{ServerApp.__SEC}"	INTEGER,
            "{ServerApp.__ROOM}"	TEXT NOT NULL,
            "{ServerApp.__DAYS}"	TEXT NOT NULL,
            "{ServerApp.__TIME_ST}"	TEXT NOT NULL,
            "{ServerApp.__TIME_END}"	TEXT NOT NULL,
            UNIQUE("{ServerApp.__ROOM}","{ServerApp.__DAYS}","{ServerApp.__TIME_ST}"),
            PRIMARY KEY("{ServerApp.__SBJ_ID}","{ServerApp.__SEC}") 
            )
        '''
        #Assumption: No overlaps will occur, and no scheduling errors in general will happen 
        
        create_attendance_table = f'''CREATE TABLE IF NOT EXISTS "{ATTENDANCE_TABLE_NAME}"(
            "{ServerApp.__STU_ID}"	INTEGER,
            "{ServerApp.__SBJ_ID}"	TEXT COLLATE NOCASE,
            "{ServerApp.__SEC}"	INTEGER,
            "{ServerApp.__BOOL_ABSENT}" BOOLEAN DEFAULT TRUE,
            "{ServerApp.__NUM_ABSENT}" INTEGER DEFAULT 0,
            
            PRIMARY KEY("{ServerApp.__STU_ID}","{ServerApp.__SBJ_ID}"),

            FOREIGN KEY("{ServerApp.__STU_ID}") REFERENCES "{STUDENTS_TABLE_NAME}"("{ServerApp.__STU_ID}"),
            FOREIGN KEY("{ServerApp.__SBJ_ID}","{ServerApp.__SEC}") REFERENCES "{TIMESHEET_TABLE_NAME}"("{ServerApp.__SBJ_ID}","{ServerApp.__SEC}")
            )
    '''
        # create_registration_view = f'''CREATE VIEW IF NOT EXISTS "registration"
        # AS
        # SELECT
        #     "student_id",
        #     "subject",
        # '''
        self.db_cursor.execute(create_students_table)
        self.db_cursor.execute(create_timesheet_table)
        self.db_cursor.execute(create_attendance_table)
        self.db_connection.commit()  

    #FIXME before using this function. It does not behave in any useful way.
    def UNUSED_get_student_absences(self,student_id):
        try:
            absences = self.db_cursor.execute(f'SELECT "{ServerApp.__SBJ_ID}","{ServerApp.__NUM_ABSENT}" FROM {ATTENDANCE_TABLE_NAME} WHERE "{ServerApp.__STU_ID}" = ?',
                                              (student_id,) )
            return [record for record in absences] 
        
        except Exception as e:
            warn_user_and_log_event(e,level=logging.WARNING)


    def accept_connections(self):
        while self.keep_threads_running.is_set(): 
            try:
                with self.lock:
                    num_active_conns = len(self.connections)

                if num_active_conns < MAX_CONN:
                    client_connection, raddr = self.server_sock.accept()
                    client_connection.settimeout(GLO_TIMEOUT)
                    with self.lock:
                        self.connections.append(client_connection)
                        t = Thread(
                        target=self.handle_client,
                        args=(client_connection,
                              raddr),
                        daemon=False)
                        self.connection_handler_threads.append(t)
                        self.connection_handler_threads[-1].start()
                else:
                    sleep(1)
                    #TODO: This might be better done with threading.Event and wait() but for now let's keep it since it works
                    
            except BlockingIOError as e:
                #If broken, check here
                sleep(1.5)
                pass # Suppress error
        with self.lock:
            for thread in self.connection_handler_threads:
                print(f'joining thread {thread.name}')
                thread.join()
        
    
    def handle_client(self, connection : socket, remote_address : str ):
        try:      
            remote_ip,remote_port = remote_address
            self.set_status_threadsafe(f'Accepted connection from {remote_ip}, port {remote_port}')
            
            building = remote_address[0].split('.')[-2]
            match building:
                case '100':
                    building_code = 'EE'
                case '150':
                    building_code = 'BU'
                case '200':
                    building_code = 'CS'
                case _:
                    building_code = 'EE'
                    #TODO: Change to none in production
                    
            if not building_code:
                #Reject connections from unknown IPs
                connection.shutdown(SHUT_RDWR)
                connection.close()
                self.connection_handler_threads.remove(current_thread())

                return
            
            room_number = building_code + str( 100 + int(remote_ip.split('.')[-1]) ) 

            while self.keep_threads_running.is_set():
                error_in_thread = None
                try:
                    msg_in = receive_strip(connection)
                    if msg_in == signals.QUIT or not self.keep_threads_running.is_set():
                        self.set_status_threadsafe(f'Connection {remote_address} has quit normally.')
                        connection.shutdown(SHUT_RDWR)
                        connection.close()
                        break
                    else:
                        if self.return_button.winfo_viewable():
                            with self.lock:
                                self.return_button.pack_forget()
                                #We are enforcing this with an iron grip I am afraid. 
                        image_size = msg_in
                        print(image_size)


                    image_size = int(image_size.split('=')[-1])
                    padded_send(connection,signals.SIZE_RECEIVED)
                    image_raw = receive_normal(connection,image_size)

                    
                    #TODO: [optional] Change out for BytesIO
                        #Thought I'd keep the file in case someone wanted to have a look at them later though.
                    
                    # file_name = 'Received_' + current_datetime.isoformat().replace(':','-') + '.png'
                    # path_img = f'./{self.path_photos}/{file_name}'
                    
                    png_bytes = BytesIO()
                    print('Writing image...')
                    png_bytes.write(image_raw)

                    #return cursor back to start of buffer
                    start_of_file = 0
                    png_bytes.seek(start_of_file)
                    print('Converting image...')

                    student_image = Image.open(png_bytes)

                    image_for_model = np.array(student_image)

                    print('Decoding Image')
                    image_for_model = cv2.cvtColor(image_for_model,cv2.COLOR_RGB2BGR)

                    print('Showing image...')
                    photo_img = ImageTk.PhotoImage(image=student_image)
                    
                    with self.lock:
                        self.student_image.image = photo_img #Prevents image being wiped on disconnect  

                    with self.lock:
                        self.student_image.config(image=photo_img)
                        last_predicted_student_id = self.predict_face(image_for_model)
                        self.update_textbox(last_predicted_student_id)

                    #TODO: Improve the communication protocol a little here
                    padded_send(connection,str(last_predicted_student_id))
                    if last_predicted_student_id in [constants.NONE,constants.UNKNOWN]:
                        continue

                    client_response = receive_strip(connection) #Client responds if name was correct or not
                    
                    last_prediction_correct = (client_response == signals.YES) 

                    if not last_prediction_correct :
                        continue
                    
                    #Mutex during db operations prevents access/synchronization errors 
                    with self.lock:
                        #Cannot access self.db_connection and its cursor from a different thread, so we need to create an ephemeral connection in each thread 
                        ephemeral_db_connection = sqlite3.connect(self.path_db , detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
                        
                        ephemeral_cursor = ephemeral_db_connection.cursor()
                        try:    
                            error_in_writing_db = None
                            day_of_week_as_string = datetime.today().strftime('%A')
                            ephemeral_cursor.execute(f'''UPDATE {ATTENDANCE_TABLE_NAME} 
                                                    SET {ServerApp.__BOOL_ABSENT} = ? 
                                                    /*In Attendance, stu_id + sbj_id are uniquely identifying (PK)*/
                                                    WHERE {ServerApp.__STU_ID} = ?
                                                    AND {ServerApp.__SBJ_ID} = (
                                                        SELECT {ServerApp.__SBJ_ID}
                                                        FROM {TIMESHEET_TABLE_NAME}
                                                        WHERE {ServerApp.__ROOM} = ?
                                                        AND 
                                                        strftime("%H:%M", datetime({ServerApp.__TIME_ST},'-5 minutes') ) <= time('now','localtime')
                                                        /* -5 minutes to account for early arrivals */
                                                        AND
                                                        strftime("%H,%M", {ServerApp.__TIME_END}, '-30 minutes' ) >= time('now','localtime')
                                                        /* 30 minutes before end of lecture at least */
                                                        AND
                                                        {ServerApp.__DAYS} LIKE "%{day_of_week_as_string[0:3]}%" 
                                                        /* Days of week are unique on first 2 letters. 
                                                        Allows for flexibility in spreadsheet days (Monday or Mon or Mo) */)
                                                        ''',
                                                        ('FALSE',
                                                        last_predicted_student_id,
                                                        room_number) )
                            
                            assert ephemeral_cursor.rowcount < 2
                            # Raise a serious error if more than 1 student has been marked present
                            # If that happened, something has gone VERY wrong with the sql code 

                            if ephemeral_cursor.rowcount > 0:
                                log_result = signals.SUCCESSFULLY_MARKED_PRESENT(last_predicted_student_id)
                            else:
                                log_result = signals.NOT_ENROLLED(last_predicted_student_id)
                            
                            padded_send(connection,log_result)

                        except Exception as exc:
                            error_in_writing_db = exc
                        else:
                            ephemeral_db_connection.commit()
                            ephemeral_db_connection.close()

                        finally:
                            if error_in_writing_db:
                                level = logging.ERROR if isinstance(error_in_writing_db,sqlite3.IntegrityError) else logging.CRITICAL
                                warn_user_and_log_event(error_in_writing_db,level)
                                self.set_status_threadsafe(f'Error in writing to database: {error_in_writing_db}')
                                self.keep_threads_running.clear()
                                #If an error happens here there's a programming issue.
                                
                except TimeoutError as timeout_error:
                    self.set_status_threadsafe(f'Connection with {remote_ip} timed out')
                    warn_user_and_log_event(timeout_error,level=logging.DEBUG,suppress=True)
                    
                except ConnectionError as dropped_connection:
                    self.set_status_threadsafe(f'Remote host {remote_ip} unexpectedly disconnected')
                    warn_user_and_log_event(dropped_connection,level=logging.INFO,suppress=True)

                except Exception as misc_exc:
                    self.set_status_threadsafe(f'Exception in thread {current_thread().name} with connection {remote_ip}\n: {misc_exc}')
                    warn_user_and_log_event(misc_exc,level=logging.CRITICAL,suppress=True)
                    self.keep_threads_running.clear()
                
                finally:
                    with self.lock:
                        if not self.return_button.winfo_viewable(): #prevent double-packs in case of multiple accesses
                            self.return_button.pack()

        except ConnectionError as err_connection :
            
            self.set_status_threadsafe(err_connection)
            warn_user_and_log_event(err_connection,level=logging.INFO,suppress=True,suppress_condition='not a socket')
        finally:
            with self.lock:
                self.connections.remove(connection)
                self.connection_handler_threads.remove(current_thread())
                print(self.connection_handler_threads)
                print(self.connections)
            print('Ok done')
            #TODO: Remove prints in "production"
            return 

    def set_status_threadsafe(self,message):
        with self.lock:
            self.status.set(str(message))   
            
    def table_isempty(self,table_name):
        self.db_cursor.execute(f'SELECT * FROM {table_name}')
        data = self.db_cursor.fetchone()
        return False if data is not None else True
    #table is not empty if non-null data exists
    
    def start_attendance(self):
        #Guard clause to check if system will work. Do not start if not.
            #TODO: Uncomment it when running a full system test

        for table in [STUDENTS_TABLE_NAME,ATTENDANCE_TABLE_NAME,TIMESHEET_TABLE_NAME]:
            if self.table_isempty(table):
                tkinter.messagebox.showerror(title='Table Empty' , message=f'Table {table} is empty. Attendance system will not work.')
                return

        self.attendance_window = tk.Toplevel(self.root)        
        self.root.withdraw()
        
        #Disable the exit button. I want to have full control over when to show it.
        self.attendance_window.protocol('WM_DELETE_WINDOW',lambda : None )
        self.server_sock.listen(MAX_CONN) 

        self.student_image = tk.Label(master=self.attendance_window)
        self.student_image.pack()

        photoimg= ImageTk.PhotoImage(image=Image.open('Placeholder.png'))  # Convert PIL Image to ImageTk 
        self.student_image.config(image=photoimg)
        self.student_image.image = photoimg 

        self.status = tk.StringVar(self.attendance_window,value=f'Listening on {self.server_sock.getsockname()}')
        self.status_label = tk.Label(self.attendance_window,textvariable=self.status)
        self.status_label.pack()

        self.last_student_name_label = tk.Label(self.attendance_window,text="Last detected student:")
        self.last_student_name_label.pack()
        
        self.last_student_textbox = tk.Text(self.attendance_window,height=2,width=50)
        self.last_student_textbox.config(state=tk.DISABLED)
        self.last_student_textbox.pack()

        self.return_button = tk.Button(self.attendance_window,command=self.return_to_main,text='Return to main menu')
        self.return_button.pack()
        self.attendance_window.lift()
        self.keep_threads_running.set()

        self.server_thread = Thread()
        self.accepter_thread = Thread(target=self.accept_connections,name="Thread_Client",daemon=False)
        
        self.accepter_thread.start()
            
    def return_to_main(self): 
        #Used to raise error in client handling thread. With the flag it should no longer do so.
        print('Stopping accept')
        self.keep_threads_running.clear()
        print('Waiting for accept')

        
        print('Destorying window')
        self.attendance_window.destroy()
        self.root.deiconify()

        #TODO: Unused. Either replace with a shared queue or remove. 
        # if self.error_in_thread:
        #     self.ungraceful_exit(self.error_in_thread)

    def update_textbox(self,new_text : str,textbox : tk.Text = None) -> None:
        if textbox == None:
            textbox = self.last_student_textbox
        textbox.config(state=tk.NORMAL)
        textbox.delete(constants.TK_START_INDEX,tk.END)
        textbox.insert(constants.TK_START_INDEX,new_text)
        textbox.config(state=tk.DISABLED)

    def load_model(self) -> None:
        self.labels_to_encodings = load_face_encodings_from_photos()
        print('Model loaded')
        self.known_face_labels, self.known_face_encodings = list(self.labels_to_encodings.keys()) , list(self.labels_to_encodings.values()) 

    def clear_all_photos(self) -> None:
        received_photos = os.listdir(self.path_photos)
        for photo in received_photos:
            os.remove(photo)

    def predict_face(self,image) -> str:   
        
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        name = constants.NONE

        if face_locations:
            for encoding in face_encodings:
                # Compare the face encoding with known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, encoding)
                name = constants.UNKNOWN
                
                if True in matches:
                    
                    first_match_index = matches.index(True)
                    name = self.known_face_labels[first_match_index]
            
        return name


# %%
try:
    instance = ServerApp()
    instance.launch()

#shouldn't happen anymore since I set the listening socket to non-blocking 
    #and the client connection threads are not to propagate a timeouterror to the main thread
# except TimeoutError as te:
#     warn_user_and_log_event(te,level=logging.DEBUG,suppress=True)

except Exception as e:
    exited_correctly = False
    warn_user_and_log_event(e,suppress=False,level=logging.CRITICAL)
    # print('writing last run result in exc')

else:
    exited_correctly = True
    # print('writing last run result in else')
            
finally:
    time_of_exit = datetime.now()
    run_result = exited_correctly,time_of_exit
    with open(LAST_RUN_PICKLE_FILENAME,'wb') as run_result_dump:  
        pickle.dump(run_result,run_result_dump)

    del instance




# %%
#!jupyter nbconvert --to script ServerDemo.ipynb

# %%
#%load_ext mypy_ipython
#%system net session /delete


