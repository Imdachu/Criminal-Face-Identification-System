import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import face_recognition
import datetime
import time
import matplotlib.pyplot as plt
import logging
from mtcnn import MTCNN  # MTCNN for face detection

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
filename = ""
filename1 = ""

# Denoising Function
def denoise_frame(frame):
    return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

# Data Augmentation Function
def augment_frame(frame):
    augmented_frames = []
    flipped_frame = cv2.flip(frame, 1)
    augmented_frames.append(flipped_frame)
    return augmented_frames

# Preprocessing Function
def preprocess_frame(frame):
    frame = denoise_frame(frame)
    return frame

# MTCNN Face Detection
def detect_faces_with_mtcnn(frame):
    detector = MTCNN()
    faces = detector.detect_faces(frame)
    face_locations = [(face['box'][1], face['box'][0] + face['box'][2], face['box'][1] + face['box'][3], face['box'][0]) for face in faces]
    return face_locations

def main1(frame1, filename, filename1):
    try:
        logging.debug("Entering main1 function")
        message1 = tk.Label(frame1, text="Searching...", bg="green yellow", fg="black", width=15, height=1, activebackground="yellow", font=('times', 15, ' bold '))
        message1.place(x=140, y=380)

        logging.debug(f"Opening video file: {filename1}")
        video_capture = cv2.VideoCapture(filename1)

        logging.debug(f"Loading criminal image: {filename}")
        criminal_image = face_recognition.load_image_file(filename)
        criminal_face_encoding = face_recognition.face_encodings(criminal_image)

        if len(criminal_face_encoding) > 0:
            criminal_face_encoding = criminal_face_encoding[0]
            logging.debug("Criminal face encoding found")
        else:
            message1.configure(text="Not Detected")
            logging.warning("No face found in criminal image")
            return

        known_face_encodings = [criminal_face_encoding]
        known_face_names = ["Criminal"]

        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        while True:
            ret, frame = video_capture.read()
            logging.debug("Reading frame from video")

            if not ret:
                logging.info("End of video file reached")
                break

            if frame is None:
                logging.error("Frame is None")
                messagebox.showerror("Error", "Frame is None")
                return

            if frame.dtype != np.uint8:
                logging.error("Frame is not an 8-bit image")
                messagebox.showerror("Error", "Frame is not an 8-bit image")
                return

            frame = preprocess_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            logging.debug("Converted frame to RGB and preprocessed")

            if process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                face_locations = detect_faces_with_mtcnn(small_frame)
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)
                logging.debug("Processed frame for face locations and encodings")

                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                    name = ""

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]

                    face_names.append(name)
                    logging.debug(f"Face matched with name: {name}")

            process_this_frame = not process_this_frame

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left + 10, top + 10), (right + 10, bottom + 30), (0, 0, 255), 2)
                cv2.rectangle(frame, (left + 10, bottom + 30), (right + 10, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom + 20), font, 0.5, (255, 255, 255), 1)

                if name == "Criminal":
                    logging.info("Criminal detected")
                    plt.imshow(frame)
                    plt.title('Criminal Detected')
                    plt.show()
                    
                        # Save the frame with the detected face
                    cv2.imwrite("detected_criminal.jpg", frame)
                    
                    # Load and resize the saved image
                    detected_image = Image.open("detected_criminal.jpg")
                    resized_image = detected_image.resize((440, 280), Image.ANTIALIAS)
                    test1 = ImageTk.PhotoImage(resized_image)
                    
                    # Display the image in the Tkinter window
                    label3 = tk.Label(window, image=test1, width=440, height=280, bg="#262523")
                    label3.image = test1
                    label3.place(x=720, y=190)
                    
                    message1.configure(text="Criminal Detected!!!")
                    video_capture.release()
                    cv2.destroyAllWindows()
                    return

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            plt.imshow(frame)
            plt.title('Video Frame')
            plt.pause(0.01)

        video_capture.release()
        plt.close()
        logging.info("No criminal detected after processing the entire video")
        message1.configure(text="Not Detected")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def select_file():
    try:
        logging.debug("Entering select_file function")
        filetypes = (
            ('Image files', '.jpg;.jpeg;*.png'),
            ('All files', '.')
        )
        global filename
        filename = fd.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes)

        if filename:
            try:
                logging.debug(f"Selected file: {filename}")
                image = cv2.imread(filename)
                if image is None:
                    logging.error("Failed to load image")
                    raise ValueError("Failed to load image")

                logging.debug(f"Image shape: {image.shape}, Image data type: {image.dtype}")

                if len(image.shape) == 2 and image.dtype == 'uint8':
                    logging.debug("Image is 8-bit grayscale")
                elif len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4) and image.dtype == 'uint8':
                    logging.debug("Image is RGB or RGBA")
                    if image.shape[2] == 4:
                        logging.debug("Converting RGBA to RGB")
                        image = image[:, :, :3]
                else:
                    logging.error("Unsupported image type")
                    raise ValueError("Unsupported image type, must be 8bit gray or RGB image")

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                temp_filename = "temp.jpg"
                cv2.imwrite(temp_filename, image)
                filename = temp_filename

                criminal_image = face_recognition.load_image_file(filename)
                criminal_face_encoding = face_recognition.face_encodings(criminal_image)

                if len(criminal_face_encoding) > 0:
                    logging.debug("Face detected in image")
                    criminal()
                    select.config(text="Selected")
                else:
                    logging.warning("No face detected in the selected image")
                    messagebox.showerror("Error", "No face detected in the selected image. Please choose a different image.")
            except ValueError as ve:
                logging.error(str(ve))
                messagebox.showerror("Error", str(ve))
            except Exception as e:
                logging.error(f"Failed to process the image: {str(e)}")
                messagebox.showerror("Error", f"Failed to process the image: {str(e)}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def select_file1():
    try:
        logging.debug("Entering select_file1 function")
        filetypes = (
            ('mp4', '*.mp4'),
            ('All files', '.'),
        )
        global filename1
        filename1 = fd.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes)
        if filename1:
            logging.debug(f"Selected video file: {filename1}")
            main1(frame1, filename, filename1)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def criminal():
    try:
        logging.debug("Entering criminal function")
        global filename
        if filename:
            image11 = Image.open(filename)
            resized_image = image11.resize((150, 150), Image.LANCZOS)
            test = ImageTk.PhotoImage(resized_image)
            label1 = tk.Label(window, image=test, width=150, height=150)
            label1.image = test
            label1.place(x=256, y=280)
            logging.debug("Displayed criminal image")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


############################### GUI #######################################################

window = tk.Tk()
window.geometry("1280x720")
window.resizable(True, False)
window.title("Criminal Identification system")
window.configure(background='#262523')

frame1 = tk.Frame(window, highlightthickness=2, bg="#262523")
frame1.place(relx=0.55, rely=0.17, relwidth=0.38, relheight=0.70)

frame2 = tk.Frame(window, highlightthickness=2, bg="#262523")
frame2.place(relx=0.10, rely=0.17, relwidth=0.34, relheight=0.70)

message3 = tk.Label(window, text="Criminal Identification System", fg="white", bg="#262523", width=55, height=1,
                    font=('times', 29, ' bold '))
message3.place(x=10, y=10)

frame3 = tk.Frame(window, bg="#c4c6ce")
frame3.place(relx=0.46, rely=0.09, relwidth=0.22, relheight=0.07)

frame4 = tk.Frame(window, bg="#c4c6ce")
frame4.place(relx=0.30, rely=0.09, relwidth=0.2, relheight=0.07)

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day, month, year = date.split("-")

mont = {'01': 'January', '02': 'February', '03': 'March', '04': 'April', '05': 'May', '06': 'June', '07': 'July',
        '08': 'August', '09': 'September', '10': 'October', '11': 'November', '12': 'December'}

datef = tk.Label(frame4, text=day + "-" + mont[month] + "-" + year + "", fg="green yellow", bg="#262523", width=55,
                 height=1, font=('times', 20, ' bold '))
datef.pack(fill='both', expand=1)

clock = tk.Label(frame3, fg="green yellow", bg="#262523", width=55, height=1, font=('times', 22, ' bold '))
clock.pack(fill='both', expand=1)

head2 = tk.Label(frame2, text="       CRIMINAL'S IMAGE      ", fg="black", bg="lime green",
                 font=('times', 17, ' bold '))
head2.grid(row=0, column=1)
head2.place(x=60, y=10)
head3 = tk.Label(frame2, text="   SELECT THE FOOTAGE  ", fg="black", bg="lime green",
                 font=('times', 17, ' bold '))
head3.place(x=60, y=350)

head1 = tk.Label(frame1, text="       DETECTION     ", fg="black", bg="lime green", font=('times', 17, ' bold '))
head1.place(x=140, y=10)


def criminal():
    try:
        logging.debug("Entering criminal function")
        global filename
        if filename:
            image11 = Image.open(filename)
            resized_image = image11.resize((150, 150), Image.LANCZOS)
            test = ImageTk.PhotoImage(resized_image)
            label1 = tk.Label(window, image=test, width=150, height=150)
            label1.image = test
            label1.place(x=256, y=280)
            logging.debug("Displayed criminal image")
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

image1 = Image.open("img/img.png")
resized_image = image1.resize((100, 100), Image.LANCZOS)
image2 = ImageTk.PhotoImage(resized_image)
image_label = tk.Label(window, image=image2, bg="#262523")
image_label.place(x=160, y=5)


# BUTTONS
quitWindow = tk.Button(frame1, text="QUIT", command=window.destroy, fg="black", bg="green yellow", width=10, height=1,
                       activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=180, y=450)

select = tk.Button(frame2, text="Select file", command=select_file, fg="black", bg="green yellow", width=10, height=1,
                   activebackground="white", font=('times', 12, ' bold '))
select.place(x=150, y=100)

select2 = tk.Button(frame2, text="Select file", command=select_file1, fg="black", bg="green yellow", width=10, height=1,
                    activebackground="white", font=('times', 12, ' bold '))
select2.place(x=150, y=400)

# END GUI
window.mainloop()