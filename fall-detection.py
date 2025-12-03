# Fall detection project
#___________________________________________________________# LIBRARIES USED _____________________________________________________
import cv2
import mediapipe as mp
from playsound import playsound
import time
import threading
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime

#___________________________________________________________# EMAIL FUNCTION _____________________________________________________
def send_email_with_frame(image_path, frame_id):
    sender_email = ""   # Put your email inside
    password = " 16 Letters"  # Gmail App Password. (Put your Gmail App Password inside) (Check under)
    receiver_emails = ["@gmail.com"]   # Gmails that RECEIVE the email. (Yes u can add up to as many as u want)

#Gmail App Password: search for "app password" on google and create a random name (project) => Save the password (16 letters)
    # Sending time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"[ALERT] Fall detected at frame {frame_id}"
    body = f"Fall detected at frame {frame_id}. Time: {timestamp}. Check the images!"

    # Email creating
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(receiver_emails)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Images:
    with open(image_path, "rb") as attachment:
        mime_base = MIMEBase('application', 'octet-stream')
        mime_base.set_payload(attachment.read())
        encoders.encode_base64(mime_base)
        mime_base.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
        msg.attach(mime_base)

    try:
        # Connect to SMTP Gmail:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_emails, msg.as_string())
        server.quit()
        print(f"Email sent successfully to: {receiver_emails}")
    except Exception as e:
        print("Failed to send email:", e)
    finally:
        # Delete temporary images after using for memory saving.
        if os.path.exists(image_path):
            os.remove(image_path)

#___________________________________________________________# SETUP _______________________________________________________________
event_log = []
max_events_display = 3
startup_ignore_frames = 200     # Enough x frames -> run (append y_center to y_center list).

frame_id = 0
y_center_list = []
fall_detected = False
possible_fall = False
cooldown_time = 5  # seconds
last_fall = 0

threshold_fall = 0.08
threshold_possible_fall = 0.05

cap = cv2.VideoCapture(r"")  # change the path to the video for running. Put the full path in like "path"
# cap = cv2.VideoCapture(0)  # For Webcam test.

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

#___________________________________________________________# LOOP ________________________________________________________________
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    #___________________________________________________________# Pose Detection _______________________________________________________
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
            y_center = (left_hip.y + right_hip.y) / 2

            if frame_id >= startup_ignore_frames and isinstance(y_center, (float, int)):
                y_center_list.append(y_center)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    #___________________________________________________________# Fall Detection Logic _________________________________________________
    if len(y_center_list) >= 10:
        delta_y = y_center_list[-1] - y_center_list[-10]    # delta_y in 10 closest frames.
        # -1 is the closest y.  # -10 is the y 10 frames away (adjustable).

        if delta_y > threshold_fall:
            current_time = time.time()
            if not fall_detected and (current_time - last_fall > cooldown_time):
                fall_detected = True
                last_fall = current_time
                print("FALL DETECTED at frame:", frame_id)
                event_log.append(f"[{frame_id}] FALL DETECTED!")

                # Save the frame and email in a seperated thread.
                img_path = f"fall_frame_{frame_id}.jpg"
                cv2.imwrite(img_path, frame)
                threading.Thread(target=send_email_with_frame, args=(img_path, frame_id)).start()

        elif delta_y > threshold_possible_fall:
            possible_fall = True
        else:
            possible_fall = False

    #___________________________________________________________# Reset fall detection _________________________________________________
    if fall_detected and (time.time() - last_fall > cooldown_time):
        fall_detected = False
        possible_fall = False

    #___________________________________________________________# Visualisation _______________________________________________________
    status_y = 20
    if frame_id < startup_ignore_frames:
        cv2.putText(frame, "SYSTEM WARMING UP...", (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 255), 2)
    elif fall_detected:
        cv2.putText(frame, "FALL DETECTED!", (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    elif possible_fall:
        cv2.putText(frame, "POSSIBLE FALL...", (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    y_offset = status_y + 25
    if len(y_center_list) > 0:
        cv2.putText(frame, f"y_center: {y_center_list[-1]:.3f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    y_offset += 22
    cv2.putText(frame, f"delta_y threshold (FALL): {threshold_fall}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    y_offset += 22
    cv2.putText(frame, f"delta_y threshold (POSSIBLE): {threshold_possible_fall}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2)

    # Event log
    log_start_y = frame.shape[0] - 10
    line_height = 18
    for i, log in enumerate(reversed(event_log[-6:])):
        y = log_start_y - i * line_height
        cv2.putText(frame, log, (10, y), cv2.FONT_HERSHEY_PLAIN, 1.2, (200, 255, 200), 2)

    # Minimap
    minimap_w = 150
    minimap_h = 100
    minimap_margin = 10
    minimap_x = frame.shape[1] - minimap_w - minimap_margin
    minimap_y = frame.shape[0] - minimap_h - minimap_margin
    cv2.rectangle(frame, (minimap_x, minimap_y), (minimap_x + minimap_w, minimap_y + minimap_h), (50, 50, 50), -1)
    cv2.putText(frame, "y_center Tracking", (minimap_x + 5, minimap_y + 15),
                cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)

    recent_vals = y_center_list[-minimap_w:]
    if recent_vals:
        y_vals_scaled = [int(minimap_y + minimap_h - ((y - min(recent_vals)) / (max(recent_vals) - min(recent_vals) + 1e-6)) * (minimap_h - 20)) for y in recent_vals]
        for idx in range(1, len(y_vals_scaled)):
            x1 = minimap_x + idx - 1
            x2 = minimap_x + idx
            y1 = y_vals_scaled[idx - 1]
            y2 = y_vals_scaled[idx]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    #___________________________________________________________# FRAME _______________________________________________________________
    cv2.namedWindow("AI System", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("AI System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("AI System", frame)

    if cv2.waitKey(10) & 0xFF == ord('.'):
        break

#___________________________________________________________# ENDING _______________________________________________________________
cap.release()
cv2.destroyAllWindows()
