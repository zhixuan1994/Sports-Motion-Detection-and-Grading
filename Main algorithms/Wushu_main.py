import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from video_grade import video_grade
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
A_list = os.listdir("test_video_A/")
B_list = os.listdir("test_video_B/")

def stand_frame_out(video_url):
    processed_img_list_std, marks_std = [], []
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    mp_drawing = mp.solutions.drawing_utils
    cap_std = cv2.VideoCapture(video_url)
    while cap_std.isOpened():
        ret, frame = cap_std.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect pose
        results = pose.process(image)
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        processed_img_list_std.append(image)
        marks_std.append(results.pose_landmarks)
    cap_std.release()
    return processed_img_list_std[1:], marks_std[1:]

processed_img_list_std, marks_std = stand_frame_out('standard_A.mp4')
standard_frame_marks_A,standard_frame_list_A = [], []
A_key_frame = [80, 120, 159, 195, 265, 337, 372, 413]
for i in A_key_frame:
    standard_frame_marks_A.append(marks_std[i])
    standard_frame_list_A.append(processed_img_list_std[i])

processed_img_list_std, marks_std = stand_frame_out('standard_B.mp4')
standard_frame_marks_B,standard_frame_list_B = [], []
B_key_frame = [75, 125,  170, 210, 270, 343, 382, 416]
for i in B_key_frame:
    standard_frame_marks_B.append(marks_std[i])
    standard_frame_list_B.append(processed_img_list_std[i])

A_th = 0.1*np.ones(shape=(len(standard_frame_marks_A),))
A_res, A_frames, A_sim_res = [], [], []
for url_one in tqdm(A_list):
    url_one = 'test_video_A/' + url_one
    grade_class = video_grade(standard_frame_marks_A, A_th, url_one)
    grade_res, frame_res, sim_res = grade_class.grading_main()
    A_res.append(grade_res)
    A_frames.append(frame_res)
    A_sim_res.append(sim_res)

B_th = 0.12*np.ones(shape=(len(standard_frame_marks_B),))
B_res, B_frames, B_sim_res = [], [], []
for url_one in tqdm(B_list):
    url_one = 'test_video_B/' + url_one
    grade_class = video_grade(standard_frame_marks_B, B_th, url_one)
    grade_res, frame_res, sim_res = grade_class.grading_main()
    B_res.append(grade_res)
    B_frames.append(frame_res)
    B_sim_res.append(sim_res)

def freq_fig(save_title, sim_res, target_ind):
    fig, axes = plt.subplots(2, 5, figsize=(80, 35))
    for i in range(4):
        axes[0,i].plot(sim_res[i][target_ind])
        axes[0,i].tick_params(axis='x', labelsize=20)
        axes[0,i].tick_params(axis='y', labelsize=30)
        axes[1,i].plot(sim_res[i+4][target_ind])
        axes[1,i].tick_params(axis='x', labelsize=20)
        axes[1,i].tick_params(axis='y', labelsize=30)

    axes[1,4].plot(sim_res[-1][target_ind])
    axes[1,4].tick_params(axis='x', labelsize=20)
    axes[1,4].tick_params(axis='y', labelsize=30)
    plt.savefig(save_title, dpi=150)
    plt.show()

# Plot standard frame
for i in range(8):
    plt.imshow(A_frames[i])
    plt.show()

for i in range(1,8):
    plt.imshow(B_frames[i])
    plt.show()

# Save Frame frequence from pose 0 to pose 8
for i in range(8):
    frame_name = 'Frame_' + str(i) + '_A.jpg'
    freq_fig(frame_name, A_sim_res, i)

for i in range(1, 8):
    frame_name = 'Frame_' + str(i) + '_B.jpg'
    freq_fig(frame_name, B_sim_res, i)

# Save grading results 
columns = ['Frame_' + str(i) for i in range(8)]
A_res = pd.DataFrame(columns=columns, data=A_res)
A_res.to_csv('A_res.csv', index=False)

B_res = pd.DataFrame(columns=columns, data=B_res)
B_res.to_csv('B_res.csv', index=False)