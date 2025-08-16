import cv2
import mediapipe as mp
import numpy as np
import numpy.linalg as LA
import itertools
# Key Triangle:
# {0, 11, 12}, {0, 13, 14}, {0, 15, 16}, {0, 23, 24}, {0, 25, 26}, {0,27,28}
class video_grade():
    def __init__(self, standard_frame_marks, frame_th_list, target_video_url):
        self.std_mark_list = standard_frame_marks
        self.frame_th_list = frame_th_list
        self.tar_url = target_video_url
        self.key_triangle_point = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def marks_out(self):
        cap = cv2.VideoCapture(self.tar_url)
        processed_img_list, marks = [], []
        pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detect pose
            results = pose.process(image)
            marks.append(results.pose_landmarks)
            # Draw pose landmarks
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            processed_img_list.append(image)
        cap.release()
        self.marks = marks[1:]
        self.processed_img = processed_img_list[1:]

    def angle_one(self, ind_orig, ind_1, ind_2):
        vec_1 = ind_1 - ind_orig
        vec_2 = ind_2 - ind_orig
        return np.dot(vec_1.reshape(1,-1), vec_2.reshape(-1,1)) / (LA.norm(vec_1)*LA.norm(vec_2))

    def triangle_one(self, ang_0, ang_1, ang_2):
        cos_0 = self.angle_one(ang_0, ang_1, ang_2).item()
        cos_1 = self.angle_one(ang_1, ang_0, ang_2).item()
        cos_2 = self.angle_one(ang_2, ang_0, ang_1).item()
        return [cos_0, cos_1, cos_2]

    def angle_distance(self, ang_0, ang_1):
        return np.mean(np.abs(ang_0 - ang_1))

    def img_one_triangle(self, land_mark, key_point_list):
        triangle_list, land_mark_list, triangle_cos_list = [], [], []
        combinations_generator = itertools.combinations(key_point_list, 3)
        for combo in combinations_generator:
            triangle_list.append(combo)
        for mark in land_mark.landmark:
            land_mark_list.append([mark.x, mark.y])
        land_mark_list = np.array(land_mark_list)
        for triangle in triangle_list:
            ang_0, ang_1, ang_2 = triangle
            # print(land_mark_list[ang_0])
            triangle_cos_list.append(self.triangle_one(land_mark_list[ang_0], land_mark_list[ang_1], land_mark_list[ang_2]))
        return np.array(triangle_cos_list)

    def grading_main(self):
        self.marks_out()
        grade_results, detect_frames = [],[]
        frame_matrix_list = []
        frame_sim_res_all = []
        sample_shape = self.img_one_triangle(self.std_mark_list[0], self.key_triangle_point).shape
        for video_frame in self.marks:
            try:
                frame_matrix = self.img_one_triangle(video_frame, self.key_triangle_point)
                frame_matrix_list.append(frame_matrix)
            except AttributeError:
                frame_matrix_list.append(np.ones(shape = sample_shape)*2)
        for i in range(len(self.std_mark_list)):
            img_std = self.img_one_triangle(self.std_mark_list[i], self.key_triangle_point)
            sim_res = []
            for video_frame in frame_matrix_list:
                sim_res.append(self.angle_distance(img_std, video_frame))
            frame_target = np.argmin(sim_res)
            frame_sim_res_all.append(sim_res)
            detect_frames.append(self.processed_img[frame_target])
            if sim_res[frame_target] < self.frame_th_list[i]:
                grade_results.append(1)
            else:
                grade_results.append(0)
        return grade_results, detect_frames, frame_sim_res_all
