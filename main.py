import argparse
import cv2
import mediapipe as mp
import time
import numpy as np
import io
import csv
import os
from pose_classifier import PoseClassifier
from EMA_smoothing import EMADictSmoothing
from embedder import FullBodyPoseEmbedder
from rep_count import RepetitionCounter
from PoseClassification import PoseClassificationVisualizer

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pose_tracker = mp_pose.Pose()

parser = argparse.ArgumentParser()
parser.add_argument('video', type=str, help='Input dir for videos')

pose_samples_folder = 'fitness_poses_csvs_out'

pose_embedder = FullBodyPoseEmbedder()

pose_classifier = PoseClassifier(
    pose_samples_folder=pose_samples_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

pose_classification_filter = EMADictSmoothing(
    window_size=10,
    alpha=0.2)

class_name='pushups_down'
repetition_counter = RepetitionCounter(
    class_name=class_name,
    enter_threshold=6,
    exit_threshold=4)

def main():
  args = parser.parse_args()

  vid_capture = cv2.VideoCapture(args.video)
  if (vid_capture.isOpened() == False):
    print("Error opening video file")
  else:
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0
    # ширина видео
    frame_width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
     #высоты видео
    frame_height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_n_frames = vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = vid_capture.get(cv2.CAP_PROP_FPS)
    print('FPS: ', video_fps)

   # Initialize renderer.
  pose_classification_visualizer = PoseClassificationVisualizer(
        class_name=class_name,
        plot_x_max=video_n_frames,
        # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
        plot_y_max=10)
  print('\nPress ESC to quit\n')

  file_count = 0
  while (vid_capture.isOpened()):
    ret, input_frame = vid_capture.read()
    if ret == True:

      # Run pose tracker
      start_time=time.time()
      output_frame = input_frame.copy()
      font = cv2.FONT_HERSHEY_SIMPLEX
      input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
      result = pose_tracker.process(image=input_frame)
      pose_landmarks = result.pose_landmarks
      end_time=time.time()
      inf_time=end_time - start_time
      inf_time = str(inf_time)
      mess="inf_time:"
      cv2.putText(output_frame, mess, (470, 140), font, 1, (100, 255, 0), 3, cv2.LINE_AA) 
      cv2.putText(output_frame, inf_time, (470, 170), font, 1, (100, 255, 0), 3, cv2.LINE_AA) 

      # Draw pose prediction
      output_frame = input_frame.copy()
      if pose_landmarks is not None:
        mp_drawing.draw_landmarks(
          image=output_frame,
          landmark_list=pose_landmarks,
          connections=mp_pose.POSE_CONNECTIONS)

      if pose_landmarks is not None:
        # Get landmarks.
        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
        pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                  for lmk in pose_landmarks.landmark], dtype=np.float32)
        assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

        # Classify the pose on the current frame.
        pose_classification = pose_classifier(pose_landmarks)
        #print(pose_classification)

        # Smooth classification using EMA.
        pose_classification_filtered = pose_classification_filter(pose_classification)

        repetitions_count = repetition_counter(pose_classification_filtered)
      
      #Draw classification plot and repetition counter.
      output_frame = pose_classification_visualizer(
        frame=output_frame,
        pose_classification=pose_classification,
        pose_classification_filtered=pose_classification_filtered,
        repetitions_count=repetitions_count)

      output_frame = cv2.resize(output_frame, (500, 300))

      output_frame = cv2.resize(output_frame, (500, 300))
      font = cv2.FONT_HERSHEY_SIMPLEX #шрифт
      new_frame_time = time.time()
      fps = 1/(new_frame_time-prev_frame_time)
      prev_frame_time = new_frame_time
      fps = int(fps)
      fps = str(fps)
      mess_fps="FPS:"
      cv2.putText(output_frame, mess_fps, (370, 140), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
      cv2.putText(output_frame, fps, (440,140), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
      output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
      cv2.imshow('Frames', output_frame)
      file_count += 1
      key = cv2.waitKey(1)
      if (key == 27):
          break
    else:
      break

  print('repeats = ' + str(repetitions_count))
  vid_capture.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()