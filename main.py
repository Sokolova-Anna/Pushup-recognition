import argparse
import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pose_tracker = mp_pose.Pose()

parser = argparse.ArgumentParser()
parser.add_argument('video', type=str, help='Input dir for videos')

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
    video_fps = vid_capture.get(cv2.CAP_PROP_FPS)
    print('FPS: ', video_fps)

  print('\nPress ESC to quit\n')

  file_count = 0
  while (vid_capture.isOpened()):
    ret, input_frame = vid_capture.read()
    if ret == True:

      # Run pose tracker
      input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
      result = pose_tracker.process(image=input_frame)
      pose_landmarks = result.pose_landmarks

      # Draw pose prediction
      output_frame = input_frame.copy()
      if pose_landmarks is not None:
        mp_drawing.draw_landmarks(
          image=output_frame,
          landmark_list=pose_landmarks,
          connections=mp_pose.POSE_CONNECTIONS)
        
      output_frame = cv2.resize(output_frame, (500, 300))
      font = cv2.FONT_HERSHEY_SIMPLEX #шрифт
      new_frame_time = time.time()
      fps = 1/(new_frame_time-prev_frame_time)
      prev_frame_time = new_frame_time
      fps = int(fps)
      fps = str(fps)
      cv2.putText(output_frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
      output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
      cv2.imshow('Frames', output_frame)
      file_count += 1
      key = cv2.waitKey(1)
      if (key == 27):
          break
    else:
      break

  vid_capture.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
