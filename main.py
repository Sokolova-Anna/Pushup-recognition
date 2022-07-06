import argparse
import cv2
import mediapipe as mp

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