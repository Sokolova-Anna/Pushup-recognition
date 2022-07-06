import argparse
import cv2

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
    ret, frame = vid_capture.read()
    if ret == True:
      cv2.imshow('Frames', frame)
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