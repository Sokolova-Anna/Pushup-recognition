import argparse
parser = argparse.ArgumentParser()
parser.add_argument('video', type=str, help='Input dir for videos')
args = parser.parse_args()

import cv2

vid_capture = cv2.VideoCapture(args.video)
if (vid_capture.isOpened() == False):
  print("Error opening video file")

print('\nPress ESC to quit\n')

file_count = 0
while (vid_capture.isOpened()):
    ret, frame = vid_capture.read()
    if ret == True:
        cv2.imshow('Window', frame)
        file_count += 1
        print('Frame {0:04d}'.format(file_count))
        key = cv2.waitKey(20)
        if (key == 27):
            break
    else:
        break

vid_capture.release()
cv2.destroyAllWindows()