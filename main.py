import argparse
parser = argparse.ArgumentParser()
parser.add_argument('video', type=str, help='Input dir for videos')

def main ():
    args = parser.parse_args()

if __name__ == '__main__':
  main()