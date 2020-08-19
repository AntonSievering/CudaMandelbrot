import cv2
import os
import sys

if len(sys.argv) > 2:
    image_folder = 'frames'
    video_name = sys.argv[1]

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, int(sys.argv[2]), (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
        print(image)

    cv2.destroyAllWindows()
    video.release()
