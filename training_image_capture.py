import cv2
import os


def video_to_frames(path_output_dir, roll_no:str):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    vidcap = cv2.VideoCapture('/home/ayushsharma/test/5.mp4')
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            image = cv2.flip(image, 0)
            cv2.imwrite(os.path.join(path_output_dir, f'{roll_no} ({count}).jpeg'), image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()


roll_no = input("Enter your roll.no: ")
video_to_frames('./images', roll_no)
