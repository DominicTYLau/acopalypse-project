import numpy as np
import cv2
from matplotlib import pyplot as plt

def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1, a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]


def is_color_between(color, lower_rgb, upper_rgb):
    return all(
        lower <= value <= upper
        for lower, upper, value in zip(lower_rgb, upper_rgb, color)
    )



def findFace():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    img = cv2.imread("opencv_frame_0.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Example usage
    color = (100, 150, 200)  # RGB color to check
    lower_rgb = (152, 252, 3)  # Lower RGB value
    upper_rgb = (38, 64, 45)  # Upper RGB value

    if is_color_between(color, lower_rgb, upper_rgb):
        print("Color is between the lower and upper RGB values")
    else:
        print("Color is not between the lower and upper RGB values")


    if len(faces) == 0:
        print("No face detected")
    else:
        for x, y, w, h in faces:
            face_img = img[y : y + h, x : x + w]
            print(unique_count_app(face_img))
            cv2.imshow("Face", face_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(0)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        findFace()

cam.release()

cv2.destroyAllWindows()
