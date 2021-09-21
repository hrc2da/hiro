import picamera
from time import sleep
import cv2
import pyuarm

arm = pyuarm.UArm()
arm.connect()
arm.set_position(0,200,240)

#width=4032
#height=3040
width=1920
height=1200
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#camera.set(cv2.CAP_PROP_EXPOSURE, 1000) 
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1) # don't capture multiple frames

i = 0
while True:
    success, frame = camera.read()
    if not success:
        continue
    cv2.imshow('Preview', cv2.resize(frame,(1920,1080)))
    key = cv2.waitKey(1)
    if key%256 == 27:
        break # escape
    elif key == ord('c'):
        # c - key means capture
        print(f"capturing to chessboard_img_fhd/image{i}_0.jpg")
        cv2.imwrite(f'chessboard_img_fhd/image{i}_0.jpg', frame)
        i += 1

camera.release()
cv2.destroyAllWindows()    


# camera = picamera.PiCamera()
# camera.resolution(4032,3040)

# camera.start_preview()
# for i in range(15):
#     sleep(5)
#     camera.capture('chessboard_img/image%s.jpg' % i)
# camera.stop_preview()