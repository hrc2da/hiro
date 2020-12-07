import cv2
import numpy as np
from pupil_apriltags import Detector
import time
import glob
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage.transform import resize

at_detector = Detector(families='tag36h11',nthreads=1,quad_decimate=1.0,quad_sigma=0.0,refine_edges=1,decode_sharpening=0.25,debug=0)

# overhead_path = '/home/dev/scratch/armtui/arm_fixed_base_and_z_200/images/*.png'
overhead_path = '/home/dev/hiro_data/12_6/images/*.png'
overhead_files = glob.glob(overhead_path)
overhead_files.sort()
targets = []


# arm_path = '/home/dev/scratch/armtui/arm_fixed_base_and_z_200/pipics/*.png'
arm_path = '/home/dev/hiro_data/12_6/pipics/*.png'
arm_files = glob.glob(arm_path)
arm_files.sort()
images = []
grayscale_images = []
target_fields = []


for i, overheadname in enumerate(overhead_files):
    arm_img = cv2.imread(arm_files[i])
    arm_img_gray = cv2.imread(arm_files[i], cv2.IMREAD_GRAYSCALE)
    overhead_img = cv2.imread(overheadname, cv2.IMREAD_GRAYSCALE)
    _, overhead_img = cv2.threshold(overhead_img, 50,255,cv2.THRESH_BINARY)
    tags = at_detector.detect(overhead_img, estimate_tag_pose=False, camera_params=None, tag_size=None)
    
    if len(tags) > 0:
        try:
            # images.append(arm_img[475:575,790:790+100,:]) #crop to square
            images.append(resize(arm_img, (125,200)))
            # grayscale_images.append(arm_img_gray[475:575,790:790+100])
            grayscale_images.append(resize(arm_img_gray, (125,200)))
            loc = tags[0].center
            # import pdb; pdb.set_trace()
            targets.append(loc)
            tf = np.zeros((480,600))
            tf[int(loc[1]),int(loc[0])] = 1
            target_fields.append(tf)
        except:
            import pdb; pdb.set_trace()
    else:
        print(f"No detection in {overheadname}")
    # img[int(loc[0])][int(loc[1])] = 0
    # for corner in tags[0].corners:
    #     x,y = map(int,corner)
    #     img[x][y] = 0
    # fig,ax = plt.subplots(1)
    # ax.imshow(img)
    # circle = patches.Circle(loc, radius=5, edgecolor='r', facecolor='r')
    # rect = patches.Polygon(tags[0].corners,linewidth=1,edgecolor='r',facecolor='none')
    # ax.add_patch(rect)
    # ax.add_patch(circle)
    # plt.show()
    # import pdb; pdb.set_trace()

np.save('armpicscolor.npy', np.array(images))
np.save('armpicsgray.npy', np.array(grayscale_images))
np.save('locs.npy', np.array(targets))
np.save('loc_fields.npy', np.array(target_fields))

print(f"Total Number of Samples: {len(targets)}") #3394




# while(t<end_time):
#     t = time.time() - t0

#     # take webcam picture
#     ret, frame = webcam.read() # Capture frame-by-frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Our operations on the frame come here
#     cv2.imshow('frame',gray)# Display the resulting frame
#     cv2.imwrite('/home/pi/HRCD/Apriltag Test Pictures/pic.jpg', gray)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'): # need to include this for preview to work
#         break
#     img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
#     #gray_image = np.array(ImageOps.grayscale(img))

#     tags = at_detector.detect(img, estimate_tag_pose=False, camera_params=None, tag_size=None)
#     if tags:
#         print("tag detected")
#     else:
#         print("OH NO! NO TAG! PANIC!")

# webcam.release()
# cv2.destroyAllWindows()

"""
import numpy as np
import cv2
import time
from io import BytesIO
from PIL import Image, ImageOps
from pupil_apriltags import Detector


stream = BytesIO()
at_detector = Detector(families='tag36h11',nthreads=1,quad_decimate=1.0,quad_sigma=0.0,refine_edges=1,decode_sharpening=0.25,debug=0)

webcam = cv2.VideoCapture(1)

t0 = time.time() #start time
t = 0 # time that's passed in seconds
end_time = 60*0.5 # total time for data collection

# data collection loop
while(t<end_time):
    t = time.time() - t0

    # take webcam picture
    ret, frame = webcam.read() # Capture frame-by-frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Our operations on the frame come here
    cv2.imshow('frame',gray)# Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    gray_image = np.array(ImageOps.grayscale(frame))

    tags = at_detector.detect(frame, estimate_tag_pose=False, camera_params=None, tag_size=None)
    print(tags)
    time.sleep(2)
    
webcam.release()
cv2.destroyAllWindows()
"""