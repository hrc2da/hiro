# provides a class to hold high-level funcitons for HIRO system

#imports
from picamera import PiCamera
import pyuarm
import numpy as np
import cv2
import numpy as np
from PIL import Image, ImageOps
from pupil_apriltags import Detector
import time
import math

at_detector = Detector(families='tag36h11',nthreads=1,quad_decimate=1.0,quad_sigma=0.0,refine_edges=1,decode_sharpening=0.25,debug=0)

class HIRO():
    def __init__(self, mute=False):
        #uArm
        self.arm = pyuarm.UArm()
        self.arm.connect()
        self.speed = 100 # speed limit
        self.ground = 62 # z value to touch suction cup to ground 
        self.position = np.array([[0],[150],[150]]) # default start position
        self.arm.set_position(0, 150, 150, speed=self.speed, wait=True) #just to be safe
        self.mute = mute # controls if sounds are made of not
        # camera
        self.camera = PiCamera() #camera
        self.view = None #most recent camera image captured
    
    def disconnect(self):
        #disconnect uArm
        self.arm.disconnect()
    
    #--------------------------------------------------------------------------
    # basic movements
    #--------------------------------------------------------------------------
    
    def move(self, pos):
        '''
        move arm to pos while maintaing the angle of the end effector
        pos is a numpy array in the form [[x],[y],[z]]
        '''
        angle = 90-math.atan2(pos[0,0], pos[1,0])*180/math.pi
        if self.arm.set_position(pos[0,0], pos[1,0], pos[2,0], speed=self.speed, wait=True):
            pass
        else:
            # warnng for if move doesn't happen
            print('Requested move not possible!')
            self.beep(0)
        self.arm.set_wrist(angle,wait=True)
        self.position = pos #update position
        
    def pick_place(self, start, end):
        '''
        picks up card at start and puts it at end
        start and end are tuples of form (x,y)
        '''
        self.move(np.array([[start[0]],[start[1]],[self.ground+20]])) # hover over start
        self.move(np.array([[start[0]],[start[1]],[self.ground]])) #drop to start
        self.arm.set_pump(True) #grab card
        self.move(np.array([[start[0]],[start[1]],[self.ground+50]])) # lift card up so it doesn't mess up other cards
        self.move(np.array([[end[0]],[end[1]],[self.ground+50]])) # hover over end
        self.move(np.array([[end[0]],[end[1]],[self.ground+10]])) # lower over end
        self.arm.set_pump(False) #drop card
        self.move(np.array([[end[0]],[end[1]],[self.ground+50]])) # lift up to get out of the way
    
    #--------------------------------------------------------------------------
    # fiducial localization / transformations
    #--------------------------------------------------------------------------
    
    def capture(self, imagepath):
        '''
        takes a picture with the camera, saves it to imagepath,
        and updates view
        '''
        self.camera.capture(imagepath)
        self.view = cv2.rotate(cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE), cv2.ROTATE_180)
    
    def localize_fiducial(self, fid_num):
        '''
        Localizes center of fiducial assocalted with fid_num in workspace frame
        '''
        # conversion factor current height (assume height hasn't changed since last capture)
        pixel2mm = self.position[2,0]*0.001138
        #detect fiducials
        tags = at_detector.detect(self.view, estimate_tag_pose=False, camera_params=None, tag_size=None)
        # pick out location of desired fiducial 
        for tag in tags: # for each tag detected
            if tag.tag_id == fid_num: # if its the tag we are lookig for
                p_cam = tag.center # fiducial position in camera FoV
                break
        # frame with origin centered in FoV
        p_camcenter_pix = (p_cam[0]-512, 384-p_cam[1]) #pixels
        p_camcenter = (p_camcenter_pix[0]*pixel2mm, p_camcenter_pix[1]*pixel2mm) #convert to mm
        # wrist frame
        p_wrist = np.array([[p_camcenter[0]], [p_camcenter[1]+50], [1]]) # use array now so next step is easier
        # workspace frame
        phi = np.arctan2(self.position[0,0], self.position[1,0]) #robot angle
        T = np.array([[np.cos(phi), -np.sin(phi), self.position[0,0]],
                      [np.sin(phi),  np.cos(phi), self.position[1,0]],
                      [          0,            0,                 1]])
        p_work = T@p_wrist
        return (p_work[0,0], p_work[1,0]) #[mm]
    
    def localize_notecard(self, fid_num, beta=0):
        '''
        localizes notecard given the fiducial ID and the angle of the notecard measured
        CCW from the workspace x-axis
        Requires that the desired fiducial is in the current view
        '''
        # measure l and k on caibration card
        l = 23.4 # horizontal distance from card cener to fiducial center [mm]
        k = 13.5 # vertical distance from card cener to fiducial center [mm]
        P_f = self.localize_fiducial(fid_num) #locaiton of fiducial center
        x_nc = P_f[0]-l*np.cos(beta)+k*np.sin(beta)
        y_nc = P_f[1]-l*np.sin(beta)-k*np.cos(beta)
        return (x_nc, y_nc)
    
    def find_new_card(self, seen):
        '''
        takes in list of seen fiducial IDs and keeps looking for a new one
        with the camera until one is found and that ID is retunred
        '''
        search_pos = np.array([[0],[280],[200]]) # spot to wait at for new card
        self.move(search_pos)
        newfound = False
        print("place next card in FoV")
        self.beep(1) # alert the user of readyness
        while not newfound:
            self.capture('/home/pi/hiro/views/test.jpg') # take a picture
            tags = at_detector.detect(self.view, estimate_tag_pose=False, camera_params=None, tag_size=None)
            for tag in tags: # for each tag detected
                if tag.tag_id not in seen:
                    new_id = tag.tag_id
                    self.beep(3) #alert new card detercted
                    newfound = True
                    break
        return new_id
            
        
    #--------------------------------------------------------------------------
    # beep
    #--------------------------------------------------------------------------
    
    def beep(self, type):
        '''
        type indicates the sound played:
        0: there is a problem
        1: requesting something from user
        2: warning user that arm is about to move
        3: new card detected
        '''
        if not self.mute:
            if type == 0:
                self.arm.set_buzzer(800, 0.4, wait=True)
                self.arm.set_buzzer(600, 0.5, wait=True)
                self.arm.set_buzzer(400, 0.6, wait=True)
            elif type == 1:
                self.arm.set_buzzer(600, 0.3, wait=True)
                self.arm.set_buzzer(900, 0.3, wait=True)
                self.arm.set_buzzer(600, 0.3, wait=True)
                self.arm.set_buzzer(900, 0.3, wait=True)
                self.arm.set_buzzer(600, 0.3, wait=True)
                self.arm.set_buzzer(900, 0.3, wait=True)
            elif type == 2:
                self.arm.set_buzzer(500, 0.5, wait=True)
                self.arm.set_buzzer(300, 0.5, wait=True)
                self.arm.set_buzzer(1000, 0.5, wait=True)
            elif type == 3:
                self.arm.set_buzzer(900, .25, wait=True)
    
    
    