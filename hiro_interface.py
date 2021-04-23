# provides a class to hold high-level funcitons for HIRO system

#imports
import pyuarm
import numpy as np
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
from pupil_apriltags import Detector
import time
import math
import subprocess

at_detector = Detector(families='tag36h11',nthreads=1,quad_decimate=1.0,quad_sigma=0.0,refine_edges=1,decode_sharpening=0.25,debug=0)

class HIRO():
    pixel_scalar = 0.001209
    def __init__(self, mute=False, projector=True):
        #uArm
        self.arm = pyuarm.UArm(debug=False,mac_address='FC:45:C3:24:76:EA', ble=True)
        self.arm.connect()
        self.speed = 200 # speed limit
        self.ground = 62 + 12 # z value to touch suction cup to ground 
        self.position = np.array([[0],[150],[150]]) # default start position
        self.arm.set_position(0, 150, 150, speed=self.speed, wait=True) #just to be safe
        self.mute = mute # controls if sounds are made of not
        #Projector
        self.projector = projector # controls if projections are made or not
        cv2.startWindowThread()
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        self.projection_process = None
        self.project() # start with blank projection
        # camera
        self.setup_camera()
        self.view = None #most recent camera image captured
        
    def disconnect(self):
        self.arm.disconnect() #disconnect uArm
        cv2.destroyAllWindows() # close projection
    
    def setup_camera(self, width=1024, height=768):
        # to use v4l2, must run sudo modprobe bcm2835-v4l2 to setup camera
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1) # don't capture multiple frames

    def close_camera(self):
        self.camera.release()

    def shutdown(self):
        self.close_camera()
        self.arm.set_position(0, 150, 150, speed=self.speed, wait=True)
        self.arm.disconnect()

    #--------------------------------------------------------------------------
    # basic movements
    #--------------------------------------------------------------------------
    
    def move(self, pos, wrist_mode=0, wrist_angle=0, max_tries=5, no_move_tolerance=2):
        '''
        pos is a numpy array in the form [[x],[y],[z]]
        wristmode determines how wrist position changes at end of move:
            0: wrist doesn't move
            1: wrist is moved to wristangle
            2: wrist is moved to be facing straight in the worspace accounting for the arm angle
            3: wrist is moved to be facing straight + wristangle
        returns False if move is not possible
        '''
        # move wrist
        if wrist_mode==0: # no wrist movement requested
            pass
        elif wrist_mode==1: # specified relative angle
            self.arm.set_wrist(wrist_angle,wait=True)
        elif wrist_mode==2: 
            angle = 90-math.atan2(pos[0,0], pos[1,0])*180/math.pi #calculate angle to face wrist forward
            self.arm.set_wrist(angle,wait=True)
        else: # specified absolute angle
            angle = 90-math.atan2(pos[0,0], pos[1,0])*180/math.pi-wrist_angle #calculate desired angle
            self.arm.set_wrist(angle,wait=True)
        # move arm
        if np.sign(self.position[0,0])!=np.sign(pos[0,0]) and self.position[1,0]<100 and pos[1,0]<100:
            # if we are moving fom one side to the other side of the robot's base
            rad = np.sqrt(self.position[0,0]**2+self.position[1,0]**2) #radius of current position
            self.arm.set_position(0, rad, self.position[2,0], speed=self.speed, wait=True) # go to middle first
        #print(pos) # for debugging
        if np.linalg.norm(pos-self.position) < no_move_tolerance: # no move required
            # the seet_position() fucntion returns false if the command is already the position
            # so return True here to avoid unintenionally throwing a movement failure message
            print("moving to the same place! returning...")
            return True
        else:
            num_tries = 0
            while(num_tries < max_tries):
                if self.arm.set_position(int(pos[0,0]), int(pos[1,0]), int(pos[2,0]), speed=self.speed, wait=True):
                    self.position = pos #update position
                    return True # move successful
                else:
                    print("move failed, trying again!")
                    num_tries +=1
                # time.sleep(0.5)
            else:
                # warnng for if move doesn't happen
                print(f'Requested move to {pos} from {self.position} not possible!')
                self.beep(0)
                return True # move unsuccessful
            
        
    def pick_place(self, start, end):
        '''
        picks up card at start and puts it at end
        start and end are tuples of form (x,y,angle) and (x,y) respectively
        angle is measured CCW from workspace x axis
        returns true iff all moves were successful
        '''
        if self.position[2,0] < 100:
            print(f"Raising z before move")
            position = self.position[:]
            position[2,0] = 150
            
            if not self.move(position):
                return False

        print(f"Moving to start:{[[start[0]],[start[1]],[self.ground+60]]}")
        if not self.move(np.array([[start[0]],[start[1]],[self.ground+60]]), wrist_mode=3, wrist_angle=start[2]): # hover over start
            return False
        print(f"Dropping to ground: {[[start[0]],[start[1]],[self.ground]]}")
        if not self.move(np.array([[start[0]],[start[1]],[self.ground]]), wrist_mode=0): #drop to start
            return False
        print("activating pump")
        self.arm.set_pump(True) #grab card
        time.sleep(0.5)
        print(f"Lifting card:{[[start[0]],[start[1]],[self.ground+60]]}")
        if not self.move(np.array([[start[0]],[start[1]],[self.ground+60]]), wrist_mode=0): # lift card up so it doesn't mess up other cards
            return False
        print(f"Moving to end:{[[end[0]],[end[1]],[self.ground+60]]}")
        if not self.move(np.array([[end[0]],[end[1]],[self.ground+60]]), wrist_mode=2): # hover over end
            return False
        print(f"Dropping to ground: {[[end[0]],[end[1]],[self.ground+10]]}")
        if not self.move(np.array([[end[0]],[end[1]],[self.ground+10]]), wrist_mode=0): # lower over end
            return False
        print("Deactivating pump")
        self.arm.set_pump(False) #drop card
        print(f"Retracting: {[[end[0]],[end[1]],[self.ground+40]]}")
        if not self.move(np.array([[end[0]],[end[1]],[self.ground+60]]), wrist_mode=0): # lift up to get out of the way
            return False
        return True # pick-place movements successful
    
    #--------------------------------------------------------------------------
    # fiducial localization / transformations
    #--------------------------------------------------------------------------
    
    def capture(self, imagepath=None):
        '''
        takes a picture with the camera, optionally saves it to imagepath,
        and updates view
        '''
        # self.camera.release()
        # self.camera = cv2.VideoCapture(0)
        success, frame = self.camera.read() # read the next frame (buffer length is 1)
        self.view = cv2.cvtColor(cv2.rotate(frame, cv2.ROTATE_180), cv2.COLOR_BGR2GRAY) # store the frame for apriltags
        if imagepath is not None:
            cv2.imwrite(imagepath, frame) # write image to file for parser
        return frame
    
    def localize_fiducial(self, fid_num):
        '''
        Localizes center of fiducial assocalted with fid_num in workspace frame
        returns tuple of form (x, y, angle)
        '''
        # conversion factor current height (assume height hasn't changed since last capture)
        
        pixel2mm = self.position[2,0]*self.pixel_scalar
        #detect fiducials
        tags = at_detector.detect(self.view, estimate_tag_pose=False, camera_params=None, tag_size=None)
        # pick out location of desired fiducial 
        for tag in tags: # for each tag detected
            if tag.tag_id == fid_num: # if its the tag we are lookig for
                p_cam = tag.center # fiducial position in camera FoV
                (ptA, ptB, ptC, ptD) = tag.corners # locations of four corners in camera FoV
                # beta_cam = np.arctan2(ptB[1]-ptA[1], ptB[0]-ptA[0])*180/math.pi #fiducial angle in camera frame
                
                beta_cam = np.arctan2(-ptA[1]+ptB[1], ptA[0]-ptB[0])*180/math.pi
                
                break
        # express the center of the fiducial with respect 
        # to the center of the camera frame (with the y-axis flipped to point up)
        p_camcenter_pix = (p_cam[0]-512, 384-p_cam[1]) #pixels
        # now convert that to mm from the center of the photograph
        p_camcenter = (p_camcenter_pix[0]*pixel2mm, p_camcenter_pix[1]*pixel2mm) #convert to mm
        # add an offset to get a representation wrt the wrist frame
        p_wrist = np.array([[p_camcenter[0]], [p_camcenter[1]+45.7], [1]]) # use array now so next step is easier
        # now rotate and translate  workspace frame
        phi = -np.arctan2(self.position[0,0], self.position[1,0]) #robot angle (need to verify sign)
        T = np.array([[np.cos(phi), -np.sin(phi), self.position[0,0]],
                      [np.sin(phi),  np.cos(phi), self.position[1,0]],
                      [          0,            0,                 1]])
        p_work = T@p_wrist
        
        # convert fiducial angle to world view angle
        # beta_work = 180 + beta_cam - np.arctan2(self.position[0,0], self.position[1,0])*180/math.pi
        beta_work = beta_cam + phi*180/math.pi
        
        # 
        if beta_work > 180:
            beta_work = beta_work-360 # angle correction
        return (p_work[0,0], p_work[1,0], beta_work) #[mm]
    
    def localize_notecard(self, fid_num):
        '''
        localizes notecard given the fiducial ID and the angle of the notecard measured
        CCW from the workspace x-axis
        Requires that the desired fiducial is in the current view
        returns tuple of form (x, y, angle)
        '''
        # measure l and k on caibration card
        l = 22.5 # horizontal distance from card cener to fiducial center [mm]
        k = 12.8 # vertical distance from card cener to fiducial center [mm]
        P_f = self.localize_fiducial(fid_num) #locaiton of fiducial center
        
        beta = P_f[2]*math.pi/180 #convert to radians
        x_nc = P_f[0]-l*np.cos(beta)+k*np.sin(beta)
        y_nc = P_f[1]-l*np.sin(beta)-k*np.cos(beta)
        return (x_nc, y_nc, P_f[2]) # angle remains the same
    
    def find_new_card(self, seen, reposition = False,
                        search_pos = np.array([[0],[280],[200]]),
                        reading_pos = np.array([[0],[280],[120]]),
                        reading_loc = (0,330)):
        '''
        takes in list of seen fiducial IDs and keeps looking for a new one
        with the camera until one is found and that ID is retunred
        search_pos is the position the robot waits at while it searches
        
        seen: a list of fid ids that have already been detected
        search_pos: broad FoV
        reading_pos: narrow FoV
        reading_loc: card location

        position := robot pose
        location := card location

        param reposition = False

        if reposition == False:
        search_position == reading_location

        1. go to the search position (broad FoV)
        2. wait until a card enters the field of view
        3. if reposition == False:
            1. process and return   
        4. else:
            1. localize the card
            2. pick up the card and move it to the reading location
            3. return to the reading position (close to the table)
            4. process and return
        '''
        if reposition == False:
            search_pos = reading_pos
         # spot to wait at for new card
        self.move(search_pos, wrist_mode=2)
        newfound = False
        print("place next card in FoV")
        self.project('place card above')
        self.beep(1) # alert the user of readyness
        while not newfound:
            time.sleep(1)
            self.capture(None) # take a picture but don't write to disk
            tags = at_detector.detect(self.view, estimate_tag_pose=False, camera_params=None, tag_size=None)
            for tag in tags: # for each tag detected
                if tag.tag_id not in seen:
                    new_id = tag.tag_id
                    self.beep(3) #alert new card detercted
                    self.project() # blank projection
                    newfound = True
                    break
        
        if reposition == False:
            return new_id
        else:
            cur_loc = self.localize_notecard(new_id)
            print(f'CURLOC: {cur_loc}')
            
            self.pick_place(cur_loc, reading_loc)
            self.move(reading_pos)
            time.sleep(0.5)
            # recapture the image for reading the word.
            self.camera.grab()
            cap = self.capture('/home/pi/hiro/views/view.jpg') # take a picture
            cv2.imwrite('/home/pi/hiro/views/read_imgs/%d.jpg' % new_id, cap) # save picture to memory
            # TODO: handle two failure modes possible here:
            # 1) the pick failed, in this case go back to search_pos and look for it
            # 2) the place was outside of some tolerance, in this case repick the card from view and replace it
            # for now we are not checking if the place happened and is in the right location.
            return new_id
    
 
    def sweep(self, sweep_points = [(-230, 50), (-150, 30), (-150, 200), (0, 150), (150, 200), (150, 30), (230, 50)], sweep_height=220):
        # performs sweep over workspace and returns dictionay containing updated locations of cards
        # dictionary entries in form fiducial_ID : (x,y,theta)
        # sweep_points: list of (x,y) tuples for positions to go to in sweep
        # sweep_height: heihgt sweep pictures are taken at
        self.project('starting sweep')
        time.sleep(1)
        self.project() # clear projection
        updated_locs = {} # dictionary to be returned
        seen_count = {} # dictionary to hold number of times each card is seen
        for i,sweep_point in enumerate(sweep_points):
            search_loc = np.array([[sweep_point[0]],[sweep_point[1]],[sweep_height]]) # location to take next picture
            self.move(search_loc) # move to locaiton to take picture
            self.camera.grab()
            time.sleep(0.1)
            cap = self.capture('/home/pi/hiro/views/view.jpg') # take a picture
            cv2.imwrite('/home/pi/hiro/views/sweep_imgs/%d.jpg' % i, cap) # save picture to memory
            tags = at_detector.detect(self.view, estimate_tag_pose=False, camera_params=None, tag_size=None) # detected tags
            for tag in tags: # for each tag detected
                id = tag.tag_id
                card_loc = self.localize_notecard(id)
                if id in updated_locs.keys(): # if card had already been added to dictionary
                    #sum detected locations
                    updated_locs[id] = (updated_locs[id][0]+card_loc[0], updated_locs[id][1]+card_loc[1], updated_locs[id][2]+card_loc[2])
                    seen_count[id] = seen_count[id]+1 # card seen one more time
                else:
                    # add new card and locaiton to dictionary
                    updated_locs[id] = card_loc
                    seen_count[id] = 1
            # divide by number of times card seen to find average
            for card in updated_locs:
                updated_locs[card] = (updated_locs[card][0]/seen_count[card],updated_locs[card][1]/seen_count[card],updated_locs[card][2]/seen_count[card])
        self.project('sweep complete')
        time.sleep(1)
        self.project() # clear projection
        return updated_locs

        
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
                
    #--------------------------------------------------------------------------
    # Projector
    #--------------------------------------------------------------------------
    def project(self, string=''):
        #projects an image if in project mode set in instantiation
        #projects a black screen by default
        if self.project:
            # generate image
            image_width = 864
            image_height = 480
            img = Image.new('RGB', (image_width, image_height), color=(0, 0, 0)) #black bakground
            canvas = ImageDraw.Draw(img) # create the canvas
            font_size = int(round(1300/(len(string)+1))) # adaptive font size
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", size=font_size)
            text_width, text_height = canvas.textsize(string, font=font)
            x_pos = int((image_width - text_width) / 2)
            y_pos = int((image_height - text_height) / 2)
            canvas.text((x_pos, y_pos), string, font=font, fill='#FFFFFF')
            img = img.rotate(180) #make it right-side-up
            prjimgpth = '/home/pi/hiro/projections/new_proj.jpg'
            img.save(prjimgpth)
            # full-screen projection
            # if self.projection_process is not None:
            #     self.projection_process.kill()
            # self.projection_process = subprocess.Popen(['feh', prjimgpth, '--fullscreen'])
            proj = cv2.imread('/home/pi/hiro/projections/new_proj.jpg')
            
            cv2.imshow("window", proj)
            cv2.waitKey(50)
            
            