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
from picamera import PiCamera
from undistort import undistort_img
import pickle as pkl
import requests
import base64
import threading
from config import Config
from simplejpeg import encode_jpeg
from sklearn.cluster import DBSCAN


at_detector = Detector(families='tag36h11',nthreads=1,quad_decimate=1.0,quad_sigma=0.0,refine_edges=1,decode_sharpening=0.25,debug=0)


# https://stackoverflow.com/questions/30032063/opencv-videocapture-lag-due-to-the-capture-buffer
# Define the thread that will continuously pull frames from the camera
class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.stop_flag = False
        self.capture_flag = False
        print("initializing camera thread")
        self.start()

    def capture(self):
        self.capture_flag = True
        while self.capture_flag == True:
            time.sleep(0.01)
        return self.last_frame

    def stop(self):
        print("stopping")
        self.stop_flag = True
    def run(self):
        while True:
            if self.capture_flag is True:
                ret, self.last_frame = self.camera.read()
                self.capture_flag = False
            else:
                self.camera.grab()
            time.sleep(0.01)
            if self.stop_flag is True:
                break


class HIROLightweight():
    # pixel_scalar = 0.00115964489 #0.001209
    #pixel to mm conversion (at different heights, etc.)
    pixel_scalar = 0.002046951936
    # pixel_intercept = 0.062
    pixel_intercept = 0.134
    #calibration for detection (fish-eyed distortion fix) and locomotion --> linear regressor on data
    move_calibration_file = "random_forest_locomotion_regressor.pkl"
    view_calibration_file = "random_forest_view_regressor.pkl"
    api_url = Config.API_URL
    def __init__(self, mute=False, projector=True):
        #uArm
        self.arm = pyuarm.UArm(debug=False,mac_address='FC:45:C3:24:76:EA', ble=False)
        
        self.arm.connect()
        
        self.speed = 200 # speed limit
        self.ground = 30 # z value to touch suction cup to ground 
        self.position = np.array([[0],[150],[150]]) # default start position
        self.arm.set_position(0, 150, 150, speed=self.speed, wait=True) #just to be safe
        self.mute = mute # controls if sounds are made of not
        #Projector
        self.projector = projector # controls if projections are made or not
        # commenting out projection
        ####cv2.startWindowThread()
        ####cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        ####cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        self.projection_process = None
        # commenting out projection
        ####self.project() # start with blank projection
        # camera
        self.setup_camera()
        self.setup_calibrators()
        self.view = None #most recent camera image captured
        self.tempimgpath = '/tmp/hirocurrentframe.jpg'
        
    def disconnect(self):
        self.arm.disconnect() #disconnect uArm
        cv2.destroyAllWindows() # close projection

    def setup_calibrators(self):
        with open(self.move_calibration_file, 'rb') as infile:
            self.move_x_regr, self.move_y_regr = pkl.load(infile)
        with open(self.view_calibration_file, 'rb') as infile:
            self.view_x_regr, self.view_y_regr = pkl.load(infile)

    
    def offset_move_pos(self,pos):
        x = pos[0,0]
        y = pos[1,0]
        # a=0.6589183388224794
        # b=-0.004368464940542216
        # c=0.0011403865227060843
        # offsetx = a*x + 0.95*b*y + c*x*y
        # d=0.24410060062390104
        # e=0.8834219026112635
        # f=-0.001073335224126937
        # offsety = d*x + e*y + f*x*y
        # # offsety = 0.98*d*x + 1.35*e*y + f*x*y -65 # post hoc magic number!
        # return np.array([[offsetx], [offsety], pos[2]])
        return np.array([self.move_x_regr.predict([(x,y)]), self.move_y_regr.predict([(x,y)]), pos[2]])
    def offset_view_loc(self,loc):
        x,y,rot = loc
        # a=1.05924180288876
        # b=-0.03637635864768285
        # c=-5.933499577953245
        # # offsetx = a*x + 0.9*b*y + c +10
        # offsetx = a*x + b*y + c

        # a=0.020185365200176768
        # b=1.0818751885134938
        # c=-17.457824323173057
        # offsety = a*x + 1*b*y + c
        offsetx = self.view_x_regr.predict([(x,y)])[0]
        offsety = self.view_y_regr.predict([(x,y)])[0]
        return (offsetx, offsety, rot)

    def setup_camera(self, width=1024, height=768):
        # to use v4l2, must run sudo modprobe bcm2835-v4l2 to setup camera
        # width=4032
        # height=3040
        width = self.width =1920
        height = self.height = 1200
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # self.camera.set(cv2.CAP_PROP_EXPOSURE, 1000) 
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1) # don't capture multiple frames
        self.cam_cleaner = CameraBufferCleanerThread(self.camera)
        # wait for camera to initialize
        self.cam_cleaner.capture()
        while self.cam_cleaner.last_frame is None:
            time.sleep(0.01)
        print("camera started")
        
        
        # self.camera = PiCamera()
        # self.camera.resolution = (4032,3040)

    def close_camera(self):
        self.camera.release()
        self.cam_cleaner.stop()
        # self.camera.close()

    def shutdown(self):
        self.close_camera()
        self.arm.set_position(0, 150, 150, speed=self.speed, wait=True)
        self.arm.disconnect()

    #--------------------------------------------------------------------------
    # basic movements
    #--------------------------------------------------------------------------
    
    def move(self, pos, wrist_mode=0, wrist_angle=0, max_tries=5, no_move_tolerance=2, speed=None, wait=True):
        '''
        pos is a numpy array in the form [[x],[y],[z]]
        wristmode determines how wrist position changes at end of move:
            0: wrist doesn't move
            1: wrist is moved to wristangle
            2: wrist is moved to be facing straight in the worspace accounting for the arm angle
            3: wrist is moved to be facing straight + wristangle
        offset is the offset in r (in polar coordinates) of the arm, which tends to overshoot the mark
        -- this is going to be different at different heights
        -- we probably care strongly about getting it right at two heights: the ground, and the localization point
        -- it also seems like there might be a dependence on r itself, but we are not accounting for that here
        returns False if move is not possible
        '''
        if speed is None:
            speed = self.speed
        if pos is not None:
            # r = np.sqrt(pos[0]**2+pos[1]**2)
            # print(f"r: {r}")
            # if r > 180:
            #     offset *= 350/r # this is crazy but trying to scale by r
            # theta = np.arctan2(pos[1],pos[0])
            # x_off = offset*np.cos(theta) 
            # y_off = offset*np.sin(theta)
            # pos[0,0] += x_off
            # pos[1,0] += y_off
            # print(f'theta: {theta}, x_offset: {x_off}, y_offset: {y_off}')
            pos = self.offset_move_pos(pos)
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
            self.arm.set_position(0, rad, self.position[2,0], speed=speed, wait=wait) # go to middle first
        #print(pos) # for debugging
        if np.linalg.norm(pos-self.position) < no_move_tolerance: # no move required
            # the seet_position() fucntion returns false if the command is already the position
            # so return True here to avoid unintenionally throwing a movement failure message
            print("moving to the same place! returning...")
            return True
        else:
            num_tries = 0
            while(num_tries < max_tries):
                if self.arm.set_position(int(pos[0,0]), int(pos[1,0]), int(pos[2,0]), speed=speed, wait=wait):
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
        print(f"Dropping to ground: {[[start[0]],[start[1]],[self.ground-10]]}")
        if not self.move(np.array([[start[0]],[start[1]],[self.ground-10]]), wrist_mode=0): #drop to start
            return False
        print("activating pump")
        self.arm.set_pump(True) #grab card
        time.sleep(1)
        print(f"Lifting card:{[[start[0]],[start[1]],[self.ground+60]]}")
        if not self.move(np.array([[start[0]],[start[1]],[self.ground+60]]), wrist_mode=0): # lift card up so it doesn't mess up other cards
            return False
        print(f"Moving to end:{[[end[0]],[end[1]],[self.ground+60]]}")
        if not self.move(np.array([[end[0]],[end[1]],[self.ground+60]]), wrist_mode=2): # hover over end
            return False
        print(f"Dropping to ground: {[[end[0]],[end[1]],[self.ground+10]]}")
        if not self.move(np.array([[end[0]],[end[1]],[self.ground]]), wrist_mode=0): # lower over end
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
    def undistort_img(self, img):
        _, img_encoded = cv2.imencode('.jpg', img)
        r = requests.post(self.api_url+"/undistort", data=img_encoded.tostring())
        img_decoded = cv2.imdecode(np.fromstring(r.content, np.uint8), cv2.IMREAD_COLOR)
        return img_decoded

    def capture(self, imagepath=None, raw=False):
        '''
        takes a picture with the camera, optionally saves it to imagepath,
        and updates view
        if raw is true, don't undistor the image for fishehye (useful e.g. if we are capturing new images for distortion calibration)
        '''
        # self.camera.release()
        # self.camera = cv2.VideoCapture(0)
        while self.arm.get_is_moving():
            print("waiting for stop")
            time.sleep(0.01)
        self.cam_cleaner.capture()
        frame = self.cam_cleaner.last_frame # read the next frame (buffer length is 1)
        if not raw:
            print("undistorting")
            frame = self.undistort_img(frame)
            print("undistorted")
        # self.view = cv2.cvtColor(cv2.rotate(frame, cv2.ROTATE_180), cv2.COLOR_BGR2GRAY) # store the frame for apriltags
        self.view = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if imagepath is not None:
            cv2.imwrite(imagepath, frame) # write image to file for parser
        return frame
        # for Picamera:
        # self.camera.capture(self.tempimgpath)
        # frame = cv2.imread(self.tempimgpath)
        # self.view = cv2.cvtColor(cv2.rotate(frame, cv2.ROTATE_180), cv2.COLOR_BGR2GRAY) #TODO: check if you need to rotate still with picamera
        # if imagepath is not None:
        #     cv2.imwrite(imagepath, frame) # write image to file for parser
        # return frame

    def localize_fiducial(self, fid_num):
        '''
        Localizes center of fiducial assocalted with fid_num in workspace frame
        returns tuple of form (x, y, angle)
        '''
        # conversion factor current height (assume height hasn't changed since last capture)
        
        pixel2mm = self.position[2,0]*self.pixel_scalar + self.pixel_intercept
        # import pdb; pdb.set_trace()
        #detect fiducials
        print("checking fiducials")
        tags = at_detector.detect(self.view, estimate_tag_pose=False, camera_params=None, tag_size=None)
        # import pdb; pdb.set_trace()
        # pick out location of desired fiducial 
        print(f"found {len(tags)} tags")
        for tag in tags: # for each tag detected
            if tag.tag_id == fid_num: # if its the tag we are lookig for
                p_cam = tag.center # fiducial position in camera FoV
                (ptA, ptB, ptC, ptD) = tag.corners # locations of four corners in camera FoV
                # beta_cam = np.arctan2(ptB[1]-ptA[1], ptB[0]-ptA[0])*180/math.pi #fiducial angle in camera frame
                
                beta_cam = np.arctan2(-ptA[1]+ptB[1], ptA[0]-ptB[0])*180/math.pi
                print("found it")
                break
        # import pdb; pdb.set_trace()
                
        # express the center of the fiducial with respect 
        # to the center of the camera frame (with the y-axis flipped to point up)
        width= self.width #4032
        height= self.height #3040
        x_trans = width/2
        y_trans = height/2
        p_camcenter_pix = (p_cam[0]-x_trans, y_trans-p_cam[1]) #pixels
        # now convert that to mm from the center of the photograph
        p_camcenter = (p_camcenter_pix[0]*pixel2mm, p_camcenter_pix[1]*pixel2mm) #convert to mm
        # add an offset to get a representation wrt the wrist frame
        offset_from_end_effector_to_cam_center_z = 55 #mm
        offset_from_end_effector_to_cam_center_y = -25 #mm
        p_wrist = np.array([[p_camcenter[0]+offset_from_end_effector_to_cam_center_y], [p_camcenter[1]+offset_from_end_effector_to_cam_center_z], [1]]) # use array now so next step is easier
        # now rotate and translate  workspace frame
        y_err = 10 # an offset in the y due to error in the robot's pose of about 1 cm (height is good) ASSUMING WE ARE AT (0,200,200)
        phi = -np.arctan2(self.position[0,0], self.position[1,0]+ y_err) #robot angle (need to verify sign)
        T = np.array([[np.cos(phi), -np.sin(phi), self.position[0,0]],
                      [np.sin(phi),  np.cos(phi), self.position[1,0]+y_err],
                      [          0,            0,                 1]])
        p_work = T@p_wrist
        
        # convert fiducial angle to world view angle
        # beta_work = 180 + beta_cam - np.arctan2(self.position[0,0], self.position[1,0])*180/math.pi
        beta_work = beta_cam + phi*180/math.pi
        
        # 
        if beta_work > 180:
            beta_work = beta_work-360 # angle correction
        # p_work[0,0] -= 30 # there seems to be a consistent x-error of ~ 3cm
        return (p_work[0,0], p_work[1,0], beta_work) #[mm]
    
    def localize_notecard(self, fid_num, raw=False):
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
        ### DANGER: MATT IS CHANGING THIS FOR VIEW CALIBRATION!!!!
        # return (x_nc, y_nc, P_f[2]) # angle remains the same
        print(f"CURLOC_BEFORE_TRANFORM: {(x_nc, y_nc, P_f[2])}")
        if raw:
            return (x_nc, y_nc, P_f[2])
        else:
            return self.offset_view_loc((x_nc, y_nc, P_f[2]))

    def get_fiducial_map(self, search_pos = np.array([[0],[200],[240]]), mask=None, raw=False):
        '''
        Move to the search location, take a picture, and get the locations of all fiducials seen in the image
        '''
        self.move(search_pos, wait=True)
        time.sleep(3) # stabilize before taking photo
        frame = self.capture(raw=True,imagepath="temp.jpg")
        if mask is not None:
            x,y,w,h = mask
            black_frame = np.zeros(frame.shape, dtype=np.uint8)
            black_frame[y:y+h,x:x+w,:] = frame[y:y+h,x:x+w,:]
            frame = black_frame
        # cv2.imwrite("temp2.jpg", frame)
        start = time.time()
        # _, img_encoded = cv2.imencode('.jpg', frame)
        img_encoded = encode_jpeg(frame)
        #base64 encode the img
        img_encoded = base64.b64encode(img_encoded).decode('utf-8')
        print(f"Encoding time: {time.time()-start}")
        pixel2mm = float(self.position[2,0]*self.pixel_scalar + self.pixel_intercept)
        print("getting fiducial map")
        start = time.time()
        r = requests.post(self.api_url+"/get_fiducials", json={"image":img_encoded,"pixel2mm":pixel2mm, "pose":self.position.tolist()})
        print(f"Request time: {time.time()-start}")
        print("request back")
        res = r.json()
        self.fiducial_map = res['location_dict']
        self.raw_fiducial_map = res['location_dict']
        if raw:
            # use raw for calibration
            return {int(fid):loc for fid, loc in res['location_dict'].items()}
        else:
            self.fiducial_map = {int(fid):self.offset_view_loc(loc) for fid, loc in res['location_dict'].items()}
            return self.fiducial_map
    
    def checkzone(self,loc,zone):
        x = loc[0]
        y = loc[1]
        xmin, ymin = zone[0]
        xmax, ymax = zone[1]
        if x > xmin and x < xmax and y > ymin and y < ymax:
            return True
        return False

    def checkrange(self,loc):
        x,y,z = loc
        if np.abs(x) > 320:
            return False
        if y < 130 and np.abs(x) < 130:
            return False
        if np.linalg.norm((x,y)) > 360:
            return False
        return True

    def checkpickplacerange(self,loc1,loc2):
        return self.checkrange(loc1) and self.checkrange(loc2)

    def getmoves(self, new_locs, old_locs=None, dummy_spot=[-200,75,0], add_fid= None, add_zone=None, threshold=15):
        if old_locs is None:
            old_locs = self.fiducial_map
        # go through the dict of new locs, see if the fiducial id is in old locs
        adds = []
        moves = []
        removes = []
        invalids = []
        
        for k,v in new_locs.items():
            if k not in old_locs: 
                if self.checkrange(v):
                    if add_fid and k != add_fid: # error: v is the target location, not the current location
                        print("add out of zone")
                        invalids.append((k,v))
                    else:
                        adds.append((k,v))
                else:
                    invalids.append((k,v))
            # if the move is greater than the threshold, otherwise ignore
            elif np.linalg.norm(np.array(v[:2])-np.array(old_locs[k][:2])) > threshold:
                if self.checkpickplacerange(old_locs[k],v):
                    moves.append((k,old_locs[k],v))
                else:
                    invalids.append((k,old_locs[k],v))
        for k,v in old_locs.items():
            if k not in new_locs:
                if self.checkrange(old_locs[k]):
                    removes.append((k,v))
                else:
                    invalids.append((k,v))

        
        moves = self.sort_moves(moves, dummy_spot)
        return adds,moves,removes,invalids
        # if it is not, add it to the list of things to move
        # if it is, check to see if there is a distance of greater than threshold
        # if there is, then add it to the list of moves
        # finally, see if there are any fids in old_locs that aren't in new_locs
        # remove those
        
    def sort_moves(self, moves, dummy_spot, threshold=60):
        i = 0
        new_moves = moves
        dummy_moves = []
        while i < len(new_moves):
            print(i)
            fid_i, source_i, destination_i = new_moves[i]
            last_overlap = i
            for j in range(i+1,len(new_moves)):
                # get the last overlap in the list
                _,source_j,destination_j = new_moves[j]
                if np.linalg.norm(np.array(destination_i[:2])-np.array(source_j[:2])) < threshold:
                    last_overlap = j
            if last_overlap > i:
                # move the card after last overlap
                print(f"swapping {new_moves[i]} and {new_moves[last_overlap]}")
                _,source_target,destination_target = new_moves[last_overlap]
                # first, if the cards overlap each other, setup a dummy swap. we'll add the dummy swaps later back, at the end.
                if np.linalg.norm(np.array(destination_target[:2])-np.array(source_i[:2])) < threshold:
                    print("too close: dummy swapping")
                    # dummy_spot = next(dummy_generator)
                    dummy_move_start = [fid_i, source_i, dummy_spot]
                    dummy_move_end = [fid_i, dummy_spot, destination_target]
                    # new_moves = moves[:i] + [dummy_move_start] + moves[i+1:last_overlap] + [dummy_move_end] + moves[last_overlap+1:]
                    new_moves = new_moves[:i] + [dummy_move_start] + new_moves[i+1:] # replace moves[i]
                    dummy_moves.append(dummy_move_end)
                else:
                    # else remove the move and put it after last_overlap (remember upper bound is exclusive)
                    # note that we don't increment i here because i is now the element after the one we just moved
                    new_moves = new_moves[:i] + new_moves[i+1:last_overlap+1] + [new_moves[i]] + new_moves[last_overlap+1:]
            else:
                i += 1
        # add all the dummy move endings in reverse order (we are picking up off a stack)
        if len(dummy_moves) > 0:
            new_moves = new_moves + list(reversed(dummy_moves))

        return new_moves



    def wait_for_card(self, loading_zone=[(-300,0),(-200,100)], timeout=None):
        '''
        Wait for a card to show up in the loading zone
        '''
        time_started = time.time()
        zone_mask = None
        while True:
            if timeout is not None:
                if time.time() - time_started > timeout:
                    break
            start = time.time()
            fmap = self.get_fiducial_map(mask=zone_mask)#(1200,1500,500,500))
            print(f"{(time.time()-start)} s")
            print(fmap)
            print("checking for zone")
            for fid,loc in fmap.items():
                if self.checkzone(loc, loading_zone):
                    self.social_move("wake")                    
                    print(f"Found {fid} at {loc}!")
                    # self.beep(1)
                    if zone_mask is not None:
                        fmap = self.get_fiducial_map(mask=None)
                    fmap.pop(fid) # remove the fid from the current diagram
                    return fid,loc,fmap
            
            self.social_move("breathe")

    
    def social_move(self,move, start_pos = None, speed = None):
        print("social move")
        if start_pos is None:
            start_pos = np.array([[0],[200],[240]])
        if speed is None:
            speed = self.speed
        if move is None:
            print("no move")
            return  
        
        elif move == 'wake':
            print("waking up")
            locs = np.array(list(self.fiducial_map.values()))[:,:2]
            if len(locs) > 1:
                # quick and dirty cluster the list and get the centroids
                # hover over the centroids
                clustering = DBSCAN(eps=100, min_samples=2).fit(locs)
                n_clusters = max(clustering.labels_) + 1
                print("I am here")
                centroids = [np.mean(locs[clustering.labels_==i], axis=0) for i in range(n_clusters)]
                for c in centroids:
                    self.move(np.array([[c[0]],[c[1]],[150]]), wait=True)
                    time.sleep(0.5)

            else:
                self.move(np.array([[-30],[200],[150]]))
                self.move(np.array([[-250],[150],[150]]),wrist_mode=1, wrist_angle=45)
                self.move(np.array([[-250],[150],[150]]),wrist_mode=1, wrist_angle=0)
                self.move(np.array([[-250],[150],[150]]),wrist_mode=1, wrist_angle=45)
                self.move(np.array([[30],[200],[150]]),wrist_mode=1, wrist_angle=0)
                self.move(np.array([[0],[250],[250]]))
            self.move(np.array([[0],[200],[240]]))
        elif move == 'follow':
            print("following")
            self.move(start_pos,wait=True)
            frame = self.capture(raw=True, imagepath="tempfollow.jpg")
            # frame = cv2.imread("/home/pi/hiro/hand_detection/data/1.jpg")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert color format b/c backend uses PIL
            img_encoded = encode_jpeg(frame)
            img_encoded = base64.b64encode(img_encoded).decode('utf-8')
            resp = requests.post(self.api_url+"/gethand", json={"image":img_encoded}).json()
            side = resp["side"]
            output = resp["output"]
            print(f"{output},{side}")
            if side == 1: #right
                self.move(np.array([[-100],[200],[200]]),wait=True,speed=4)
                cv2.imwrite(f"temp_right_{time.time()}.jpg",frame)
                time.sleep(1)
            elif side == 2:
                self.move(np.array([[100],[200],[200]]),wait=True, speed=4)
                cv2.imwrite(f"temp_left_{time.time()}.jpg",frame)
                time.sleep(1)
            else:
                return False
            return True
        elif move == 'breathe':
            print("breathing")
            x1 = 0 #np.random.random()*30-15
            x2 = np.random.random()*30-15
            z1 = (np.random.random()*20-10)+215
            z2 = (np.random.random()*20-10)+190
            self.move(np.array([[x1],[170],[z1]]),speed=2)
            time.sleep(2)
            if np.random.random() < 0.3:
                if np.random.random() < 0.5:
                    self.move(np.array([[x2-20],[170],[z2]]),speed=2)
                else:
                    self.move(np.array([[x2+20],[170],[z2]]),speed=2)
                time.sleep(1)
            # self.move(np.array([[x2],[170],[z2]]),speed=2)
            self.move(np.array([[0],[200],[240]]),speed=2)
            
            
            followed = self.social_move('follow', start_pos = np.array([[0],[200],[240]]), speed=2)
            if followed:
                time.sleep(2)
            else:
                if np.random.random() < 0.3:
                    time.sleep(2)
                    if np.random.random() < 0.5:
                        self.move(np.array([[x2-20],[170],[z2]]),speed=2)
                    else:
                        self.move(np.array([[x2+20],[170],[z2]]),speed=2)
            self.move(start_pos, speed=2)
    def find_new_card(self, seen, reposition = False,
                        search_pos = np.array([[0],[200],[240]]),
                        reading_pos = np.array([[0],[200],[240]]),
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
            cur_loc = self.localize_notecard(new_id)
            print(f'CURLOC: {cur_loc}')
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
    
 
    def sweep(self, sweep_points=[(245.0,49.7), (140.8,51.8), (214.1,129.1), (115.2,96.1), (158.2,193.5), (76.2,129.2), (84.0,235.5), (28.4,147.3), (0.0,250.0), (-22.8,148.3), (-84.0,235.5), (-71.3,132.0), (-158.2,193.5), (-111.5,100.4), (-214.1,129.1), (-138.7,57.1), (-245.0,49.7)], sweep_height=220):
        
        # arcs:
        # sweep_points=[(147.0,29.8), (128.4,77.5), (94.9,116.1), (50.4,141.3), (0.0,150.0), (-50.4,141.3), (-94.9,116.1), (-128.4,77.5), (-147.0,29.8), (-245.0,49.7), (-214.1,129.1), (-158.2,193.5), (-84.0,235.5), (0.0,250.0), (84.0,235.5), (158.2,193.5), (214.1,129.1), (245.0,49.7)]

        # zigzag:
        # sweep_points=[(147.0,29.8), (245.0,49.7), (128.4,77.5), (214.1,129.1), (94.9,116.1), (158.2,193.5), (50.4,141.3), (84.0,235.5), (0.0,150.0), (0.0,250.0), (-50.4,141.3), (-84.0,235.5), (-94.9,116.1), (-158.2,193.5), (-128.4,77.5), (-214.1,129.1), (-147.0,29.8), (-245.0,49.7)]

        # pizza:
        # sweep_points=[(245.0,49.7), (140.8,51.8), (214.1,129.1), (115.2,96.1), (158.2,193.5), (76.2,129.2), (84.0,235.5), (28.4,147.3), (0.0,250.0), (-22.8,148.3), (-84.0,235.5), (-71.3,132.0), (-158.2,193.5), (-111.5,100.4), (-214.1,129.1), (-138.7,57.1), (-245.0,49.7)]
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
            time.sleep(0.1) # wait for arm to stop moving
            self.camera.grab()
            # time.sleep(0.1)
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
        print(f'sightings: {seen_count}')
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
            
            