import math
import numpy as np

def localize_fiducial(tag, pixel2mm, pose):
    '''
    Localizes center of fiducial assocalted with fid_num in workspace frame
    returns tuple of form (x, y, angle)
    '''
    # conversion factor current height (assume height hasn't changed since last capture)
    
    # import pdb; pdb.set_trace()
    #detect fiducials
    print("checking fiducials")
    
        
    p_cam = tag.center # fiducial position in camera FoV
    (ptA, ptB, ptC, ptD) = tag.corners # locations of four corners in camera FoV
    # beta_cam = np.arctan2(ptB[1]-ptA[1], ptB[0]-ptA[0])*180/math.pi #fiducial angle in camera frame
    
    beta_cam = np.arctan2(-ptA[1]+ptB[1], ptA[0]-ptB[0])*180/math.pi
    # import pdb; pdb.set_trace()
            
    # express the center of the fiducial with respect 
    # to the center of the camera frame (with the y-axis flipped to point up)
    # width= 4032
    # height= 3040
    width=1900
    height=1200
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
    phi = -np.arctan2(pose[0,0], pose[1,0]+ y_err) #robot angle (need to verify sign)
    T = np.array([[np.cos(phi), -np.sin(phi), pose[0,0]],
                    [np.sin(phi),  np.cos(phi), pose[1,0]+y_err],
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

def localize_notecard(tag, pixel2mm, pose, offset_view_loc = None):
    '''
    localizes notecard given the fiducial ID and the angle of the notecard measured
    CCW from the workspace x-axis
    Requires that the desired fiducial is in the current view
    returns tuple of form (x, y, angle)
    '''
    # measure l and k on caibration card
    l = 22.5 # horizontal distance from card cener to fiducial center [mm]
    k = 12.8 # vertical distance from card cener to fiducial center [mm]
    P_f = localize_fiducial(tag, pixel2mm, pose) #locaiton of fiducial center
    
    beta = P_f[2]*math.pi/180 #convert to radians
    x_nc = P_f[0]-l*np.cos(beta)+k*np.sin(beta)
    y_nc = P_f[1]-l*np.sin(beta)-k*np.cos(beta)
    ### DANGER: MATT IS CHANGING THIS FOR VIEW CALIBRATION!!!!
    # return (x_nc, y_nc, P_f[2]) # angle remains the same
    
    loc = (x_nc, y_nc, P_f[2])
    if offset_view_loc is not None:
        loc = offset_view_loc(loc)
    return loc