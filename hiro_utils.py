import numpy as np

'''
# mkii
MATH_PI = 3.141592653589793238463
MATH_TRANS = 57.2958    
MATH_L1 = 88.9	#90.00	
MATH_L2 = 10 #21.17	
MATH_LOWER_ARM = 142.07	#148.25	
MATH_UPPER_ARM = 158.8	#160.2 	
MATH_FRONT_HEADER = 29.4 #25.00// the distance between wrist to the front point we use
MATH_UPPER_LOWER = MATH_UPPER_ARM/MATH_LOWER_ARM
MAX_Z = 260		# MAX height
MIN_Z =	-120
'''

# metal
MATH_PI = 3.141592653589793238463
MATH_TRANS = 57.2958    
MATH_L1 = 107.45	
MATH_L2 = 21.17	
MATH_LOWER_ARM = 148.25	
MATH_UPPER_ARM = 160.2 	
MATH_FRONT_HEADER = 25.00 #// the distance between wrist to the front point we use
MATH_UPPER_LOWER = MATH_UPPER_ARM/MATH_LOWER_ARM
MAX_Z = 260		#// max height
MIN_Z = -120




def angles2xyz(left_servo, right_servo, base_servo):
    stretch = MATH_LOWER_ARM * np.cos(left_servo / MATH_TRANS) + MATH_UPPER_ARM * np.cos(right_servo / MATH_TRANS) + MATH_L2 + MATH_FRONT_HEADER

	# projection length on Z axis 在Z轴的投影长度
    height = MATH_LOWER_ARM * np.sin(left_servo / MATH_TRANS) - MATH_UPPER_ARM * np.sin(right_servo / MATH_TRANS) + MATH_L1
    x = stretch * np.cos(base_servo / MATH_TRANS)
    y = stretch * np.sin(base_servo / MATH_TRANS)
    z = height
    return x,y,z

def get_xyz(arm):
    left_servo = arm.get_analog(0)
    right_servo = arm.get_analog(1)
    base_servo = arm.get_analog(2)
    print(left_servo, right_servo, base_servo)
    return angles2xyz(left_servo, right_servo, base_servo)