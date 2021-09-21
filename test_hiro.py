from hiro_interface import *
from time import sleep
from hiro_lightweight_interface import HIROLightweight
hiro = HIROLightweight()
# hiro = HIRO(mute=False)
# hiro.arm.set_pump(False)

#time.sleep(50)
#hiro.arm.set_pump(False)
#hiro.move(np.array([[0], [300], [100]]))
#hiro.arm.set_position(0,300,100)
'''
hiro.move(np.array([[-200],[190],[150]]))
time.sleep(0.5)
hiro.move(np.array([[0],[210],[150]]))
time.sleep(0.5)
hiro.move(np.array([[200],[190],[150]]))
time.sleep(0.5)
hiro.move(np.array([[200],[260],[150]]))
time.sleep(0.5)
hiro.move(np.array([[0],[280],[150]]))
time.sleep(0.5)
hiro.move(np.array([[-200],[260],[150]]))
time.sleep(0.5)
hiro.move(np.array([[-200],[190],[150]]))
hiro.move(np.array([[-200],[210],[150]]))
hiro.move(np.array([[-200],[190],[150]]))
time.sleep(0.5)
hiro.move(np.array([[0],[190],[150]]))
hiro.move(np.array([[0],[210],[150]]))
hiro.move(np.array([[0],[190],[150]]))
time.sleep(0.5)
hiro.move(np.array([[200],[190],[150]]))
hiro.move(np.array([[200],[210],[150]]))
hiro.move(np.array([[200],[190],[150]]))
time.sleep(0.5)
hiro.move(np.array([[-200],[250],[150]]))
hiro.move(np.array([[200],[250],[150]]))
'''




# test basic pick and place
# start = (150, 150)
# end = (-100, 200)
# hiro.pick_place(start, end)

# hiro.move(np.array([[0], [200], [200]]))

'''
# test beeps
hiro.beep(0)
time.sleep(2)
hiro.beep(1)
time.sleep(2)
hiro.beep(2)
time.sleep(0.5)
hiro.beep(3)
time.sleep(0.5)
hiro.beep(4)
time.sleep(2)
hiro.beep(5)
'''
#test loclizaiton
# hiro.move(np.array([[0], [200], [200]]))
# print(hiro.position)
# hiro.capture('/home/pi/hiro/views/test.jpg')
# nc_loc = hiro.localize_notecard(23)
# print(f"found card at: {nc_loc}")
# hiro.pick_place(nc_loc, (-50, 100))

''''
'''
# hiro.move(np.array([[0],[180],[200]]))
# # hiro.arm.set_position(0,150,200)
# # test search for new card

#####################
# seen = [] #ids of notecards already seen
# for i in range(400):
#     input("Press enter to capture")
    
#     hiro.move(np.array([[0],[200],[240]]))
#     test = hiro.capture("tempa.jpg")
#     # hiro.localize_fiducial(new)
#     new = hiro.find_new_card([], search_pos=np.array([[0],[200],[240]]), reading_pos=np.array([[0],[200],[240]]), reposition=False)
#     print(new)
#     seen.append(new)
##################

while True:
    x = int(input("Enter x"))
    y = int(input("Enter y"))
    hiro.move(np.array([[0],[200],[240]]))
    hiro.move(np.array([[x],[y],[70]]))



#     print(hiro.get_fiducial_map())
#     hiro.close_camera()
#     hiro.setup_camera()
    # time.sleep(2)
# while True:
#     hiro.move(np.array([[0],[200],[200]]),speed=2)
#     time.sleep(1)
#     hiro.social_move("follow", speed=2)
# while True:
#     hiro.social_move("breathe")
    
'''
# test fake clustering

# centerpoints for clusters
c0_centers = [(0, 190), (0, 130), (-80, 130), (80, 130), (-180, 190), (80, 190)]
c1_centers = [(-240, 0), (-240, 60), (-160, 0), (-160, 60), (-320, 0), (-320, 60)] 
c2_centers = [(240, 0), (240, 60), (160, 0), (160, 60), (320, 0), (320, 60)]
c3_centers = [(-260, 230), (-290, 170), (-210, 170), (-180, 230), (-190, 290), (-110, 290)]
c4_centers = [(260, 230), (290, 170), (210, 170), (180, 230), (190, 290), (110, 290)]
cluster_centers = [c0_centers, c1_centers, c2_centers, c3_centers, c4_centers]


#test projector
hiro.move(np.array([[0], [250], [250]]))
hiro.project('Hello World')
time.sleep(1)
hiro.project('3')
time.sleep(0.5)
hiro.project('2')
time.sleep(0.5)
hiro.project('1')
time.sleep(0.5)
hiro.project('Goodbye')
time.sleep(3)
hiro.disconnect()
'''
# test slowing movements
# hiro.move(np.array([[-200], [50], [200]]))
# hiro.move(np.array([[200], [50], [200]]))
# hiro.move(np.array([[-200], [30], [200]]))
# hiro.move(np.array([[200], [30], [200]]))
# hiro.move(np.array([[-200], [10], [200]]))
# hiro.move(np.array([[200], [10], [200]]))