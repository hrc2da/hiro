from hiro_interface import *

hiro = HIRO(mute=False)
'''
# test basic pick and place
start = (150, 150)
end = (-100, 200)
hiro.pick_place(start, end)

hiro.move(np.array([[0], [200], [200]]))


# test beeps
hiro.beep(0)
time.sleep(2)
hiro.beep(1)
time.sleep(2)
hiro.beep(2)

# test loclizaiton
hiro.move(np.array([[0], [200], [200]]))
print(hiro.position)
hiro.capture('/home/pi/hiro/views/test.jpg')
nc_loc = hiro.localize_notecard(0)
print(nc_loc)
hiro.pick_place(nc_loc, (-50, 100))


# test search for new card
seen = [] #ids of notecards already seen
for i in range(4):
    new = hiro.find_new_card(seen)
    print(new)
    seen.append(new)
    time.sleep(2)


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
hiro.move(np.array([[-200], [50], [200]]))
hiro.move(np.array([[200], [50], [200]]))
hiro.move(np.array([[-200], [30], [200]]))
hiro.move(np.array([[200], [30], [200]]))
hiro.move(np.array([[-200], [10], [200]]))
hiro.move(np.array([[200], [10], [200]]))