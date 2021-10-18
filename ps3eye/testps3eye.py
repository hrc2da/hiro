import cv2
import numpy as np
import time
import os
# time.sleep(10)
camera = cv2.VideoCapture(1)
frames = []

audio_files = ['audio/Welcome_my_name_is_Hiro.mp3',
                'audio/lets_brainstorm_together.mp3',
                'audio/if_you_write_ideas_on_notes_I_will_try_to_organize_them.mp3',
                'audio/start_writing_new_cards.mp3',
                'audio/thirty_seconds_left.mp3',
                'audio/ten_seconds_left.mp3',
                'audio/start_rearranging_cards.mp3']

for fname in audio_files[:4]:
    os.system(f'mpg321 {fname}')
# for i in range(1000):
#     s,frame = camera.read()
#     frames.append(frame)
# print("SAVINGG VIDEO")
# np.save('ps3eyetest.np',frames)

# frames = np.load('ps3eyetest.np.npy')
# for frame in frames:
#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

cv2.destroyAllWindows()