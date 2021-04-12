'''
Trying out basic functions of BLE with HIRO
'''
import sys
import time
sys.path.append('./bluepy/bluepy')
from bluepy.btle import Peripheral, DefaultDelegate
BLE_MAC = 'FC:45:C3:24:76:EA'

hiro_ble = Peripheral(BLE_MAC)
hiro_channel = hiro_ble.getCharacteristics()[-1] # handle is 17


GET_SIMULATION          = "M222 X{} Y{} Z{} P0"
GET_FIRMWARE_VERSION    = "P203"
GET_HARDWARE_VERSION    = "P202"
GET_COOR                = "P220"
GET_SERVO_STATUS        = "M203"
GET_SERVO_ANGLE         = "P200"
GET_IS_MOVE             = "M200"
GET_TIP_SENSOR          = "P233"
GET_POLAR               = "P221"
GET_GRIPPER             = "P232"
GET_EEPROM              = "M211 N0 A{} T{}"
SET_EEPROM              = "M212 N0 A{} T{} V{}"
GET_ANALOG              = "P241 N{}"
GET_DIGITAL             = "P240 N{}"

serial_id = 1
resp = 'empty'
def serial_encode(command):
    global serial_id
    serial_id += 1
    ret = f'#{serial_id} {command}\n'.encode()
    print(ret)
    return ret

# delegate task to handle notifications from the BLE module
class HiroDelegate(DefaultDelegate):
    def __init__(self, params=None):
        DefaultDelegate.__init__(self)
    def handleNotification(self, cHandle, data):
        global resp
        resp = f'Message from {cHandle}: {data}'
        print(resp)

# instantiate and attach the delegate
hiro_ble = hiro_ble.withDelegate(HiroDelegate(None))
# start by getting the firmware version
status = hiro_channel.write(serial_encode(GET_FIRMWARE_VERSION))
hiro_ble.waitForNotifications(15)
status = hiro_channel.write(serial_encode(GET_COOR))
hiro_ble.waitForNotifications(15)
hiro_ble.disconnect()

