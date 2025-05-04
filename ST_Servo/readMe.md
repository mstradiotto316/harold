# STEP 1: INSTALL & SETUP
Donwload Arduino and install the relevant boars, libraries, etc. Go to the following page and follow the instructions under "Compile Arduino IDE

https://www.waveshare.com/wiki/ST3215_Servo

# STEP 2: VERIFY CONNECTION
--> Connect a SINGLE servo to the ESP32 Servo Driver Board
--> Open SCServo/examples/STSCL/Ping/Ping.ino with arduino
--> Make sure int TEST_ID = 1; or change it to the ID of the servo you want to ping (NOTE: factory default is ID=1)
--> Flash the firmware to the ESP32, if it does not write, you need to press and hold the BOOT button on the ESP32 while Arduino compiles and attempts to write the new firmware. You can release the BOOT button once you see the write process beggining.

# STEP 3: UPDATE SERVO ID
--> NOTE: We are going to update the IDs so they are as follows:
ID1 = Front Left Leg Shoulder Joint
ID2 = Front Left Leg Thigh Joint
ID3 = Front Left Leg Calf Joint

ID4 = Front Right Leg Shoulder Joint
ID5 = Front Right Leg Thigh Joint
ID6 = Front Right Leg Calf Joint

ID7 = Back Left Leg Shoulder Joint
ID8 = Back Left Leg Thigh Joint
ID9 = Back Left Leg Calf Joint

ID10 = Back Right Leg Shoulder Joint
ID11 = Back Right Leg Thigh Joint
ID12 = Back Right Leg Calf Joint

--> Open SCServo/examples/STSCL/ProgramEprom/ProgramEprom.ino
--> Update int ID_ChangeFrom = 1; and int ID_Changeto   = 2; so they match the current ID and the intended new ID

# STEP 4: SET MIDDLE POSITION
--> Open SCServo/examples/STSCL/CalibrationOfs/CalibrationOfs.ino
--> Change the SERVO_ID to the ID of the servo you just set
--> Run this once, if you see "Offset set on ID 1" or the ID you selected then you have done this correctly

