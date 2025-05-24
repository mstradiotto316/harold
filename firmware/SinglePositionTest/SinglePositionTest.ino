/* Robot control via Serial Monitor with radian inputs for 12 servos.
 * All servos start at 0 degrees. Includes a ping test during setup.
 * Expects input like: [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12]
 * where rX is the target angle in radians for servo X (ID 1 to 12).
 */
#include <SCServo.h>
#include <cmath>   // For M_PI (pi constant) and atof (string to float)
#include <cstdlib> // For atof, though often included with cmath or Arduino core
#include <cstring> // For strtok

SMS_STS st; // Servo object

// Serial port pins for servo communication
// !!! ESP32 USER: VERIFY THESE PINS ARE CORRECT FOR Serial1 ON YOUR SPECIFIC ESP32 BOARD !!!
// Common defaults for Serial1 might be GPIO9 (RX1) and GPIO10 (TX1) if available.
// Using incorrect pins here can cause the sketch to crash or hang.
#define S_RXD 18 // RX pin for Serial1
#define S_TXD 19 // TX pin for Serial1
const uint32_t BUS_BAUD = 1000000; // Baud rate for the servo bus - REVERTED based on working example

// Safety limits for robot movements (in degrees for easier clamping logic)
const float MAX_SHOULDER_ANGLE_DEG = 30.0f; // Max angle for shoulder joints from center
const float MAX_LEG_ANGLE_DEG = 90.0f;      // Max angle for thigh/calf joints from center

// Servo direction multipliers (ID 0 is unused, IDs 1-12 are used)
const int DIR[13] = { 0,
  /*Front‑Left*/   +1,  /*ID 1: Shoulder*/   +1,  /*ID 2: Thigh*/    +1,  /*ID 3: Calf*/
  /*Front‑Right*/  -1,  /*ID 4: Shoulder*/   -1,  /*ID 5: Thigh*/    -1,  /*ID 6: Calf*/
  /*Rear‑Left*/    +1,  /*ID 7: Shoulder*/   +1,  /*ID 8: Thigh*/    +1,  /*ID 9: Calf*/
  /*Rear‑Right*/   -1,  /*ID 10: Shoulder*/  -1,  /*ID 11: Thigh*/   -1   /*ID 12: Calf*/
};

/* shoulder, thigh, calf IDs per leg. This array helps map an index (0-11)
 * from the input array to a specific servo ID and its type (shoulder/thigh/calf).
 */
const uint8_t LEG_SERVO_IDS[4][3] = {
    {1,2,3},   // Front-Left leg: Shoulder ID 1, Thigh ID 2, Calf ID 3
    {4,5,6},   // Front-Right leg: Shoulder ID 4, Thigh ID 5, Calf ID 6
    {7,8,9},   // Rear-Left leg: Shoulder ID 7, Thigh ID 8, Calf ID 9
    {10,11,12} // Rear-Right leg: Shoulder ID 10, Thigh ID 11, Calf ID 12
};

// Servo movement parameters
const int SERVO_SPEED = 1000; // Default speed for servo movements
const int SERVO_ACC = 50;     // Default acceleration for servo movements

// Function to convert degrees to servo position units
inline int degToPos(uint8_t id, float deg, int mid = 2047) {
    constexpr float UNITS_PER_DEGREE = 4096.0f / 360.0f;
    return mid + static_cast<int>(DIR[id] * deg * UNITS_PER_DEGREE + 0.5f);
}

// Function to clamp a value between a minimum and maximum
float clamp(float value, float min_val, float max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

// Function to convert radians to degrees
float radiansToDegrees(float radians) {
    return radians * (180.0f / M_PI);
}

// Function to initialize all 12 servos to 0 degrees
void initializeServosToZero() {
    Serial.println("Attempting to initialize all servos to 0 degrees..."); 
    uint8_t servo_ids_array[12];
    int16_t target_positions_array[12];
    uint16_t speeds_array[12];
    uint8_t accs_array[12];

    for (int i = 0; i < 12; i++) {
        servo_ids_array[i] = LEG_SERVO_IDS[i / 3][i % 3]; 
        target_positions_array[i] = degToPos(servo_ids_array[i], 0.0f);
        speeds_array[i] = SERVO_SPEED;
        accs_array[i] = SERVO_ACC;
    }
    st.SyncWritePosEx(servo_ids_array, 12, target_positions_array, speeds_array, accs_array);
    Serial.println("Servo initialization command (SyncWritePosEx) sent."); 
    delay(2000); 
}

// Function to move all servos based on an array of target radian values
void moveAllServosRadians(float target_radians[12]) {
    Serial.println("Processing new radian commands...");
    uint8_t servo_ids_cmd[12];
    int16_t servo_target_positions_cmd[12];
    uint16_t servo_speeds_cmd[12];
    uint8_t servo_accs_cmd[12];
    float commanded_degrees_log[12]; 

    for (int i = 0; i < 12; i++) {
        servo_ids_cmd[i] = LEG_SERVO_IDS[i / 3][i % 3]; 
        float target_degrees = radiansToDegrees(target_radians[i]);
        float clamped_degrees;
        int joint_type_in_leg = i % 3; 

        if (joint_type_in_leg == 0) { 
            clamped_degrees = clamp(target_degrees, -MAX_SHOULDER_ANGLE_DEG, MAX_SHOULDER_ANGLE_DEG);
        } else { 
            clamped_degrees = clamp(target_degrees, -MAX_LEG_ANGLE_DEG, MAX_LEG_ANGLE_DEG);
        }

        if (abs(target_degrees - clamped_degrees) > 0.01f) { 
            Serial.print("Servo ID "); Serial.print(servo_ids_cmd[i]);
            Serial.print(": Input "); Serial.print(target_radians[i], 3);
            Serial.print(" rad ("); Serial.print(target_degrees, 2);
            Serial.print(" deg) was CLAMPED to "); Serial.print(clamped_degrees, 2);
            Serial.println(" deg.");
        }
        commanded_degrees_log[i] = clamped_degrees; 
        servo_target_positions_cmd[i] = degToPos(servo_ids_cmd[i], clamped_degrees);
        servo_speeds_cmd[i] = SERVO_SPEED;
        servo_accs_cmd[i] = SERVO_ACC;
    }

    Serial.print("Commanding degrees (after clamping): [");
    for(int i=0; i<12; i++){
      Serial.print(commanded_degrees_log[i], 2);
      if(i<11) Serial.print(", ");
    }
    Serial.println("]");
    st.SyncWritePosEx(servo_ids_cmd, 12, servo_target_positions_cmd, servo_speeds_cmd, servo_accs_cmd);
    Serial.println("SyncWritePosEx command sent for movement."); 
}

void setup() {
    // STEP 1: Try to get basic Serial (USB) communication working.
    Serial.begin(115200);
    delay(1000); // Give Serial a moment to initialize
    Serial.println("Sketch setup started. Basic Serial (USB) communication is working.");
    Serial.println("--------------------------------------------------------------------");
    Serial.println("IMPORTANT ESP32 Serial1 PIN CHECK:");
    Serial.print("Attempting to use S_RXD (Serial1 RX) on pin: "); Serial.println(S_RXD);
    Serial.print("Attempting to use S_TXD (Serial1 TX) on pin: "); Serial.println(S_TXD);
    Serial.println("VERIFY these pins are correct for Serial1 on YOUR ESP32 board model.");
    Serial.println("If incorrect, Serial1.begin() might crash or hang the ESP32.");
    Serial.println("Common ESP32 Serial1 pins (if available) are GPIO9 (RX1) & GPIO10 (TX1).");
    Serial.println("Consult your ESP32 board's pinout diagram or documentation.");
    Serial.println("--------------------------------------------------------------------");
    delay(2000); // Extra delay before trying to init Serial1

    // STEP 2: Attempt to initialize Serial1 for servo communication.
    Serial.println("Attempting to initialize Serial1 for servo bus...");
    Serial.print("Using BUS_BAUD: "); Serial.println(BUS_BAUD); // Log the baud rate being used
    Serial1.begin(BUS_BAUD, SERIAL_8N1, S_RXD, S_TXD); 
    st.pSerial = &Serial1; 
    Serial.println("Serial1 initialization attempted. st.pSerial assigned.");
    Serial.println("If sketch hangs here or no further messages, Serial1 init or pin assignment is likely the issue.");
    delay(500);

    // STEP 3: Attempt to ping servos.
    Serial.println("Attempting to ping servos (IDs 1, 4, 7, 10)...");
    int ping_ids[] = {1, 4, 7, 10}; // Shoulder servos
    bool any_servo_responded = false;
    for (int i = 0; i < 4; i++) {
        int servo_id_to_ping = ping_ids[i];
        Serial.print("Pinging servo ID: "); Serial.println(servo_id_to_ping);
        int response = st.Ping(servo_id_to_ping);
        if (response == servo_id_to_ping) { // Successful ping returns the ID
            Serial.print("Servo ID "); Serial.print(servo_id_to_ping); Serial.println(" responded to ping!");
            any_servo_responded = true;
        } else { // Ping failed or returned an unexpected value (like -1 for error)
            Serial.print("Servo ID "); Serial.print(servo_id_to_ping); Serial.print(" DID NOT respond as expected. Ping response code: "); Serial.println(response);
        }
        delay(100); // Increased delay between pings slightly
    }

    if (!any_servo_responded) {
        Serial.println("--------------------------------------------------------------------");
        Serial.println("CRITICAL: NO SERVOS RESPONDED TO PING AS EXPECTED.");
        Serial.println("Potential Issues:");
        Serial.println("1. Incorrect Serial1 pins (S_RXD, S_TXD) for your ESP32 board. VERIFY PINOUT.");
        Serial.println("2. Wiring error between ESP32 and servo bus (RX to TX, TX to RX, GND).");
        Serial.println("3. Servo power issue (ensure servos have adequate and correct voltage).");
        Serial.println("4. Servo baud rate mismatch (BUS_BAUD in code vs. actual servo setting - currently set to 1000000).");
        Serial.println("5. Incorrect servo IDs (are servos 1, 4, 7, 10 actually on the bus and configured correctly?).");
        Serial.println("6. Faulty servo or ESP32 hardware.");
        Serial.println("Halting further operations. Please diagnose the communication issue.");
        Serial.println("--------------------------------------------------------------------");
        while(true){ 
            Serial.print("."); // Print something in the halt loop to show it's alive
            delay(1000); 
        } 
    } else {
         Serial.println("At least one servo responded. Proceeding with servo initialization to zero.");
    }

    delay(1000); 

    initializeServosToZero(); 

    Serial.println("\nRobot ready for commands.");
    Serial.println("Send 12 servo positions in radians, comma-separated, enclosed in brackets.");
    Serial.println("Example: [-0.349, 0.349, 0.349, -0.349, 0.785, 0.785, 0.785, 0.785, 0.785, 0.785, 0.785, 0.785]");
}

void loop() {
    if (Serial.available() > 0) {
        String inputString = Serial.readStringUntil('\n'); 
        inputString.trim(); 

        Serial.print("Received command: ");
        Serial.println(inputString);

        if (inputString.startsWith("[") && inputString.endsWith("]")) {
            String valuesString = inputString.substring(1, inputString.length() - 1);
            float parsed_radians[12]; 
            int value_count = 0;      
            char str_buffer[valuesString.length() + 1];
            valuesString.toCharArray(str_buffer, sizeof(str_buffer));
            char *token = strtok(str_buffer, ",");
            while (token != NULL && value_count < 12) {
                parsed_radians[value_count++] = atof(token); 
                token = strtok(NULL, ","); 
            }

            if (value_count == 12 && token == NULL) { 
                moveAllServosRadians(parsed_radians);
            } else {
                Serial.print("Error: Invalid number of values. Parsed ");
                Serial.print(value_count);
                Serial.println(" values. Expected 12 comma-separated float values.");
                if (token != NULL) {
                     Serial.println("Possible cause: Too many values or malformed string after 12th value.");
                }
            }
        } else {
            Serial.println("Error: Invalid command format. Input must start with '[' and end with ']'.");
        }
        Serial.println("\nRobot ready for next command."); 
    }
}
