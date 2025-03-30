/******************************************************************
 * Harold Quadruped Robot Controller - SIMPLIFIED & ROBUST
 *
 * Features:
 * - Direct angle-to-pulse mapping with low-pass filtering for smoothness.
 * - Relies on accurate calibration to prevent drift.
 * - Precise timing control (200Hz loop).
 * - Robust communication and safety features.
 ******************************************************************/

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h>    // Include for fabs, round, constrain
#include <string.h>  // Include for strcmp, strcpy, strtok_r
#include <stdlib.h>  // Include for atof
#include <ctype.h>   // Include for isdigit

//==========================//
//     GLOBAL CONSTANTS     //
//==========================//
#define MAX_MESSAGE_LENGTH 128
#define SERVO_FREQ 50            // Standard servo update rate
#define HANDSHAKE_MSG "ARDUINO_READY"

// Control timing parameters
#define CONTROL_INTERVAL_US 5000 // 5ms (200Hz) control loop
#define COMMAND_TIMEOUT_MS 1500  // Timeout before reverting to safe target angles
#define DEBUG_INTERVAL_MS 1000   // Interval for debug messages

// Servo physical limits (absolute pulse bounds)
#define SERVO_MIN_PULSE 150      // Absolute minimum pulse width the driver can send
#define SERVO_MAX_PULSE 600      // Absolute maximum pulse width the driver can send

// Smoothing Filter
// Lower value = more smoothing, slower response. Higher value = less smoothing, faster response.
// Good range: 0.02 (very smooth) to 0.2 (responsive)
#define PULSE_FILTER_ALPHA 0.05f // Low-pass filter coefficient for pulse smoothing

// Debug flag
#define DEBUG_OUTPUT 1           // Enable/disable Serial debug output

// Joint definitions
#define NUM_SERVOS 12
#define NUM_LEGS 4
#define JOINTS_PER_LEG 3

// Joint types (used for limits and safe pos)
#define SHOULDER 0
#define THIGH 1
#define KNEE 2

//==========================//
//    HARDWARE INTERFACE    //
//==========================//
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

//==========================//
//    JOINT CONFIGURATION   //
//==========================//
// Maps leg/joint type to servo index
const int legJointMap[NUM_LEGS][JOINTS_PER_LEG] = {
  {0, 4, 8},   // Front Left (shoulder, thigh, knee)
  {1, 5, 9},   // Front Right
  {2, 6, 10},  // Back Left
  {3, 7, 11}   // Back Right
};

// Servo calibration: Pulse width corresponding to min/max angles.
// !! CRITICAL FOR ACCURACY AND DRIFT PREVENTION !! MUST BE CALIBRATED !!
// Indices: 0-11 matching servo driver channels
int servoMinPulse[NUM_SERVOS] = {315, 305, 290, 360, 380, 185, 385, 215, 375, 185, 395, 185};
int servoMaxPulse[NUM_SERVOS] = {395, 225, 370, 280, 190, 375, 195, 405, 185, 375, 205, 375};
// Note: If a servo physically rotates opposite to convention, swap its min/max pulse values here.
// The mapping logic assumes minPulse corresponds to minAngle, maxPulse to maxAngle *after* any angle inversion.

// Joint Angle Limits (radians) - Used for clamping commands
// Defined per JOINT TYPE
const float jointMinAngle[JOINTS_PER_LEG] = {-0.35f, -0.79f, -0.79f}; // Shoulder, Thigh, Knee
const float jointMaxAngle[JOINTS_PER_LEG] = { 0.35f,  0.79f,  0.79f}; // Shoulder, Thigh, Knee
const float jointSafeAngle[JOINTS_PER_LEG] = { 0.0f,   0.3f,  -0.75f}; // Shoulder, Thigh, Knee

//==========================//
//       JOINT STATE        //
//==========================//
struct JointState {
  float targetAngleRad; // Target angle (radians) received from host
  float currentPulse;   // Current pulse width being sent (filtered value)
  uint8_t jointType;    // 0=Shoulder, 1=Thigh, 2=Knee (for limits/safe pos lookup)
  bool isLeftKnee;      // Flag for special mapping logic
};

JointState joints[NUM_SERVOS];

//==========================//
//   COMMUNICATION BUFFER   //
//==========================//
static char incomingBuffer[MAX_MESSAGE_LENGTH];
static int bufferIndex = 0;

//==========================//
//      TIMING CONTROL      //
//==========================//
static uint32_t lastControlMicros = 0;
static uint32_t lastCommandMillis = 0;

//==========================//
//    DIAGNOSTIC METRICS    //
//==========================//
// Simplified - just track command count and timeouts
uint32_t commandCount = 0;
uint32_t timeoutCount = 0;

//==========================//
//    FORWARD DECLARATIONS  //
//==========================//
void setupJoints();
void processSerialData();
void parseCommand(const char* command);
void moveToSafeTargetAngles();
float mapAngleToTargetPulse(int jointIdx, float angleRad);
void setServoPulse(int jointIdx, int pulse);
float constrainf(float x, float a, float b); // Explicit float constrain
int constrain(int x, int a, int b);        // Standard int constrain

//====================================================================
// SETUP
//====================================================================
void setup() {
  Serial.begin(115200);
  Serial.println("\nStarting Harold Controller (Simplified)...");

  // Send handshake early
  for (int i = 0; i < 3; i++) {
    Serial.println(HANDSHAKE_MSG);
    delay(20);
  }

  Serial.println("Initializing PWM driver...");
  pwm.begin();
  pwm.setOscillatorFrequency(27000000); // Verify for your PCA9685 board
  pwm.setPWMFreq(SERVO_FREQ);
  delay(100);
  Serial.println("PWM driver initialized.");

  // Configure initial joint states and move servos to safe position
  setupJoints();

  // Initialize timing
  lastCommandMillis = millis();
  lastControlMicros = micros();

  Serial.println("Controller ready.");
  Serial.println(HANDSHAKE_MSG); // Final handshake
}

//====================================================================
// MAIN LOOP
//====================================================================
void loop() {
  // 1. Process incoming serial data
  processSerialData(); // Updates joint target angles, resets command timer

  // 2. Run control loop at precise interval
  uint32_t currentMicros = micros();
  if (currentMicros - lastControlMicros >= CONTROL_INTERVAL_US) {
    lastControlMicros = currentMicros; // Update time for next interval

    // 3. Check for command timeout
    uint32_t currentMillis = millis(); // Get millis only once per loop
    if (currentMillis - lastCommandMillis > COMMAND_TIMEOUT_MS) {
      moveToSafeTargetAngles(); // Set target angles to safe values
      // Only log timeout periodically to avoid spam
      if (timeoutCount % 100 == 0) {
         Serial.println("Command timeout - Targets set to safe angles.");
      }
      timeoutCount++;
      // Don't reset lastCommandMillis here; let the next valid command do it.
    }

    // 4. Update all servos
    for (int i = 0; i < NUM_SERVOS; i++) {
      // 4a. Map the current target angle to a target pulse width using calibration
      // This function also handles angle clamping and knee inversion.
      float targetPulse = mapAngleToTargetPulse(i, joints[i].targetAngleRad);

      // 4b. Apply low-pass filter to the pulse width for smoothing
      // current = alpha * target + (1 - alpha) * current
      joints[i].currentPulse = (PULSE_FILTER_ALPHA * targetPulse) + ((1.0f - PULSE_FILTER_ALPHA) * joints[i].currentPulse);

      // 4c. Send the filtered pulse to the servo (setServoPulse handles absolute bounds)
      setServoPulse(i, (int)round(joints[i].currentPulse)); // Round to nearest integer pulse
    }

    // 5. Periodic Debug Output (throttled)
#if DEBUG_OUTPUT
    static uint32_t lastDebugTime = 0;
    if (currentMillis - lastDebugTime > DEBUG_INTERVAL_MS) {
        lastDebugTime = currentMillis;
        int j = 0; // Joint to debug (e.g., first shoulder)
        Serial.print("J"); Serial.print(j);
        Serial.print(" TgtAng: "); Serial.print(joints[j].targetAngleRad, 3);
        float targetPulseDebug = mapAngleToTargetPulse(j, joints[j].targetAngleRad);
        Serial.print(" TgtPulse: "); Serial.print(targetPulseDebug, 1);
        Serial.print(" CurPulse: "); Serial.println((int)round(joints[j].currentPulse));
    }
#endif
  } // End timed control loop

  // Less frequent tasks (e.g., idle handshake) - run outside timed loop
  static uint32_t lastIdleTaskTime = 0;
  uint32_t nowMillis = millis(); // Use cached millis if available
  if (nowMillis - lastIdleTaskTime > 2000) { // Check every 2 seconds
      lastIdleTaskTime = nowMillis;
      if (nowMillis - lastCommandMillis > 5000) { // If idle for 5 seconds
          Serial.println(HANDSHAKE_MSG); // Send keep-alive handshake
      }
  }

} // End main loop

//====================================================================
// JOINT SETUP
//====================================================================
void setupJoints() {
  Serial.println("Setting up initial joint states...");
  for (int leg = 0; leg < NUM_LEGS; leg++) {
    for (int joint = 0; joint < JOINTS_PER_LEG; joint++) {
      int idx = legJointMap[leg][joint];
      if (idx < 0 || idx >= NUM_SERVOS) {
          Serial.print("ERROR: Invalid servo index "); Serial.print(idx);
          Serial.print(" for leg "); Serial.print(leg); Serial.print(", joint "); Serial.println(joint);
          continue; // Skip invalid configuration
      }

      joints[idx].jointType = joint; // Store type (SHOULDER, THIGH, KNEE)
      joints[idx].isLeftKnee = (joint == KNEE && (leg == 0 || leg == 2)); // Legs 0 (FL) and 2 (BL) are left

      // Set initial target angle to the safe angle for this joint type
      joints[idx].targetAngleRad = jointSafeAngle[joint];

      // Calculate the initial pulse corresponding to the safe angle
      // Map directly, don't filter yet
      float initialPulse = mapAngleToTargetPulse(idx, joints[idx].targetAngleRad);
      joints[idx].currentPulse = initialPulse; // Initialize currentPulse

      // Move servo directly to initial pulse
      setServoPulse(idx, (int)round(initialPulse));
      delay(15); // Small delay for servo to start moving
    }
  }
  Serial.println("Joints set to safe angles. Allowing servos to settle...");
  delay(1000); // Wait for servos to reach initial position
  Serial.println("Joint setup complete.");
}

//====================================================================
// ANGLE TO PULSE MAPPING (Core Calibration Logic)
//====================================================================
// Maps a requested angle (radians) to the target servo pulse width.
// Handles joint limits, knee inversion, and calibration values.
float mapAngleToTargetPulse(int jointIdx, float angleRad) {
    // 1. Get joint type specific limits and safe angle
    uint8_t type = joints[jointIdx].jointType;
    float minAngle = jointMinAngle[type];
    float maxAngle = jointMaxAngle[type];

    // 2. Handle Left Knee Inversion: Invert angle *before* clamping
    float angleToProcess = angleRad;
    if (joints[jointIdx].isLeftKnee) {
        angleToProcess = -angleRad;
    }

    // 3. Clamp the angle (after potential inversion) to the joint's limits
    angleToProcess = constrainf(angleToProcess, minAngle, maxAngle);

    // 4. Get calibrated pulse range for this specific servo
    int minPulse = servoMinPulse[jointIdx];
    int maxPulse = servoMaxPulse[jointIdx];

    // 5. Perform linear mapping from angle range to pulse range
    float angleRange = maxAngle - minAngle;
    float pulseRange = (float)(maxPulse - minPulse); // Ensure float calculation

    float normalizedPos = 0.0f;
    // Avoid division by zero if angle range is negligible
    if (fabs(angleRange) > 1e-6) {
        normalizedPos = (angleToProcess - minAngle) / angleRange;
    }

    // Clamp normalized position just in case (shouldn't be needed if angle clamped)
    normalizedPos = constrainf(normalizedPos, 0.0f, 1.0f);

    float targetPulse = (float)minPulse + normalizedPos * pulseRange;

    return targetPulse; // Return the calculated target pulse (float)
}

//====================================================================
// SET SERVO PULSE
//====================================================================
// Sets the PWM pulse for a servo, constraining to absolute limits.
void setServoPulse(int jointIdx, int pulse) {
  // Constrain to the absolute physical limits of the PWM driver/servo
  pulse = constrain(pulse, SERVO_MIN_PULSE, SERVO_MAX_PULSE);
  pwm.setPWM(jointIdx, 0, pulse);
}

//====================================================================
// MOVE TO SAFE ANGLES
//====================================================================
// Sets the *target* angle for all joints to their safe values.
// The filtering in the main loop will handle the smooth transition.
void moveToSafeTargetAngles() {
  bool changed = false;
  for (int i = 0; i < NUM_SERVOS; i++) {
    float safeAngle = jointSafeAngle[joints[i].jointType];
    // Only update if not already targeting safe angle (within tolerance)
    if (fabs(joints[i].targetAngleRad - safeAngle) > 1e-5) {
        joints[i].targetAngleRad = safeAngle;
        changed = true;
    }
  }
  // if (changed && DEBUG_OUTPUT) { // Reduce spam
  //   Serial.println("Targets updated to safe angles.");
  // }
}

//====================================================================
// SERIAL COMMUNICATION
//====================================================================
void processSerialData() {
    // Quick check for handshake request 'R' before buffering
    if (Serial.peek() == 'R') {
         String maybeReady = "";
         unsigned long readStart = millis();
         // Read quickly to check for "READY?#" pattern
         while (Serial.available() > 0 && (millis() - readStart < 50) && maybeReady.length() < 10) {
             char c = (char)Serial.read();
             maybeReady += c;
             if (c == '#') break;
         }

         if (maybeReady.indexOf("READY?") != -1) {
             // Respond to handshake
             for(int i=0; i<3; ++i) { Serial.println(HANDSHAKE_MSG); delay(5); }
             lastCommandMillis = millis(); // Treat handshake as activity
             bufferIndex = 0; incomingBuffer[0] = '\0'; // Clear buffer
             while(Serial.available() > 0) Serial.read(); // Consume rest of line
             return; // Handled
         } else {
             // Not handshake, put read data into main buffer
              if (bufferIndex + maybeReady.length() < MAX_MESSAGE_LENGTH -1) {
                  memcpy(incomingBuffer + bufferIndex, maybeReady.c_str(), maybeReady.length());
                  bufferIndex += maybeReady.length();
                  incomingBuffer[bufferIndex] = '\0';
              } else {
                   Serial.println("ERR: Buffer overflow on non-handshake read");
                   bufferIndex = 0; incomingBuffer[0] = '\0';
              }
         }
    }

  // Read remaining chars into buffer until '#' or full
  while (Serial.available() > 0 && bufferIndex < (MAX_MESSAGE_LENGTH - 1)) {
    char c = (char)Serial.read();
    if (c == '\r' || c == '\n') continue; // Ignore newline/CR

    if (c == '#') { // End of command found
      incomingBuffer[bufferIndex] = '\0'; // Terminate string
      if (bufferIndex > 0) {
        parseCommand(incomingBuffer); // Process the complete command
      }
      bufferIndex = 0; // Reset for next command
    } else {
      incomingBuffer[bufferIndex++] = c; // Add char to buffer
    }
  }

  // Handle buffer overflow if '#' wasn't found
  if (bufferIndex >= (MAX_MESSAGE_LENGTH - 1)) {
    Serial.println("ERR: Command buffer overflow, discarding.");
    bufferIndex = 0;
    incomingBuffer[0] = '\0';
  }
}

// Parses a complete command string from the buffer
void parseCommand(const char* command) {
  // Handle simple string commands
  if (strcmp(command, "STATUS") == 0) {
#if DEBUG_OUTPUT
    Serial.print("STATUS: OK Cmds:"); Serial.print(commandCount);
    Serial.print(" Timeouts:"); Serial.println(timeoutCount);
#endif
    lastCommandMillis = millis(); // Acknowledge command
    return;
  }
  if (strcmp(command, "STOP") == 0) {
#if DEBUG_OUTPUT
    Serial.println("STOP received - moving to safe angles & stopping.");
#endif
    moveToSafeTargetAngles();
    // Force servos immediately to calculated safe pulse (bypass filter)
    for (int i = 0; i < NUM_SERVOS; i++) {
      float safePulse = mapAngleToTargetPulse(i, joints[i].targetAngleRad);
      joints[i].currentPulse = safePulse; // Update state immediately
      setServoPulse(i, (int)round(safePulse));
    }
    lastCommandMillis = millis(); // Acknowledge command
    return;
  }
  if (strcmp(command, "READY?") == 0) { // Should be handled by pre-check, but safety
     for(int i=0; i<3; ++i) Serial.println(HANDSHAKE_MSG);
     lastCommandMillis = millis();
     return;
  }

  // --- Attempt to parse 12 joint angles ---
  int jointIdx = 0;
  char commandCopy[MAX_MESSAGE_LENGTH]; // Create mutable copy for strtok_r
  strncpy(commandCopy, command, MAX_MESSAGE_LENGTH - 1);
  commandCopy[MAX_MESSAGE_LENGTH - 1] = '\0';

  char* valueStr;
  char* rest = commandCopy;
  bool parseError = false;
  float parsedAngles[NUM_SERVOS]; // Temporary storage

  while (jointIdx < NUM_SERVOS && (valueStr = strtok_r(rest, ",", &rest))) {
    if (*valueStr == '\0') { parseError = true; break; } // Handle empty token ",,"

    // Use strtof for potentially better error checking than atof if needed
    // char* endPtr;
    // float value = strtof(valueStr, &endPtr);
    // if (endPtr == valueStr) { parseError = true; break; } // No conversion happened
    float value = atof(valueStr); // Simpler atof for now

    parsedAngles[jointIdx] = value;
    jointIdx++;
  }

  // Validate parsing result
  if (jointIdx == NUM_SERVOS && !parseError) {
    // Successfully parsed 12 values - Update target angles
    for (int i = 0; i < NUM_SERVOS; ++i) {
      // Clamping happens inside mapAngleToTargetPulse, just store the raw target
      joints[i].targetAngleRad = parsedAngles[i];
    }
    lastCommandMillis = millis(); // Reset command timer
    commandCount++;
#if DEBUG_OUTPUT && 0 // Verbose logging - disable by default
    if (commandCount % 20 == 1) {
        Serial.print("CMD OK("); Serial.print(commandCount); Serial.print("): Tgt0=");
        Serial.println(joints[0].targetAngleRad, 3);
    }
#endif
  } else {
#if DEBUG_OUTPUT
    Serial.print("ERR: Invalid angle command. Parsed ");
    Serial.print(jointIdx); Serial.print("/"); Serial.print(NUM_SERVOS);
    if(parseError) Serial.print(" (parse error)");
    Serial.print(". CMD:["); Serial.print(command); Serial.println("]");
#endif
  }
}

//====================================================================
// UTILITY FUNCTIONS
//====================================================================

// Float constrain (standard library might not have it)
float constrainf(float x, float a, float b) {
    if(x < a) return a;
    if(x > b) return b;
    return x;
}

// Integer constrain (standard library usually has this)
int constrain(int x, int a, int b) {
    if(x < a) return a;
    if(x > b) return b;
    return x;
}
