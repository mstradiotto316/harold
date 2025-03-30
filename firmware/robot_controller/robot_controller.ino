/******************************************************************
 * Harold Quadruped Robot Controller - REVISED for Smoothness & Stability
 *
 * Features:
 * - Smoother PID control tuned for 20Hz command updates on 200Hz loop
 * - Simplified and stronger low-pass filtering
 * - Precise timing control
 * - Robust error handling and safety features
 * - Focused on eliminating drift and jerkiness
 ******************************************************************/

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h> // Include for fabs, constrain
#include <string.h> // Include for strcmp, strcpy
#include <stdlib.h> // Include for atof
#include <ctype.h> // Include for isspace, isdigit

//==========================//
//     GLOBAL CONSTANTS     //
//==========================//
#define MAX_MESSAGE_LENGTH 128
#define SERVO_FREQ 50
#define HANDSHAKE_MSG "ARDUINO_READY"

// Control timing parameters (in microseconds for precision)
#define CONTROL_INTERVAL_US 5000     // 5ms (200Hz) control loop
#define COMMAND_TIMEOUT_MS 1500      // Timeout for reverting to safe position (slightly increased)
#define MOVEMENT_TIMEOUT_MS 5000     // Timeout for detecting stuck movements
#define DEBUG_INTERVAL_MS 1000       // Interval for debug messages

// Servo physical limits (in pulse width)
#define SERVO_MIN_PULSE 150          // Absolute minimum pulse width
#define SERVO_MAX_PULSE 600          // Absolute maximum pulse width

// --- PID Control Configuration ---
#define ENABLE_PID_INTEGRAL 0        // <<< SET TO 1 TO ENABLE INTEGRAL TERM, 0 TO DISABLE (for drift testing)

// PID Controller parameters - RE-TUNED for SMOOTHNESS
// KP/KD lowered significantly. KI value only used if ENABLE_PID_INTEGRAL is 1.
#define PID_KP 0.8f                  // Proportional gain (significantly reduced) - Use 'f' suffix for float literals
#define PID_KI 0.002f                // Integral gain (value used only if ENABLE_PID_INTEGRAL is 1)
#define PID_KD 0.05f                 // Derivative gain (significantly reduced)
#define PID_EFFORT_LIMIT 0.2f        // Control effort limit (rad/s) - significantly lowered
#define PID_INTEGRAL_LIMIT 0.02f     // Anti-windup integral limit (reduced)
#define PID_INTEGRAL_DECAY 0.98f     // Integral term decay factor (slightly increased decay)

// Debug flags - Keep low to save memory
#define DEBUG_SERVO_MOVEMENT 1       // Enable basic movement/command logging
#define DEBUG_EXTREME 0
#define DIRECT_TEST_ENABLED 0        // Disable direct test after initial verification

// Filtering parameters - STRONGER filtering
#define FILTER_ALPHA 0.05f           // Low-pass filter coefficient (0-1) - much lower for more smoothing
#define VELOCITY_ALPHA 0.05f         // Velocity estimation filter coefficient - lower for smoother velocity

// Joint definitions
#define NUM_SERVOS 12
#define NUM_LEGS 4
#define JOINTS_PER_LEG 3

// Joint types
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
const int legJointMap[NUM_LEGS][JOINTS_PER_LEG] = {
  {0, 4, 8},   // Front Left (shoulder, thigh, knee) - Servo indices
  {1, 5, 9},   // Front Right
  {2, 6, 10},  // Back Left
  {3, 7, 11}   // Back Right
};

// Servo calibration values (min, max pulse width for each servo)
// These map radians based on jointLimits. CRITICAL for drift. Verify physically!
// Indices: 0-3=shoulders, 4-7=thighs, 8-11=knees (FL, FR, BL, BR)
// Example values - MUST BE CALIBRATED FOR YOUR ROBOT
int servoMin[NUM_SERVOS] = {315, 305, 290, 360, 380, 185, 385, 215, 375, 185, 395, 185};
int servoMax[NUM_SERVOS] = {395, 225, 370, 280, 190, 375, 195, 405, 185, 375, 205, 375};
// NOTE: For inverted joints (like left knees potentially, depending on mechanics),
// the *mapping* logic handles inversion, min should still be < max pulse if servo rotates normally.
// If servo *hardware* rotates opposite, swap min/max physically. Usually handled in mapping.

// Joint limit structure
struct JointLimit {
  int8_t type;
  float minAngle; // Min angle (rad)
  float maxAngle; // Max angle (rad)
  float safePos;  // Safe position (rad)
};

// Joint limits for each type (shoulder, thigh, knee)
// These define the desired *radian* range.
const JointLimit jointLimits[JOINTS_PER_LEG] = {
  {SHOULDER, -0.35f, 0.35f, 0.0f},
  {THIGH,    -0.79f, 0.79f, 0.3f},
  {KNEE,     -0.79f, 0.79f, -0.75f} // NOTE: This is the range for the joint type. Mapping handles direction.
};

//==========================//
//       JOINT STATE        //
//==========================//
struct JointState {
  float targetPos;      // Commanded position (rad) from Jetson
  float currentPos;     // Filtered/estimated current position (rad)
  float prevPos;        // Previous position for velocity calc (rad)
  float velocity;       // Filtered estimated velocity (rad/s)
  float integral;       // Integral term for PID
  uint32_t lastMove;    // Last time joint moved significantly (millis)
  uint8_t limitIndex:4; // Index to jointLimits (0-2)
  uint8_t flags:4;      // (bit 0: isCalibrated - currently unused effectively)
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
static uint32_t previousMicros = 0;
static float actualDt = 0.005f; // Target dt

//==========================//
//    DIAGNOSTIC METRICS    //
//==========================//
struct ControlMetrics {
  uint32_t loopCount;
  uint32_t commandCount;
  uint32_t timeoutCount;
  uint32_t maxLoopTime;
  float avgLoopTime;
};
ControlMetrics metrics = {0};

//==========================//
//    FORWARD DECLARATIONS  //
//==========================//
void setupJoints();
void updateServos();
void processSerialData();
void parseCommand(const char* command);
bool validateTargetPositions(); // Renamed for clarity
void moveToSafePosition();
float applyPidControl(int jointIdx, float dt);
void setServoPulse(int jointIdx, int pulse); // Changed from setServoPosition for clarity
float mapAngleToServoPulse(int jointIdx, float angle); // Renamed for clarity
float constrainJointAngle(int jointIdx, float angle);
void sendStatus();
void sendFeedback();
void testMovement(); // Keep for basic testing

// Arduino Helper (if not already standard)
float constrain(float x, float a, float b) {
    if(x < a) return a;
    if(x > b) return b;
    return x;
}
int constrain(int x, int a, int b) {
    if(x < a) return a;
    if(x > b) return b;
    return x;
}


/*********************************************************************
 * SETUP FUNCTION
 *********************************************************************/
void setup() {
  Serial.begin(115200);
  // Don't wait for Serial connection indefinitely on Teensy/some boards
  // while (!Serial && millis() < 2000); // Wait max 2 seconds

  Serial.println("Starting Harold Quadruped Robot Controller (REVISED)...");

  // Send handshake early and often
  for (int i = 0; i < 5; i++) {
    Serial.println(HANDSHAKE_MSG);
    delay(20);
  }

  Serial.println("Initializing PWM driver...");
  pwm.begin();
  pwm.setOscillatorFrequency(27000000); // Verify this freq for your board/driver
  pwm.setPWMFreq(SERVO_FREQ);
  delay(100);
  Serial.println("PWM driver initialized.");

#if DIRECT_TEST_ENABLED
  Serial.println("Direct Servo Test (Joint 0):");
  pwm.setPWM(0, 0, servoMin[0]); delay(500);
  int centerPulse = (servoMin[0] + servoMax[0]) / 2;
  pwm.setPWM(0, 0, centerPulse); delay(500);
  pwm.setPWM(0, 0, servoMax[0]); delay(500);
  pwm.setPWM(0, 0, centerPulse); delay(500); // Return to center
  Serial.println("Direct Test Done.");
#endif

  setupJoints();

  lastControlMicros = micros();
  previousMicros = lastControlMicros;
  lastCommandMillis = millis();

  // Memory check (optional, may vary by board)
  // ... (memory check code - keep if useful)

  Serial.println("Harold robot controller initialized and ready.");
  Serial.println(HANDSHAKE_MSG); // Send one last handshake
}

/*********************************************************************
 * MAIN LOOP
 *********************************************************************/
void loop() {
  // Process incoming serial data non-blockingly
  processSerialData();

  // Run control loop at precise intervals
  uint32_t currentMicros = micros();
  if (currentMicros - lastControlMicros >= CONTROL_INTERVAL_US) {
    // Calculate actual time delta
    actualDt = (currentMicros - previousMicros) / 1000000.0f;
    previousMicros = currentMicros;

    // Safety check for reasonable dt (e.g., after blocking calls or on first loop)
    if (actualDt <= 0.0f || actualDt > 0.02f) {
      actualDt = CONTROL_INTERVAL_US / 1000000.0f;
    }

    uint32_t loopStartMicros = micros();

    // Check for command timeout
    uint32_t currentMillis = millis();
    if (currentMillis - lastCommandMillis > COMMAND_TIMEOUT_MS) {
      if (metrics.timeoutCount % 100 == 0) { // Prevent spamming serial
         Serial.println("Command timeout - moving to safe position.");
      }
      moveToSafePosition(); // Update targetPos to safe values
      metrics.timeoutCount++;
      // Reset timer only if we actually timed out, otherwise a single valid command resets it
      // lastCommandMillis = currentMillis; // Resetting here might mask frequent timeouts
    }

    // Update servos based on PID and filtering
    updateServos();

    // Update loop timing
    lastControlMicros = currentMicros; // Use currentMicros for better accuracy

    // Update performance metrics
    uint32_t loopTime = micros() - loopStartMicros;
    metrics.loopCount++;
    metrics.avgLoopTime = (metrics.avgLoopTime * 0.99f) + (loopTime * 0.01f);
    if (loopTime > metrics.maxLoopTime) metrics.maxLoopTime = loopTime;

    // Periodically send feedback (e.g., every 100ms = 20 loops)
    static uint8_t feedbackCounter = 0;
    if (++feedbackCounter >= 20) { // Send feedback approx every 100ms
        feedbackCounter = 0;
        // sendFeedback(); // Send limited feedback if needed
    }

    // Periodically print debug info (throttled)
    static uint32_t lastDebugTime = 0;
    if (DEBUG_SERVO_MOVEMENT && (currentMillis - lastDebugTime > DEBUG_INTERVAL_MS)) {
        lastDebugTime = currentMillis;
        Serial.print("J0 State: Tgt="); Serial.print(joints[0].targetPos, 3);
        Serial.print(" Cur="); Serial.print(joints[0].currentPos, 3);
        Serial.print(" Vel="); Serial.print(joints[0].velocity, 3);
#if ENABLE_PID_INTEGRAL
        Serial.print(" Int="); Serial.print(joints[0].integral, 4);
#endif
        int pulse = mapAngleToServoPulse(0, joints[0].currentPos);
        Serial.print(" Pulse="); Serial.println(pulse);
    }
  }

   // Less frequent tasks (like sending handshake if idle)
   static uint32_t lastIdleTaskTime = 0;
   uint32_t nowMillis = millis();
   if (nowMillis - lastIdleTaskTime > 1000) { // Check every second
       lastIdleTaskTime = nowMillis;
       // If no commands for a while, periodically send handshake
       if (nowMillis - lastCommandMillis > 5000) {
           Serial.println(HANDSHAKE_MSG);
       }
   }
}

/*********************************************************************
 * JOINT SETUP AND CONFIGURATION
 *********************************************************************/
void setupJoints() {
  Serial.println("Setting up joints...");
  for (int leg = 0; leg < NUM_LEGS; leg++) {
    for (int joint = 0; joint < JOINTS_PER_LEG; joint++) {
      int idx = legJointMap[leg][joint];
      joints[idx].limitIndex = joint; // Link to the correct limits/safe pos

      // Determine the correct safe position based on joint type and leg side
      float safePos = jointLimits[joint].safePos;

      // **CRITICAL KNEE LOGIC:** Assume safePos in jointLimits is for the RIGHT side convention.
      // If this joint is a LEFT KNEE, we need to command the *logical* safe position.
      // The mapAngleToServoPulse function will handle the inversion for left knees.
      // bool isLeftKnee = (joint == KNEE && (leg == 0 || leg == 2)); // Leg 0=FL, Leg 2=BL
      // No modification to safePos needed here based on side.

      // Initialize state
      joints[idx].targetPos = safePos;
      joints[idx].currentPos = safePos; // Start at safe position
      joints[idx].prevPos = safePos;
      joints[idx].velocity = 0.0f;
      joints[idx].integral = 0.0f;
      joints[idx].lastMove = millis();
      joints[idx].flags = 0; // Reset flags

      // Move servo to initial position *directly* (bypass PID for init)
      // Important: ensure safePos is within limits first!
      float constrainedSafePos = constrainJointAngle(idx, safePos);
      if (fabs(constrainedSafePos - safePos) > 1e-5) { // Check if clamped (using float comparison)
           Serial.print("WARNING: Joint "); Serial.print(idx);
           Serial.print(" safe position "); Serial.print(safePos);
           Serial.print(" clamped to "); Serial.println(constrainedSafePos);
      }
      int initialPulse = mapAngleToServoPulse(idx, constrainedSafePos);
      setServoPulse(idx, initialPulse);
      delay(20); // Small delay between servos
    }
  }

  Serial.println("Joints initialized. Allowing servos to settle...");
  delay(1000); // Allow time for servos to reach initial positions

  // Optional: Simple test movement after settling
  // testMovement(); // Keep disabled unless needed for specific tests

  Serial.println("Joints ready.");
}

/*********************************************************************
 * SERVO CONTROL FUNCTIONS
 *********************************************************************/
void updateServos() {
  uint32_t now = millis();

  for (int i = 0; i < NUM_SERVOS; i++) {
    // 1. Calculate PID Control Output
    float controlOutput = applyPidControl(i, actualDt); // Returns target velocity adjustment

    // 2. Update Position based on Control Output (Simple Integration)
    // This is essentially target_velocity * dt
    float desiredChange = controlOutput * actualDt;

    // --- Simple Rate Limiting (Alternative to complex acceleration limit) ---
    // Limit the change per time step directly
    const float MAX_CHANGE_PER_STEP = 0.01f; // Max radians change per 5ms loop (tune this)
    desiredChange = constrain(desiredChange, -MAX_CHANGE_PER_STEP, MAX_CHANGE_PER_STEP);
    // ---

    float newPosRaw = joints[i].currentPos + desiredChange;

    // 3. Constrain to Joint Limits
    float newPosClamped = constrainJointAngle(i, newPosRaw);

    // 4. Update Previous Position (using current filtered position before update)
    joints[i].prevPos = joints[i].currentPos;

    // 5. Apply Low-Pass Filter to Smooth Position
    joints[i].currentPos = (FILTER_ALPHA * newPosClamped) + ((1.0f - FILTER_ALPHA) * joints[i].currentPos);

    // 6. Estimate Velocity (Filtered difference)
    // Use filtered positions for a smoother velocity estimate
    float rawVelocity = 0.0f;
    if (actualDt > 1e-7) { // Avoid division by zero if dt is extremely small
        rawVelocity = (joints[i].currentPos - joints[i].prevPos) / actualDt;
    }
    joints[i].velocity = (VELOCITY_ALPHA * rawVelocity) + ((1.0f - VELOCITY_ALPHA) * joints[i].velocity);

    // 7. Simplified Anti-Drift / Holding
    float posError = joints[i].targetPos - joints[i].currentPos;
    // Snap to target if very close and velocity is near zero
    if (fabs(posError) < 0.015f && fabs(joints[i].velocity) < 0.05f) {
       joints[i].currentPos = joints[i].targetPos;
       joints[i].velocity = 0.0f;
#if ENABLE_PID_INTEGRAL
       joints[i].integral = 0.0f; // Reset integral when snapped only if enabled
#endif
    }
    // --- Removed complex holding modes for simplicity ---

    // 8. Check for significant movement (for timeout detection)
    if (fabs(joints[i].currentPos - joints[i].prevPos) > 0.001f) { // Small threshold
        joints[i].lastMove = now;
    }

    // 9. Command the Servo
    int pulse = mapAngleToServoPulse(i, joints[i].currentPos);
    setServoPulse(i, pulse);
  }

  // Check for stuck joints (simplified, checks only J0 if DEBUG enabled)
#if DEBUG_SERVO_MOVEMENT
  static uint32_t lastStuckCheck = 0;
  if (now - lastStuckCheck > 2000) { // Check every 2 seconds
      lastStuckCheck = now;
      int checkJoint = 0; // Check only first joint
      if (fabs(joints[checkJoint].targetPos - joints[checkJoint].currentPos) > 0.1f && // Significant error
          now - joints[checkJoint].lastMove > MOVEMENT_TIMEOUT_MS) {
          Serial.print("WARNING: Joint "); Serial.print(checkJoint); Serial.println(" may be stuck!");
          // Consider adding recovery logic here if needed
          joints[checkJoint].lastMove = now; // Reset timer to avoid repeated warnings
      }
  }
#endif
}

float applyPidControl(int jointIdx, float dt) {
  // Calculate Error
  float error = joints[jointIdx].targetPos - joints[jointIdx].currentPos;

// --- Integral Term Calculation (Conditionally Compiled) ---
#if ENABLE_PID_INTEGRAL
    // Simplified Integral Logic: Integrate if error is persistent but not huge
    // And decay over time
    joints[jointIdx].integral = joints[jointIdx].integral * PID_INTEGRAL_DECAY;

    // Add to integral only if error is reasonably small (avoids windup during big moves)
    // and velocity is low (trying to settle)
    if (fabs(error) < 0.2f && fabs(joints[jointIdx].velocity) < 0.1f ) {
         joints[jointIdx].integral += error * dt;

        // Anti-Windup Clamping
        joints[jointIdx].integral = constrain(joints[jointIdx].integral, -PID_INTEGRAL_LIMIT, PID_INTEGRAL_LIMIT);
    }

    // If error is very small, aggressively reduce integral to prevent overshoot/drift
    if (fabs(error) < 0.02f) {
        joints[jointIdx].integral *= 0.8f; // Faster decay when very close
    }
    float integralTerm = joints[jointIdx].integral * PID_KI;

#else // If ENABLE_PID_INTEGRAL is 0, ensure integral term is zero
    float integralTerm = 0.0f;
    joints[jointIdx].integral = 0.0f; // Keep state zeroed if not used
#endif
// --- End Integral Term ---


  // Proportional Term
  float proportional = error * PID_KP;

  // Derivative Term (using filtered velocity)
  // Negative sign because derivative opposes velocity towards target
  float derivative = -joints[jointIdx].velocity * PID_KD;

  // Combine Terms
  float output = proportional + integralTerm + derivative;

  // Clamp Output Effort
  output = constrain(output, -PID_EFFORT_LIMIT, PID_EFFORT_LIMIT);

  return output; // This is the desired velocity adjustment
}

// Renamed for clarity: Sets the raw PWM pulse
void setServoPulse(int jointIdx, int pulse) {
  // Constrain pulse to absolute hardware limits
  pulse = constrain(pulse, SERVO_MIN_PULSE, SERVO_MAX_PULSE);

  // Set the PWM
  pwm.setPWM(jointIdx, 0, pulse);
}

// Renamed for clarity: Maps angle (radians) to PWM pulse
float mapAngleToServoPulse(int jointIdx, float angle) {
    // 1. Get Limits and Type for this joint
    uint8_t limitIdx = joints[jointIdx].limitIndex; // 0=Shoulder, 1=Thigh, 2=Knee
    float minAngle = jointLimits[limitIdx].minAngle;
    float maxAngle = jointLimits[limitIdx].maxAngle;
    int minPulse = servoMin[jointIdx];
    int maxPulse = servoMax[jointIdx];

    // 2. Determine Leg and Joint Type
    int legIndex = -1; // 0=FL, 1=FR, 2=BL, 3=BR
    int jointType = limitIdx; // Shoulder, Thigh, Knee

    // Find leg index (can be optimized if needed)
     for (int leg = 0; leg < NUM_LEGS; leg++) {
        for (int j = 0; j < JOINTS_PER_LEG; j++) {
             if (legJointMap[leg][j] == jointIdx) {
                 legIndex = leg;
                 break;
             }
        }
         if (legIndex != -1) break;
     }
     // If legIndex is still -1 here, there's an issue with legJointMap or jointIdx
     if (legIndex == -1) {
        Serial.print("ERROR: Could not find leg for joint index "); Serial.println(jointIdx);
        // Return a safe center pulse or handle error appropriately
        return (SERVO_MIN_PULSE + SERVO_MAX_PULSE) / 2.0f;
     }


    // 3. Handle **Left Knee Inversion**
    // Assume the calibration `servoMin`/`servoMax` corresponds physically
    // to the `minAngle`/`maxAngle` range *for that specific servo*.
    // If the left knees need to move "opposite" to the right knees for the same
    // logical command (e.g., positive angle = forward bend for both),
    // we invert the *input angle* for the left knees before mapping.
    bool isLeftKnee = (jointType == KNEE && (legIndex == 0 || legIndex == 2)); // FL or BL
    float angleToMap = angle; // Use a temporary variable
    if (isLeftKnee) {
        angleToMap = -angle; // Invert the logical angle command for mapping
    }

    // 4. Clamp the (potentially inverted) angle to the defined limits
    // This prevents mapping outside the calibrated range.
    angleToMap = constrain(angleToMap, minAngle, maxAngle);

    // 5. Map Angle (rad) to Normalized Position (0.0 to 1.0)
    float normalizedPos = 0.0f;
    float angleRange = maxAngle - minAngle;
    if (fabs(angleRange) > 1e-6) { // Avoid division by zero or near-zero
        normalizedPos = (angleToMap - minAngle) / angleRange;
    }
    // Clamp normalized position just in case of floating point inaccuracies near limits
    normalizedPos = constrain(normalizedPos, 0.0f, 1.0f);


    // 6. Map Normalized Position to Servo Pulse Width
    // Linear interpolation between minPulse and maxPulse
    float pulseRange = (float)(maxPulse - minPulse); // Cast difference to float for calculation
    float pulse = (float)minPulse + normalizedPos * pulseRange;


    // Debug print for specific joint mapping (throttled)
    static uint32_t lastMapDebugTime = 0;
    uint32_t now = millis();
    // Debug FL Knee (idx 8) or FR Knee (idx 9)
    if ((jointIdx == 8 || jointIdx == 9) && (now - lastMapDebugTime > 2000)) {
        lastMapDebugTime = now;
        Serial.print("Map J"); Serial.print(jointIdx);
        Serial.print(": AngleIn="); Serial.print(angle, 3); // Original angle
        Serial.print(" MapAngle="); Serial.print(angleToMap, 3); // Angle used for mapping
        Serial.print(" Norm="); Serial.print(normalizedPos, 3);
        Serial.print(" Pulse="); Serial.println((int)round(pulse)); // Round pulse for printing
    }


    return pulse; // Return as float, will be cast before sending to setPWM
}


float constrainJointAngle(int jointIdx, float angle) {
  uint8_t limitIdx = joints[jointIdx].limitIndex;
  // Use a small tolerance to avoid issues at the exact limits due to float inaccuracy
  float tolerance = 0.001f;
  return constrain(angle, jointLimits[limitIdx].minAngle + tolerance, jointLimits[limitIdx].maxAngle - tolerance);
}

void moveToSafePosition() {
  // Update target positions to safe values
  // The PID loop will then smoothly move the joints there
  bool changed = false;
  for (int leg = 0; leg < NUM_LEGS; leg++) {
    for (int joint = 0; joint < JOINTS_PER_LEG; joint++) {
      int idx = legJointMap[leg][joint];
      float safePos = jointLimits[joint].safePos; // Get safe pos for this type

      // **LEFT KNEE LOGIC CONSISTENCY:**
      // The target position should be the *logical* safe position.
      // mapAngleToServoPulse handles the inversion needed for left knees.
      // No inversion needed here for the target.

      // Only update if the target is not already the safe position
      if (fabs(joints[idx].targetPos - safePos) > 1e-5) { // Use float comparison
          joints[idx].targetPos = safePos;
#if ENABLE_PID_INTEGRAL
          // Reset integral immediately when commanding safe pos to prevent overshoot/windup
          joints[idx].integral = 0.0f;
#endif
          changed = true;
      }
    }
  }
  // if (changed) { // Reduce serial spamming
  //   Serial.println("Target set to safe positions.");
  // }
}

/*********************************************************************
 * SERIAL COMMUNICATION FUNCTIONS
 *********************************************************************/
void processSerialData() {
    // Quick check for handshake request before buffering more data
    if (Serial.peek() == 'R') { // Check if first char is 'R'
         String maybeReady = "";
         // Read until '#' or timeout/buffer full to get potential "READY?#"
         unsigned long readStart = millis();
         while (Serial.available() > 0 && (millis() - readStart < 50) && maybeReady.length() < 10) {
             char c = Serial.read();
             maybeReady += c;
             if (c == '#') break;
         }

         if (maybeReady.indexOf("READY?") != -1) {
             // Send handshake multiple times
             for(int i=0; i<3; ++i) {
                Serial.println(HANDSHAKE_MSG);
                delay(5); // Small delay between sends
             }
             lastCommandMillis = millis(); // Treat handshake request as activity
             bufferIndex = 0; // Clear buffer after handling handshake
             incomingBuffer[0] = '\0';
             // Consume any remaining characters from this potential command line in the hardware buffer
             while(Serial.available() > 0) Serial.read();
             return; // Handled
         } else {
             // It wasn't "READY?#", put the read data into the main buffer
              if (bufferIndex + maybeReady.length() < MAX_MESSAGE_LENGTH -1) {
                  // Copy maybeReady content into incomingBuffer
                  memcpy(incomingBuffer + bufferIndex, maybeReady.c_str(), maybeReady.length());
                  bufferIndex += maybeReady.length();
                  incomingBuffer[bufferIndex] = '\0'; // Null terminate
              } else {
                   // Overflow handling
                   Serial.println("ERR: Buffer overflow on non-handshake read");
                   bufferIndex = 0; incomingBuffer[0] = '\0';
              }
         }
    }


  // Read remaining available chars into buffer
  while (Serial.available() > 0 && bufferIndex < (MAX_MESSAGE_LENGTH - 1)) {
    char c = Serial.read();
    if (c == '\r') continue; // Ignore CR
    if (c == '\n') continue; // Ignore LF, use '#' as delimiter

    if (c == '#') { // Found command delimiter
      incomingBuffer[bufferIndex] = '\0'; // Null-terminate the command
      if (bufferIndex > 0) { // Check if there's an actual command before the '#'
        // Serial.print("Processing command: ["); Serial.print(incomingBuffer); Serial.println("]"); // Debug received command
        parseCommand(incomingBuffer);
      }
      bufferIndex = 0; // Reset buffer for next command
    } else {
      incomingBuffer[bufferIndex++] = c;
    }
  }

  // Buffer overflow check (if loop terminated because buffer is full)
  if (bufferIndex >= (MAX_MESSAGE_LENGTH - 1)) {
    Serial.println("ERR: Command buffer overflow, discarding.");
    bufferIndex = 0; // Discard partial command
    incomingBuffer[0] = '\0';
  }
}


void parseCommand(const char* command) {
  // Handle simple string commands first
  if (strcmp(command, "STATUS") == 0) {
    sendStatus();
    lastCommandMillis = millis(); // Status request counts as activity
    return;
  }
  if (strcmp(command, "STOP") == 0) {
    Serial.println("STOP received - moving to safe position immediately.");
    moveToSafePosition();
    // Force servos to target immediately (bypass PID for emergency stop)
    for (int i = 0; i < NUM_SERVOS; i++) {
        joints[i].currentPos = joints[i].targetPos; // Snap state
        joints[i].velocity = 0.0f;
        joints[i].integral = 0.0f;
        int pulse = mapAngleToServoPulse(i, joints[i].currentPos);
        setServoPulse(i, pulse);
    }
    lastCommandMillis = millis(); // Update timestamp
    return;
  }
  // Handle handshake just in case it gets here via buffer processing logic
  if (strcmp(command, "READY?") == 0) {
     for(int i=0; i<3; ++i) Serial.println(HANDSHAKE_MSG);
     lastCommandMillis = millis();
     return;
  }


  // Attempt to parse joint positions (e.g., "0.1,0.2,...,1.2")
  int jointIdx = 0;
  char* mutableCommand; // Need non-const pointer for strtok_r
  // Duplicate the command string because strtok_r modifies it
  char commandCopy[MAX_MESSAGE_LENGTH];
  strncpy(commandCopy, command, MAX_MESSAGE_LENGTH - 1);
  commandCopy[MAX_MESSAGE_LENGTH - 1] = '\0'; // Ensure null termination
  mutableCommand = commandCopy;

  char* valueStr;
  char* rest = mutableCommand;
  bool parseError = false;
  float parsedValues[NUM_SERVOS]; // Temporarily store parsed values

  while (jointIdx < NUM_SERVOS && (valueStr = strtok_r(rest, ",", &rest))) {
    // Skip leading whitespace if any (shouldn't be if format is strict)
    // while (*valueStr && isspace(*valueStr)) { valueStr++; }

    // Check if the token is empty (e.g., "1.0,,2.0")
    if (*valueStr == '\0') {
      parseError = true;
      Serial.println("ERR: Empty value between commas.");
      break;
    }

    // Convert string part to float using atof
    float value = atof(valueStr);

    // Basic validation: check if atof had issues (imperfect)
    // A more robust method like strtof could be used but adds complexity/code size
    if (value == 0.0f && !(valueStr[0] == '0' || ( (valueStr[0] == '-' || valueStr[0] == '+') && valueStr[1] == '0')) ) {
       // Could be an error if string wasn't clearly zero, e.g., "abc" -> 0.0
       // Check if the string contains non-numeric characters (excluding sign, dot, e/E)
       char* p = valueStr;
       if (*p == '-' || *p == '+') p++; // Skip sign
       bool nonNumericFound = false;
       while (*p) {
           if (!isdigit(*p) && *p != '.' && *p != 'e' && *p != 'E') {
               nonNumericFound = true;
               break;
           }
           p++;
       }
       if (nonNumericFound) {
           // parseError = true; // atof returning 0 for non-numeric is sometimes acceptable
           // Serial.print("WARN: atof parsed non-numeric string '"); Serial.print(valueStr); Serial.println("' as 0.0");
       }
    }

    // Store parsed value temporarily
    parsedValues[jointIdx] = value;
    jointIdx++;
  }

  // Check if we got exactly NUM_SERVOS values AND no parse errors occurred
  if (jointIdx == NUM_SERVOS && !parseError) {
    // Valid command received - Update target positions after constraining
    for (int i = 0; i < NUM_SERVOS; ++i) {
        joints[i].targetPos = constrainJointAngle(i, parsedValues[i]);
    }

    lastCommandMillis = millis(); // Update timestamp for valid command
    metrics.commandCount++;

    // --- REMOVED CRITICAL SECTION ---
    // Let the main loop handle the update smoothly.

    // Basic logging of received command (throttled)
    if (DEBUG_SERVO_MOVEMENT && metrics.commandCount % 20 == 1) { // Log every 20 commands (offset by 1)
        Serial.print("CMD OK ("); Serial.print(metrics.commandCount); Serial.print("): Tgt0=");
        Serial.println(joints[0].targetPos, 3);
    }

  } else {
    // Error condition
    Serial.print("ERR: Invalid command. Parsed ");
    Serial.print(jointIdx); Serial.print(" values. Expected "); Serial.print(NUM_SERVOS);
    if (parseError) Serial.print(" (Parse Error detected)."); else Serial.print(".");
    Serial.print(" CMD: ["); Serial.print(command); Serial.println("]");
  }
}


// Renamed for clarity - Validates TARGET positions (already done in parseCommand)
bool validateTargetPositions() {
  // This function might not be strictly necessary anymore if parseCommand
  // uses constrainJointAngle directly when setting targetPos.
  // Keeping it as a potential safety check layer if needed later.
  bool allValid = true;
  for (int i = 0; i < NUM_SERVOS; i++) {
      float originalTarget = joints[i].targetPos;
      float constrainedTarget = constrainJointAngle(i, originalTarget);
      if (fabs(constrainedTarget - originalTarget) > 1e-5) {
          joints[i].targetPos = constrainedTarget; // Ensure it's constrained
          allValid = false;
          // Optional: Log clamping action
          // Serial.print("WARN: Clamped target for J"); Serial.print(i); ...
      }
  }
  return allValid;
}


/*********************************************************************
 * STATUS AND FEEDBACK FUNCTIONS
 *********************************************************************/
void sendStatus() {
  Serial.print("OK ");
  Serial.print(metrics.commandCount);
  Serial.print(" LoopT(us): avg="); Serial.print(metrics.avgLoopTime, 0);
  Serial.print(" max="); Serial.print(metrics.maxLoopTime);
  Serial.print(" TimeoutCnt="); Serial.println(metrics.timeoutCount);
}

void sendFeedback() {
  // Send current estimated positions (limited to save bandwidth/memory)
  Serial.print("P:"); // Position feedback prefix
  for (int i = 0; i < 4; i++) { // Send first 4 joints only
    Serial.print(joints[i].currentPos, 2); // Send filtered position
    Serial.print(i < 3 ? "," : "");
  }
  Serial.println();
}

/*********************************************************************
 * TEST FUNCTIONS
 *********************************************************************/
void testMovement() {
  // Simple test to move one joint back and forth smoothly using the PID loop
  Serial.println("Running simple movement test on Joint 0...");
  int testJoint = 0;
  float startPos = jointLimits[joints[testJoint].limitIndex].safePos;
  float testOffset = 0.15f; // Radians to move
  int steps = 50; // Number of steps for the movement
  int delayPerStep = 30; // Milliseconds delay between target updates

  // Ensure target starts at the safe position
  joints[testJoint].targetPos = constrainJointAngle(testJoint, startPos);
  Serial.println("Centering...");
  for(int i=0; i<50; ++i) { // Give time for PID to settle at start
        delay(delayPerStep);
        // Process serial to avoid blocking handshake/commands during test
        processSerialData();
  }


  // Move forward smoothly
  Serial.println("Moving forward...");
  for (int i = 0; i <= steps; i++) {
      float target = startPos + (testOffset * i / steps);
      joints[testJoint].targetPos = constrainJointAngle(testJoint, target);
      // Let the main loop's PID handle the update by just delaying
      delay(delayPerStep);
      processSerialData(); // Allow serial processing during delay
  }
   delay(500); // Pause at end
   processSerialData();

  // Move back smoothly
   Serial.println("Moving back...");
  for (int i = 0; i <= steps; i++) {
      float target = startPos + testOffset - (testOffset * i / steps);
      joints[testJoint].targetPos = constrainJointAngle(testJoint, target);
      delay(delayPerStep);
      processSerialData();
  }
  delay(500); // Pause at start pos
  processSerialData();

  Serial.println("Movement test complete.");
}
