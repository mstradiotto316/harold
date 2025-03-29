/******************************************************************
 * Harold Quadruped Robot Controller
 * 
 * Features:
 * - Stable PID control with anti-drift mechanisms
 * - Low-pass filtering to reduce jitter
 * - Precise timing control
 * - Movement interpolation
 * - Robust error handling and safety features
 ******************************************************************/

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

//==========================//
//     GLOBAL CONSTANTS     //
//==========================//
#define MAX_MESSAGE_LENGTH 128  // Reduced from 256 to save memory
#define SERVO_FREQ 50
#define HANDSHAKE_MSG "ARDUINO_READY"

// Control timing parameters (in microseconds for precision)
#define CONTROL_INTERVAL_US 5000     // 5ms (200Hz) control loop
#define COMMAND_TIMEOUT_MS 1000      // Timeout for reverting to safe position (increased to 1000ms)
#define MOVEMENT_TIMEOUT_MS 5000     // Timeout for detecting stuck movements
#define DEBUG_INTERVAL_MS 1000       // Interval for debug messages

// Servo physical limits (in pulse width)
#define SERVO_MIN_PULSE 150          // Absolute minimum pulse width
#define SERVO_MAX_PULSE 600          // Absolute maximum pulse width

// PID Controller parameters
#define PID_KP 5.0                   // Proportional gain
#define PID_KI 0.02                  // Integral gain
#define PID_KD 1.0                   // Derivative gain
#define PID_EFFORT_LIMIT 0.8         // Control effort limit (rad/s)
#define PID_INTEGRAL_LIMIT 0.1       // Anti-windup integral limit
#define PID_INTEGRAL_DECAY 0.95      // Integral term decay factor

// Filtering parameters
#define FILTER_ALPHA 0.7             // Low-pass filter coefficient (0-1)
#define VELOCITY_ALPHA 0.8           // Velocity estimation filter coefficient

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

// Create PWM driver instance
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

//==========================//
//    JOINT CONFIGURATION   //
//==========================//

// Leg and joint indices organization
const int legJointMap[NUM_LEGS][JOINTS_PER_LEG] = {
  {0, 4, 8},   // Front Left (shoulder, thigh, knee)
  {1, 5, 9},   // Front Right
  {2, 6, 10},  // Back Left
  {3, 7, 11}   // Back Right
};

// Servo calibration values (min, max pulse width for each servo)
// These values map from -1.0 to +1.0 in joint space
// Ensure min < max for proper mapping (except for inverted joints)
// Indices: 0-3=shoulders, 4-7=thighs, 8-11=knees
// Knee order: 8=FL, 9=FR, 10=BL, 11=BR
// For left knees (8, 10), invert min/max to make negative positions point backward
int servoMin[NUM_SERVOS] = {315, 305, 290, 360, 380, 185, 385, 215, 375, 185, 395, 185};
int servoMax[NUM_SERVOS] = {395, 225, 370, 280, 190, 375, 195, 405, 185, 375, 205, 375};

// Angular limits (radians) for each joint type - DEPRECATED, use jointLimits struct instead
const float jointAngles[JOINTS_PER_LEG][2] = {
  {-0.35, 0.35},  // Shoulder min/max
  {-0.79, 0.79},  // Thigh min/max
  {-0.79, 0.79}   // Knee min/max
};

// Safe position for each joint type (in radians)
const float safePositions[JOINTS_PER_LEG] = {
  0.0,   // Shoulder
  0.3,   // Thigh
  -0.75  // Knee
};

//==========================//
//       JOINT STATE        //
//==========================//

// Joint limit structure - shared across joint types
struct JointLimit {
  int8_t type;          // 0=shoulder, 1=thigh, 2=knee
  float minAngle;       // Min angle (rad)
  float maxAngle;       // Max angle (rad)
  float safePos;        // Safe position (rad)
};

// Current state of each joint - optimized for memory
struct JointState {
  float targetPos;      // Commanded position (rad)
  float currentPos;     // Current position (rad)
  float prevPos;        // Previous position for velocity (rad)
  float velocity;       // Estimated velocity (rad/s)
  float integral;       // Integral term for PID
  uint32_t lastMove;    // Last time joint moved significantly
  uint8_t limitIndex:4; // Index to limits lookup (0-2)
  uint8_t flags:4;      // Bit flags (bit 0: isCalibrated)
};

// Joint limits for each type (shoulder, thigh, knee)
const JointLimit jointLimits[JOINTS_PER_LEG] = {
  {SHOULDER, -0.35, 0.35, 0.0},    // Shoulder
  {THIGH, -0.79, 0.79, 0.3},       // Thigh
  {KNEE, -0.79, 0.79, -0.75}       // Knee
};

// Joint state array
JointState joints[NUM_SERVOS];

//==========================//
//   COMMUNICATION BUFFER   //
//==========================//

// Ring-buffer for incoming serial data
static char incomingBuffer[MAX_MESSAGE_LENGTH];
static int bufferIndex = 0;

//==========================//
//      TIMING CONTROL      //
//==========================//

// Timing variables for precise control
static uint32_t lastControlMicros = 0;     // Last control loop time
static uint32_t lastCommandMillis = 0;     // Last valid command time
static uint32_t previousMicros = 0;        // For dt calculation
static float actualDt = 0.005;             // Actual time step in seconds

//==========================//
//    DIAGNOSTIC METRICS    //
//==========================//

// Performance metrics
struct ControlMetrics {
  uint32_t loopCount;          // Total control loops executed
  uint32_t commandCount;       // Total commands received
  uint32_t timeoutCount;       // Times entered safety mode
  uint32_t maxLoopTime;        // Maximum control loop execution time (µs)
  float avgLoopTime;           // Average control loop time (µs)
};

ControlMetrics metrics = {0};

//==========================//
//    FORWARD DECLARATIONS  //
//==========================//

// Core functions
void setupJoints();
void updateServos();
void processSerialData();
void parseCommand(const char* command);
bool validatePositions();
void moveToSafePosition();

// Joint control helpers
float applyPidControl(int jointIdx, float dt);
void setServoPosition(int jointIdx, float angle);
float mapAngleToServo(int jointIdx, float angle);
float constrainJointAngle(int jointIdx, float angle);

// Communication helpers
void sendStatus();
void sendFeedback();

/*********************************************************************
 * SETUP FUNCTION
 *********************************************************************/
void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial);
  
  // Initialize PWM driver
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);
  
  // Wait for hardware to stabilize
  delay(100);
  
  // Configure joint parameters and initialize to safe positions
  setupJoints();
  
  // Initialize timing variables
  lastControlMicros = micros();
  previousMicros = lastControlMicros;
  lastCommandMillis = millis();
  
  // Display memory information
  extern int __heap_start, *__brkval;
  int freeMemory;
  if ((int) __brkval == 0) {
    freeMemory = ((int) &freeMemory) - ((int) &__heap_start);
  } else {
    freeMemory = ((int) &freeMemory) - ((int) __brkval);
  }
  
  // Send handshake message to host immediately
  Serial.println(HANDSHAKE_MSG);
  Serial.println(HANDSHAKE_MSG);  // Send twice to increase chances of detection
  Serial.print("Free memory: ");
  Serial.println(freeMemory);
  Serial.println("Harold robot controller initialized");
}

/*********************************************************************
 * MAIN LOOP
 *********************************************************************/
void loop() {
  // Check for direct handshake request (high priority)
  if (Serial.available() > 4) {  // At least 5 bytes for "READY"
    // Peek at incoming data without removing it
    if (Serial.find("READY")) {
      // Found handshake request, clear the buffer and respond
      while (Serial.available()) Serial.read(); // Clear remaining data
      Serial.println(HANDSHAKE_MSG);
      Serial.println(HANDSHAKE_MSG);
      Serial.println(HANDSHAKE_MSG);
    } else {
      // Process regular commands
      processSerialData();
    }
  } else if (Serial.available() > 0) {
    // Process regular commands
    processSerialData();
  }
  
  // Run control loop at precise intervals
  uint32_t currentMicros = micros();
  if (currentMicros - lastControlMicros >= CONTROL_INTERVAL_US) {
    // Calculate actual time delta (for PID accuracy)
    actualDt = (currentMicros - previousMicros) / 1000000.0;
    previousMicros = currentMicros;
    
    // Safety check for reasonable dt values
    if (actualDt > 0.02 || actualDt <= 0) {
      actualDt = CONTROL_INTERVAL_US / 1000000.0;
    }
    
    // Record loop start time for performance metrics
    uint32_t loopStartMicros = micros();
    
    // Check for command timeout
    uint32_t currentMillis = millis();
    if (currentMillis - lastCommandMillis > COMMAND_TIMEOUT_MS) {
      // Haven't received valid commands recently, revert to safe position
      moveToSafePosition();
      metrics.timeoutCount++;
    }
    
    // Update servos
    updateServos();
    
    // Update control loop timing
    lastControlMicros = currentMicros;
    
    // Update performance metrics
    uint32_t loopTime = micros() - loopStartMicros;
    metrics.loopCount++;
    metrics.avgLoopTime = (metrics.avgLoopTime * 0.99) + (loopTime * 0.01);
    if (loopTime > metrics.maxLoopTime) metrics.maxLoopTime = loopTime;
    
    // Periodically send feedback (every 100ms)
    static uint32_t lastFeedbackTime = 0;
    if (currentMillis - lastFeedbackTime > 100) {
      lastFeedbackTime = currentMillis;
      sendFeedback();
    }
  }
}

/*********************************************************************
 * JOINT SETUP AND CONFIGURATION
 *********************************************************************/
void setupJoints() {
  // Initialize each joint
  for (int leg = 0; leg < NUM_LEGS; leg++) {
    for (int joint = 0; joint < JOINTS_PER_LEG; joint++) {
      int idx = legJointMap[leg][joint];
      
      // Set joint parameters based on joint type
      joints[idx].limitIndex = joint;
      joints[idx].flags = 1; // Set calibrated flag
      
      // Initialize state variables
      joints[idx].targetPos = jointLimits[joint].safePos;
      joints[idx].currentPos = jointLimits[joint].safePos;
      joints[idx].prevPos = jointLimits[joint].safePos;
      joints[idx].velocity = 0.0;
      joints[idx].integral = 0.0;
      joints[idx].lastMove = millis();
    }
  }
  
  // Move all servos to initial safe positions
  moveToSafePosition();
  
  // Allow time for servos to reach positions
  delay(1000);
  
  Serial.println("Joints initialized and in safe position");
}

/*********************************************************************
 * SERVO CONTROL FUNCTIONS
 *********************************************************************/
void updateServos() {
  // Update each joint based on PID control
  for (int i = 0; i < NUM_SERVOS; i++) {
    // Calculate control output using PID
    float controlOutput = applyPidControl(i, actualDt);
    
    // Update position based on control output
    float filtered = joints[i].currentPos;
    float newPos = filtered + controlOutput * actualDt;
    
    // Constrain to valid joint angles
    newPos = constrainJointAngle(i, newPos);
    
    // Store previous position for velocity calculation
    joints[i].prevPos = filtered;
    
    // Apply filtering to smooth motion (directly to currentPos)
    joints[i].currentPos = (FILTER_ALPHA * newPos) + ((1 - FILTER_ALPHA) * filtered);
    
    // Estimate velocity with filtering
    float rawVelocity = (joints[i].currentPos - joints[i].prevPos) / actualDt;
    joints[i].velocity = (VELOCITY_ALPHA * rawVelocity) + ((1 - VELOCITY_ALPHA) * joints[i].velocity);
    
    // Detect if joint is moving significantly
    if (fabs(joints[i].velocity) > 0.02) {
      joints[i].lastMove = millis();
    }
    
    // Command the servo to the filtered position
    setServoPosition(i, joints[i].currentPos);
  }
  
  // Check for stuck joints (no movement despite commands)
  uint32_t currentTime = millis();
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (fabs(joints[i].targetPos - joints[i].currentPos) > 0.05 && 
        currentTime - joints[i].lastMove > MOVEMENT_TIMEOUT_MS) {
      // Joint appears stuck - log but don't interrupt
      static uint32_t lastStuckWarning = 0;
      if (currentTime - lastStuckWarning > DEBUG_INTERVAL_MS) {
        lastStuckWarning = currentTime;
        Serial.print("WARNING: Joint ");
        Serial.print(i);
        Serial.print(" stuck at ");
        Serial.print(joints[i].currentPos);
        Serial.print(" target ");
        Serial.println(joints[i].targetPos);
      }
    }
  }
}

float applyPidControl(int jointIdx, float dt) {
  // Calculate error terms
  float error = joints[jointIdx].targetPos - joints[jointIdx].currentPos;
  
  // Update integral term with anti-windup
  joints[jointIdx].integral = joints[jointIdx].integral * PID_INTEGRAL_DECAY;
  joints[jointIdx].integral += error * dt;
  
  // Clamp integral term to prevent windup
  if (joints[jointIdx].integral > PID_INTEGRAL_LIMIT)
    joints[jointIdx].integral = PID_INTEGRAL_LIMIT;
  else if (joints[jointIdx].integral < -PID_INTEGRAL_LIMIT)
    joints[jointIdx].integral = -PID_INTEGRAL_LIMIT;
  
  // Calculate proportional, integral, and derivative terms
  float proportional = error * PID_KP;
  float integral = joints[jointIdx].integral * PID_KI;
  float derivative = -joints[jointIdx].velocity * PID_KD;
  
  // Combine terms into control output
  float output = proportional + integral + derivative;
  
  // Limit control effort to prevent sudden movements
  if (output > PID_EFFORT_LIMIT)
    output = PID_EFFORT_LIMIT;
  else if (output < -PID_EFFORT_LIMIT)
    output = -PID_EFFORT_LIMIT;
  
  return output;
}

void setServoPosition(int jointIdx, float angle) {
  // Map from radians to servo pulse width
  float pulse = mapAngleToServo(jointIdx, angle);
  
  // Constrain pulse to valid range
  if (pulse < SERVO_MIN_PULSE) pulse = SERVO_MIN_PULSE;
  if (pulse > SERVO_MAX_PULSE) pulse = SERVO_MAX_PULSE;
  
  // Set the servo position
  pwm.setPWM(jointIdx, 0, (uint16_t)pulse);
}

float mapAngleToServo(int jointIdx, float angle) {
  // Map joint angle from radians to normalized position (0.0 to 1.0)
  uint8_t limitIdx = joints[jointIdx].limitIndex;
  float minAngle = jointLimits[limitIdx].minAngle;
  float maxAngle = jointLimits[limitIdx].maxAngle;
  float normalizedPos = (angle - minAngle) / (maxAngle - minAngle);
  
  // Clamp to 0.0 - 1.0 range
  if (normalizedPos < 0.0) normalizedPos = 0.0;
  if (normalizedPos > 1.0) normalizedPos = 1.0;
  
  // Check if this is a left knee joint which has inverted direction
  // (joint 8 = front left knee, joint 10 = back left knee)
  bool isLeftKnee = (jointIdx == 8 || jointIdx == 10);
  if (isLeftKnee) {
    // Invert the normalized position for left knees
    normalizedPos = 1.0 - normalizedPos;
  }
  
  // Map to servo pulse range
  return (normalizedPos * (servoMax[jointIdx] - servoMin[jointIdx])) + servoMin[jointIdx];
}

float constrainJointAngle(int jointIdx, float angle) {
  uint8_t limitIdx = joints[jointIdx].limitIndex;
  if (angle < jointLimits[limitIdx].minAngle) return jointLimits[limitIdx].minAngle;
  if (angle > jointLimits[limitIdx].maxAngle) return jointLimits[limitIdx].maxAngle;
  return angle;
}

void moveToSafePosition() {
  // Update target positions to safe values
  for (int i = 0; i < NUM_SERVOS; i++) {
    uint8_t limitIdx = joints[i].limitIndex;
    joints[i].targetPos = jointLimits[limitIdx].safePos;
  }
}

/*********************************************************************
 * SERIAL COMMUNICATION FUNCTIONS
 *********************************************************************/
void processSerialData() {
  // Read all available serial data
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    // Skip carriage return and newline characters
    if (c == '\r' || c == '\n') continue;
    
    // Add character to buffer if space available
    if (bufferIndex < (MAX_MESSAGE_LENGTH - 1)) {
      incomingBuffer[bufferIndex++] = c;
      incomingBuffer[bufferIndex] = '\0';
    } else {
      // Buffer overrun - clear buffer and report error
      bufferIndex = 0;
      incomingBuffer[0] = '\0';
      Serial.println("ERROR: Buffer overflow");
    }
  }
  
  // Find and process complete messages (delimited by '#')
  int startIdx = 0;
  while (true) {
    char* hashPtr = strchr(incomingBuffer + startIdx, '#');
    if (!hashPtr) break;
    
    // Extract message
    int msgLength = hashPtr - (incomingBuffer + startIdx);
    if (msgLength < 0) break;
    
    char tempMsg[MAX_MESSAGE_LENGTH];
    strncpy(tempMsg, incomingBuffer + startIdx, msgLength);
    tempMsg[msgLength] = '\0';
    
    // Process the message
    parseCommand(tempMsg);
    
    // Update start index for next message
    startIdx = (hashPtr - incomingBuffer) + 1;
  }
  
  // Shift any remaining data to beginning of buffer
  if (startIdx > 0) {
    int remainingLength = bufferIndex - startIdx;
    if (remainingLength > 0) {
      memmove(incomingBuffer, incomingBuffer + startIdx, remainingLength);
      bufferIndex = remainingLength;
    } else {
      bufferIndex = 0;
    }
    incomingBuffer[bufferIndex] = '\0';
  }
}

void parseCommand(const char* command) {
  // Check for handshake request (both with and without the # delimiter)
  if (strstr(command, "READY") != NULL) {
    // Send handshake multiple times to increase likelihood of reception
    Serial.println(HANDSHAKE_MSG);
    Serial.println(HANDSHAKE_MSG);
    Serial.println(HANDSHAKE_MSG);
    return;
  }
  
  // Check for status request
  if (strstr(command, "STATUS") != NULL) {
    sendStatus();
    return;
  }
  
  // Parse joint position command (12 comma-separated values)
  int jointIdx = 0;
  const char* start = command;
  bool validCommand = true;
  
  while (jointIdx < NUM_SERVOS && *start && validCommand) {
    // Convert current value to float using atof
    char buffer[20]; // Temporary buffer for numeric extraction
    int i = 0;
    
    // Skip any leading whitespace
    while (*start && isspace(*start)) start++;
    
    // Extract the numeric part
    while (*start && (isdigit(*start) || *start == '.' || *start == '-' || *start == 'e' || *start == 'E' || *start == '+') && i < 19) {
      buffer[i++] = *start++;
    }
    buffer[i] = '\0';
    
    // Convert to float
    float value = atof(buffer);
    
    // Check if we parsed a value (buffer not empty)
    if (i == 0) {
      validCommand = false;
      break;
    }
    
    // Store value
    joints[jointIdx].targetPos = constrainJointAngle(jointIdx, value);
    jointIdx++;
    
    // Move to next value
    start = strchr(start, ',');
    if (start) start++; // Skip comma
    else if (jointIdx < NUM_SERVOS) {
      validCommand = false; // Not enough values
      break;
    }
  }
  
  // Validate command
  if (validCommand && jointIdx == NUM_SERVOS) {
    // Valid command - update timestamp
    lastCommandMillis = millis();
    metrics.commandCount++;
    
    // Debug output every second to avoid flooding
    static uint32_t lastCmdDebug = 0;
    uint32_t now = millis();
    if (now - lastCmdDebug > DEBUG_INTERVAL_MS) {
      lastCmdDebug = now;
      Serial.print("CMD: ");
      for (int i = 0; i < NUM_SERVOS; i++) {
        Serial.print(joints[i].targetPos, 3);
        Serial.print(i < NUM_SERVOS-1 ? "," : "\n");
      }
    }
    
    // Check for any invalid positions
    validatePositions();
  } else {
    Serial.println("ERROR: Invalid command format");
  }
}

bool validatePositions() {
  // Check all joint positions for validity
  bool allValid = true;
  
  for (int i = 0; i < NUM_SERVOS; i++) {
    uint8_t limitIdx = joints[i].limitIndex;
    
    // Constrain target to valid range
    if (joints[i].targetPos < jointLimits[limitIdx].minAngle || 
        joints[i].targetPos > jointLimits[limitIdx].maxAngle) {
        
      // Clamp to valid range
      joints[i].targetPos = constrainJointAngle(i, joints[i].targetPos);
      allValid = false;
    }
  }
  
  return allValid;
}

/*********************************************************************
 * STATUS AND FEEDBACK FUNCTIONS
 *********************************************************************/
void sendStatus() {
  Serial.print("STATUS:");
  
  // Send calibration status
  Serial.print(" CAL=");
  bool allCalibrated = true;
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (!(joints[i].flags & 0x01)) {  // Check calibrated flag (bit 0)
      allCalibrated = false;
      break;
    }
  }
  Serial.print(allCalibrated ? "1" : "0");
  
  // Send metrics
  Serial.print(" CMDS=");
  Serial.print(metrics.commandCount);
  Serial.print(" TMOUT=");
  Serial.print(metrics.timeoutCount);
  Serial.print(" LOOP_AVG=");
  Serial.print(metrics.avgLoopTime, 1);
  Serial.print(" LOOP_MAX=");
  Serial.print(metrics.maxLoopTime);
  
  // Send timing
  Serial.print(" DT=");
  Serial.print(actualDt * 1000.0, 2);
  
  Serial.println();
}

void sendFeedback() {
  // Only send if host is likely waiting for it
  if (metrics.commandCount > 0) {
    Serial.print("POS:");
    for (int i = 0; i < NUM_SERVOS; i++) {
      Serial.print(joints[i].currentPos, 3);
      if (i < NUM_SERVOS - 1) {
        Serial.print(",");
      }
    }
    Serial.println();
  }
}