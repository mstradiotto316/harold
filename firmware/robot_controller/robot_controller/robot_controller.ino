#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

//==========================//
//     USER ADJUSTMENTS     //
//==========================//
#define MAX_MESSAGE_LENGTH 256
#define SERVO_FREQ 50
#define HANDSHAKE_MSG "ARDUINO_READY"

// Control loop interval and command timeout (in ms)
#define CONTROL_INTERVAL 5     // Control loop interval (5 ms)
#define COMMAND_TIMEOUT 100    // If no valid command in 100 ms, revert to safe posture

// PD Controller Gains for the discrete controller.
// These values have been updated to more closely match the simulation's implicit PD controller
// Simulation values: stiffness=40.0, damping=75.0, effort_limit=0.8
float PD_Kp = 4.0;      // Proportional gain - scaled from simulation stiffness
float PD_Kd = 1.2;      // Derivative gain - scaled from simulation damping
float PD_EFFORT_LIMIT = 0.8;  // Maximum control effort (matches simulation)

//==========================//
// SERVO & ANGLE VARIABLES  //
//==========================//
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(); // default address 0x40

// Ring-buffer for incoming serial data
static char incomingBuffer[MAX_MESSAGE_LENGTH];
static int bufferIndex = 0;

// We have 12 servos (indices 0..11).
// currentPos: estimated joint positions (radians)
// targetPos: commanded joint positions (radians)
// prevPos: previous current positions (for velocity estimation)
static float currentPos[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static float targetPos[12]  = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static float prevPos[12]    = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// Define safe (default) positions
static float safePos[12] = {0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3, -0.75, -0.75, -0.75, -0.75};

// Min / max values are mirrored so that "positive" and "negative" positions all mean the same thing
int SERVO_MIN[12]    = {315, 305, 290, 360,   380, 185, 385, 215,   375, 185, 395, 185};
int SERVO_MAX[12]    = {395, 225, 370, 280,   190, 375, 195, 405,   185, 375, 205, 375};

// Angular limits in radians for each servo (set to match simulation parameters)
// For shoulder joints, a small range; for thigh and knee, a wider range.
static float ANGLE_MAX[12] = {
  0.3491, 0.3491, 0.3491, 0.3491, 0.7853, 0.7853, 0.7853, 0.7853, 0.7853, 0.7853, 0.7853, 0.7853
};

static float ANGLE_MIN[12] = {
  -0.3491, -0.3491, -0.3491, -0.3491, -0.7853, -0.7853, -0.7853, -0.7853, -0.7853, -0.7853, -0.7853, -0.7853
};


// For the control loop
static unsigned long lastControlTime = 0;
static unsigned long lastCommandTime = 0;

//----------------------------------------------------------
// setup()
//----------------------------------------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial);
  Serial.println(HANDSHAKE_MSG); // Announce readiness immediately
  
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);
  delay(1000);
  Serial.println("\nCOMMUNICATION LINK ESTABLISHED...");
  startServos();
}

//----------------------------------------------------------
// startServos()
// Moves each servo to its initial (safe) position at startup.
//----------------------------------------------------------
void startServos() {
  Serial.println("STARTING SERVOS...");
  for (int i = 0; i < 12; i++) {
    float anglePct = (safePos[i] - ANGLE_MIN[i]) / (ANGLE_MAX[i] - ANGLE_MIN[i]);
    if (anglePct < 0.0) anglePct = 0.0;
    if (anglePct > 1.0) anglePct = 1.0;
    float servoRange = SERVO_MAX[i] - SERVO_MIN[i];
    float pulse = (anglePct * servoRange) + SERVO_MIN[i];
    pwm.setPWM(i, 0, pulse);
    currentPos[i] = safePos[i];
    prevPos[i] = safePos[i];
    // Initialize targetPos to safe positions
    targetPos[i] = safePos[i];
    delay(250);
  }
  // Also initialize the last command time to now.
  lastCommandTime = millis();
  Serial.println("STARTUP COMPLETE!");
}

//----------------------------------------------------------
// isValidNumericChar()
//----------------------------------------------------------
bool isValidNumericChar(char c) {
  return (isDigit(c) || c == '.' || c == '-');
}

//----------------------------------------------------------
// messageDecode()
// Parses up to 12 comma-separated floats from the incoming message.
//----------------------------------------------------------
bool messageDecode(const char* buffer) {
  String angleStr = "";
  int servoCount = 0;
  
  // Handle handshake request.
  if (strstr(buffer, "READY?") != NULL) {
    Serial.println(HANDSHAKE_MSG);
    return false; // Not a position command.
  }
  
  for (int i = 0; i < MAX_MESSAGE_LENGTH; i++) {
    char c = buffer[i];
    if (c == '\0') break;
    if (c == ',') {
      if (angleStr.length() > 0) {
        if (servoCount < 12) {
          targetPos[servoCount] = angleStr.toFloat();
          servoCount++;
          angleStr = "";
        } else {
          Serial.println("ERROR: Too many servo values received");
          return false;
        }
      }
    } else {
      if (isValidNumericChar(c)) {
        angleStr += c;
      }
    }
  }
  if (angleStr.length() > 0) {
    if (servoCount < 12) {
      targetPos[servoCount] = angleStr.toFloat();
      servoCount++;
    } else {
      Serial.println("ERROR: Too many servo values received");
      return false;
    }
  }
  if (servoCount != 12) {
    Serial.print("WARNING: Received incorrect number of servo values: ");
    Serial.println(servoCount);
    return false;
  }
  // Update the time of the last valid command.
  lastCommandTime = millis();
  Serial.println("\nACK: Position command received");
  return true;
}

//----------------------------------------------------------
// processIncomingSerialData()
// Reads incoming serial data and processes messages delimited by '#'
//----------------------------------------------------------
void processIncomingSerialData() {
  // 1) Read all available bytes.
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\r' || c == '\n') continue;
    if (bufferIndex < (MAX_MESSAGE_LENGTH - 1)) {
      incomingBuffer[bufferIndex++] = c;
      incomingBuffer[bufferIndex] = '\0';
    } else {
      Serial.println("**ERROR: Maximum message length exceeded. Data was not captured!**");
      bufferIndex = 0;
      incomingBuffer[0] = '\0';
    }
  }
  
  // 2) Look for '#' delimiters.
  int startIdx = 0;
  while (true) {
    char* hashPtr = strchr(incomingBuffer + startIdx, '#');
    if (!hashPtr) break;
    int msgLength = hashPtr - (incomingBuffer + startIdx);
    if (msgLength < 0) break;
    char temp[MAX_MESSAGE_LENGTH];
    strncpy(temp, incomingBuffer + startIdx, msgLength);
    temp[msgLength] = '\0';
    messageDecode(temp);  // Decode the message.
    startIdx = (hashPtr - incomingBuffer) + 1;
  }
  
  // 3) Shift leftover data.
  if (startIdx > 0) {
    int shiftSize = bufferIndex - startIdx;
    if (shiftSize < 0) shiftSize = 0;
    memmove(incomingBuffer, incomingBuffer + startIdx, shiftSize);
    bufferIndex = shiftSize;
    incomingBuffer[bufferIndex] = '\0';
  }
}

//----------------------------------------------------------
// updateServoPosWithPD()
// Uses a PD controller to update currentPos toward targetPos.
// Also checks the watchdog timeout and, if expired, reverts targetPos to safePos.
//----------------------------------------------------------
void updateServoPosWithPD() {
  // Check for command timeout:
  if (millis() - lastCommandTime > COMMAND_TIMEOUT) {
    // No new command has been received recently ? revert target positions.
    for (int i = 0; i < 12; i++) {
      targetPos[i] = safePos[i];
    }
  }
  
  float dt = CONTROL_INTERVAL / 1000.0;  // Convert ms to seconds.
  for (int i = 0; i < 12; i++) {
    float error = targetPos[i] - currentPos[i];
    float velocity = (currentPos[i] - prevPos[i]) / dt;
    
    // Calculate control effort
    float control = PD_Kp * error - PD_Kd * velocity;
    
    // Apply effort limit (matches simulation's effort_limit)
    if (control > PD_EFFORT_LIMIT) control = PD_EFFORT_LIMIT;
    if (control < -PD_EFFORT_LIMIT) control = -PD_EFFORT_LIMIT;
    
    prevPos[i] = currentPos[i];
    currentPos[i] = currentPos[i] + control * dt;
    setServoToPosition(i, currentPos[i]);
  }
}

//----------------------------------------------------------
// setServoToPosition()
// Converts a joint angle (radians) to a PWM pulse (respecting limits)
// and commands the servo.
//----------------------------------------------------------
void setServoToPosition(int servo, float position) {
  float anglePct = (position - ANGLE_MIN[servo]) / (ANGLE_MAX[servo] - ANGLE_MIN[servo]);
  if (anglePct < 0.0) anglePct = 0.0;
  if (anglePct > 1.0) anglePct = 1.0;
  float servoRange = SERVO_MAX[servo] - SERVO_MIN[servo];
  float pulse = (anglePct * servoRange) + SERVO_MIN[servo];
  pwm.setPWM(servo, 0, pulse);
}

//----------------------------------------------------------
// loop()
// Main control loop: process serial data and update servos at fixed intervals.
//----------------------------------------------------------
void loop() {
  processIncomingSerialData();
  
  unsigned long now = millis();
  if (now - lastControlTime >= CONTROL_INTERVAL) {
    lastControlTime = now;
    updateServoPosWithPD();
  }
}
