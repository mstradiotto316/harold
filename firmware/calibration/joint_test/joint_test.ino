#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ***** Configuration *****
#define SERVO_COUNT     12      // Total number of servos (4 legs × 3 joints per leg)
#define PWM_FREQUENCY   50      // 50 Hz for standard servos
#define TEST_DELAY      2000    // Delay between movements in milliseconds

// ***** Create the PWM Servo Driver Object *****
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Joint names for readable testing
const char* JOINT_NAMES[SERVO_COUNT] = {
  "Front Left Shoulder",    // 0
  "Front Right Shoulder",   // 1
  "Back Left Shoulder",     // 2
  "Back Right Shoulder",    // 3
  "Front Left Thigh",       // 4
  "Front Right Thigh",      // 5
  "Back Left Thigh",        // 6
  "Back Right Thigh",       // 7
  "Front Left Knee",        // 8
  "Front Right Knee",       // 9
  "Back Left Knee",         // 10
  "Back Right Knee"         // 11
};

// Min / max values for each servo
int servoMin[SERVO_COUNT]    = {315, 305, 290, 360,   380, 185, 385, 215,   375, 185, 395, 185};
int servoCenter[SERVO_COUNT] = {355, 265, 330, 320,   285, 280, 290, 310,   280, 280, 300, 280};
int servoMax[SERVO_COUNT]    = {395, 225, 370, 280,   190, 375, 195, 405,   185, 375, 205, 375};

// Safe position to return to between tests
float safePos[SERVO_COUNT] = {0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3, -0.75, -0.75, -0.75, -0.75};

// Angular limits in radians for each servo
float ANGLE_MAX[SERVO_COUNT] = {
  0.3491, 0.3491, 0.3491, 0.3491, 0.7853, 0.7853, 0.7853, 0.7853, 0.7853, 0.7853, 0.7853, 0.7853
};

float ANGLE_MIN[SERVO_COUNT] = {
  -0.3491, -0.3491, -0.3491, -0.3491, -0.7853, -0.7853, -0.7853, -0.7853, -0.7853, -0.7853, -0.7853, -0.7853
};

// Current testing state
int currentServo = -1;  // No servo being tested initially
int testStage = 0;      // 0=center, 1=min, 2=max, 3=center, 4=finished
bool testInProgress = false;
bool testAllJoints = false;  // Flag to indicate if we're testing all joints
unsigned long lastMoveTime = 0;

// ***** Function Prototypes *****
void printHelp();
void listServos();
void moveServoTo(int servo, int pulse);
void setAllServosToSafe();
void testJoint(int joint);
void runTestSequence();
void processCommand(String command);

// ***** Arduino setup() *****
void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }  // Wait until Serial is available
  Serial.println(F("Joint Test Utility"));
  Serial.println(F("------------------"));
  
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(PWM_FREQUENCY);
  delay(100);
  
  printHelp();
  
  // Move all servos to safe positions
  setAllServosToSafe();
  
  Serial.println(F("Ready! Enter a command..."));
}

// ***** moveServoTo() *****
// Commands the given servo to move to a specific pulse width
void moveServoTo(int servo, int pulse) {
  if (servo < 0 || servo >= SERVO_COUNT) return;
  if (pulse < 0) pulse = 0;
  if (pulse > 4095) pulse = 4095;
  pwm.setPWM(servo, 0, pulse);
}

// ***** setAllServosToSafe() *****
// Move all servos to their safe positions
void setAllServosToSafe() {
  Serial.println(F("Moving all servos to safe positions..."));
  
  for (int i = 0; i < SERVO_COUNT; i++) {
    // Convert safe position angle to pulse
    float anglePct = (safePos[i] - ANGLE_MIN[i]) / (ANGLE_MAX[i] - ANGLE_MIN[i]);
    if (anglePct < 0.0) anglePct = 0.0;
    if (anglePct > 1.0) anglePct = 1.0;
    float servoRange = servoMax[i] - servoMin[i];
    float pulse = (anglePct * servoRange) + servoMin[i];
    
    // Move the servo
    moveServoTo(i, pulse);
    delay(50);
  }
  
  Serial.println(F("All servos in safe position."));
}

// ***** testJoint() *****
// Start testing a specific joint
void testJoint(int joint) {
  if (joint < 0 || joint >= SERVO_COUNT) {
    Serial.println(F("Error: Invalid joint number."));
    return;
  }
  
  currentServo = joint;
  testStage = 0;
  testInProgress = true;
  lastMoveTime = millis();
  
  Serial.print(F("Starting test for "));
  Serial.print(JOINT_NAMES[joint]);
  Serial.print(F(" (Joint #"));
  Serial.print(joint);
  Serial.println(F(")"));
  Serial.println(F("Sequence: Center -> Min -> Max -> Center"));
}

// ***** runTestSequence() *****
// Advance the test sequence for the current joint
void runTestSequence() {
  if (!testInProgress || currentServo < 0) return;
  
  // Check if it's time to move to the next stage
  if (millis() - lastMoveTime < TEST_DELAY) return;
  
  // Move the servo based on the current test stage
  switch (testStage) {
    case 0:  // Center
      Serial.print(F("Moving to center position ("));
      Serial.print(servoCenter[currentServo]);
      Serial.println(F(")"));
      moveServoTo(currentServo, servoCenter[currentServo]);
      testStage = 1;
      break;
      
    case 1:  // Min
      Serial.print(F("Moving to minimum position ("));
      Serial.print(servoMin[currentServo]);
      Serial.println(F(")"));
      moveServoTo(currentServo, servoMin[currentServo]);
      testStage = 2;
      break;
      
    case 2:  // Max
      Serial.print(F("Moving to maximum position ("));
      Serial.print(servoMax[currentServo]);
      Serial.println(F(")"));
      moveServoTo(currentServo, servoMax[currentServo]);
      testStage = 3;
      break;
      
    case 3:  // Back to center
      Serial.print(F("Moving back to center position ("));
      Serial.print(servoCenter[currentServo]);
      Serial.println(F(")"));
      moveServoTo(currentServo, servoCenter[currentServo]);
      testStage = 4;
      break;
      
    case 4:  // Finished
      Serial.print(F("Test complete for "));
      Serial.println(JOINT_NAMES[currentServo]);
      testInProgress = false;
      currentServo = -1;
      break;
  }
  
  lastMoveTime = millis();
}

// ***** printHelp() *****
void printHelp() {
  Serial.println(F("Available Commands:"));
  Serial.println(F("  list    - List all joints with their numbers"));
  Serial.println(F("  test N  - Test joint number N (0-11)"));
  Serial.println(F("  all     - Test all joints in sequence"));
  Serial.println(F("  safe    - Move all joints to safe positions"));
  Serial.println(F("  stop    - Stop the current test"));
  Serial.println(F("  help    - Display this help message"));
}

// ***** listServos() *****
void listServos() {
  Serial.println(F("Joint Numbers:"));
  for (int i = 0; i < SERVO_COUNT; i++) {
    Serial.print(F("  "));
    Serial.print(i);
    Serial.print(F(": "));
    Serial.println(JOINT_NAMES[i]);
  }
}

// ***** processCommand() *****
void processCommand(String command) {
  command.trim();
  if (command.length() == 0) return;
  
  if (command == "help") {
    printHelp();
    return;
  }
  
  if (command == "list") {
    listServos();
    return;
  }
  
  if (command == "safe") {
    testInProgress = false;
    currentServo = -1;
    setAllServosToSafe();
    return;
  }
  
  if (command == "stop") {
    if (testInProgress) {
      Serial.println(F("Test stopped."));
      testInProgress = false;
      testAllJoints = false;
      currentServo = -1;
    } else {
      Serial.println(F("No test currently running."));
    }
    return;
  }
  
  if (command == "all") {
    if (testInProgress) {
      Serial.println(F("Error: A test is already in progress. Use 'stop' first."));
      return;
    }
    
    Serial.println(F("Testing all joints in sequence..."));
    // This will be implemented in the main loop
    currentServo = 0;
    testStage = 0;
    testInProgress = true;
    testAllJoints = true;
    lastMoveTime = millis();
    return;
  }
  
  if (command.startsWith("test ")) {
    if (testInProgress) {
      Serial.println(F("Error: A test is already in progress. Use 'stop' first."));
      return;
    }
    
    String jointStr = command.substring(5);
    int joint = jointStr.toInt();
    testJoint(joint);
    return;
  }
  
  Serial.println(F("Unknown command. Type 'help' for a list of commands."));
}

// ***** Arduino loop() *****
void loop() {
  // Process any incoming serial commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command.length() > 0) {
      processCommand(command);
    }
  }
  
  // Run the current test sequence
  if (testInProgress) {
    runTestSequence();
    
    // If testing all joints and current joint is done, move to next joint
    if (!testInProgress && testAllJoints && currentServo < SERVO_COUNT - 1) {
      currentServo++;
      testStage = 0;
      testInProgress = true;
      lastMoveTime = millis() + 1000;  // Add extra delay between joints
      Serial.println();
      Serial.print(F("Moving to next joint: "));
      Serial.println(JOINT_NAMES[currentServo]);
    }
  }
}