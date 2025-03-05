#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ***** Configuration *****
#define SERVO_COUNT     12      // Total number of servos (4 legs × 3 joints per leg)
#define PWM_FREQUENCY   50      // 50 Hz for standard servos

// ***** Create the PWM Servo Driver Object *****
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Shoulders are on opposite sides of the robot, so their min / max values are mirrored
//int servoMin[SERVO_COUNT]    = {315, 305, 290, 360,   190, 375, 385, 215,   185, 375, 395, 185};
//int servoCenter[SERVO_COUNT] = {355, 265, 330, 320,   285, 280, 290, 310,   280, 280, 300, 280};
//int servoMax[SERVO_COUNT]    = {395, 225, 370, 280,   380, 185, 195, 405,   375, 185, 205, 375};

int servoMin[SERVO_COUNT]    = {315, 305, 290, 360,   380, 185, 385, 215,   375, 185, 395, 185};
int servoCenter[SERVO_COUNT] = {355, 265, 330, 320,   285, 280, 290, 310,   280, 280, 300, 280};
int servoMax[SERVO_COUNT]    = {395, 225, 370, 280,   190, 375, 195, 405,   185, 375, 205, 375};

// ***** Function Prototypes *****
void printHelp();
void listBoundaries();
void setBoundary(int servo, String boundary, int value);
void moveServoTo(int servo, int pulse);
void testServo(int servo, String pos);
void cycleServo(int servo);
void processCommand(String command);

// ***** printHelp() *****
// Displays the available commands.
void printHelp() {
  Serial.println(F("Servo Tuning Utility Commands:"));
  Serial.println(F("  list"));
  Serial.println(F("      -> List current servo boundaries for all servos."));
  Serial.println(F("  set <servo> <min/center/max> <value>"));
  Serial.println(F("      -> Set the boundary value for the specified servo."));
  Serial.println(F("         Example: set 3 min 235"));
  Serial.println(F("  test <servo> [min/center/max]"));
  Serial.println(F("      -> Move the specified servo to the indicated position."));
  Serial.println(F("         If position is omitted, center is used."));
  Serial.println(F("  cycle <servo>"));
  Serial.println(F("      -> Cycle the servo through min, center, and max positions."));
  Serial.println(F("  help"));
  Serial.println(F("      -> Display this help message."));
  Serial.println();
}

// ***** listBoundaries() *****
// Prints the current min, center, and max pulse values for each servo.
void listBoundaries() {
  Serial.println(F("Current Servo Boundaries:"));
  for (int i = 0; i < SERVO_COUNT; i++) {
    Serial.print(F("Servo "));
    Serial.print(i);
    Serial.print(F(": Min = "));
    Serial.print(servoMin[i]);
    Serial.print(F("   Center = "));
    Serial.print(servoCenter[i]);
    Serial.print(F("   Max = "));
    Serial.println(servoMax[i]);
  }
  Serial.println();
}

// ***** setBoundary() *****
// Sets a new value for a given servo?s min, center, or max.
void setBoundary(int servo, String boundary, int value) {
  if (servo < 0 || servo >= SERVO_COUNT) {
    Serial.println(F("Error: Invalid servo number."));
    return;
  }
  boundary.toLowerCase();
  boundary.trim();  // Remove extra whitespace
  if (boundary == "min") {
    servoMin[servo] = value;
    Serial.print(F("Servo "));
    Serial.print(servo);
    Serial.print(F(" min set to "));
    Serial.println(value);
  } else if (boundary == "center") {
    servoCenter[servo] = value;
    Serial.print(F("Servo "));
    Serial.print(servo);
    Serial.print(F(" center set to "));
    Serial.println(value);
  } else if (boundary == "max") {
    servoMax[servo] = value;
    Serial.print(F("Servo "));
    Serial.print(servo);
    Serial.print(F(" max set to "));
    Serial.println(value);
  } else {
    Serial.println(F("Error: Boundary must be one of: min, center, max."));
  }
  Serial.println();
}

// ***** moveServoTo() *****
// Commands the given servo to move to a specific pulse width.
void moveServoTo(int servo, int pulse) {
  if (servo < 0 || servo >= SERVO_COUNT) return;
  if (pulse < 0) pulse = 0;
  if (pulse > 4095) pulse = 4095;
  pwm.setPWM(servo, 0, pulse);
  Serial.print(F("Servo "));
  Serial.print(servo);
  Serial.print(F(" moved to pulse "));
  Serial.println(pulse);
}

// ***** testServo() *****
// Moves the specified servo to its min, center, or max pulse.
void testServo(int servo, String pos) {
  if (servo < 0 || servo >= SERVO_COUNT) {
    Serial.println(F("Error: Invalid servo number."));
    return;
  }
  pos.toLowerCase();
  pos.trim();  // Remove any extra whitespace
  if (pos == "" || pos == "center") {
    moveServoTo(servo, servoCenter[servo]);
  } else if (pos == "min") {
    moveServoTo(servo, servoMin[servo]);
  } else if (pos == "max") {
    moveServoTo(servo, servoMax[servo]);
  } else {
    Serial.println(F("Error: Invalid position. Use min, center, or max."));
  }
}

// ***** cycleServo() *****
// Cycles the specified servo through its min, center, and max positions.
void cycleServo(int servo) {
  if (servo < 0 || servo >= SERVO_COUNT) {
    Serial.println(F("Error: Invalid servo number."));
    return;
  }
  Serial.print(F("Cycling servo "));
  Serial.println(servo);
  moveServoTo(servo, servoMin[servo]);
  delay(1000);
  moveServoTo(servo, servoCenter[servo]);
  delay(1000);
  moveServoTo(servo, servoMax[servo]);
  delay(1000);
  moveServoTo(servo, servoCenter[servo]);
  Serial.println();
}

// ***** processCommand() *****
// Parses a command string (received over Serial) and executes it.
void processCommand(String command) {
  command.trim();
  if (command.length() == 0) return;
  
  int firstSpace = command.indexOf(' ');
  String cmd;
  if (firstSpace == -1) {
    cmd = command;
  } else {
    cmd = command.substring(0, firstSpace);
  }
  cmd.toLowerCase();
  
  if (cmd == "list") {
    listBoundaries();
  }
  else if (cmd == "help") {
    printHelp();
  }
  else if (cmd == "set") {
    int secondSpace = command.indexOf(' ', firstSpace + 1);
    int thirdSpace  = command.indexOf(' ', secondSpace + 1);
    if (firstSpace == -1 || secondSpace == -1 || thirdSpace == -1) {
      Serial.println(F("Invalid command format. Use: set <servo> <min/center/max> <value>"));
      return;
    }
    String servoToken = command.substring(firstSpace + 1, secondSpace);
    int servoIndex = servoToken.toInt();
    String boundary = command.substring(secondSpace + 1, thirdSpace);
    String valueToken = command.substring(thirdSpace + 1);
    int value = valueToken.toInt();
    setBoundary(servoIndex, boundary, value);
  }
  else if (cmd == "test") {
    int secondSpace = command.indexOf(' ', firstSpace + 1);
    int servoIndex = -1;
    String pos = "";
    if (secondSpace == -1) {
      String servoToken = command.substring(firstSpace + 1);
      servoIndex = servoToken.toInt();
      pos = "center";
    } else {
      String servoToken = command.substring(firstSpace + 1, secondSpace);
      servoIndex = servoToken.toInt();
      pos = command.substring(secondSpace + 1);
    }
    testServo(servoIndex, pos);
  }
  else if (cmd == "cycle") {
    String servoToken = command.substring(firstSpace + 1);
    int servoIndex = servoToken.toInt();
    cycleServo(servoIndex);
  }
  else {
    Serial.println(F("Unknown command. Type 'help' for a list of commands."));
  }
}

// ***** Arduino setup() *****
// Initializes the Serial port, PWM driver, and moves all servos to their center.
void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }  // Wait until Serial is available
  Serial.println(F("Servo Tuning Utility"));
  printHelp();
  
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(PWM_FREQUENCY);
  delay(10);
  
  // Initially move all servos to their center positions.
  for (int i = 0; i < SERVO_COUNT; i++) {
    moveServoTo(i, servoCenter[i]);
  }
}

// ***** Arduino loop() *****
// Reads an entire line from Serial and processes it.
void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command.length() > 0) {
      processCommand(command);
    }
  }
}
