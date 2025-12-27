/*  Harold mid-point calibrator - line-oriented for Arduino IDE
 *  Keys (type + Enter):
 *    a/d : -/+100   s/w : -/+10
 *    p   : save offset & re-centre
 *    r   : read current position (diagnostic)
 *    n   : next servo   q : quit
 *    1-12: jump to specific servo ID
 *
 *  Updated 2025-12-26: Added position readback, jump-to-servo, center all at startup
 */
#include <SCServo.h>
SMS_STS st;

#define S_RXD 18
#define S_TXD 19
const uint32_t BUS_BAUD = 1000000;

const uint8_t SERVO_IDS[] = { 1,2,3,4,5,6, 7,8,9,10,11,12 };
const uint8_t NUM_SERVOS  = sizeof(SERVO_IDS);

// Joint names for easier identification
const char* JOINT_NAMES[] = {
  "",  // ID 0 unused
  "FL Shoulder", "FL Thigh", "FL Calf",     // IDs 1-3
  "FR Shoulder", "FR Thigh", "FR Calf",     // IDs 4-6
  "BL Shoulder", "BL Thigh", "BL Calf",     // IDs 7-9
  "BR Shoulder", "BR Thigh", "BR Calf"      // IDs 10-12
};

const int SPEED = 800, ACC = 100, MID = 2047;

void center(uint8_t id) {
  st.WritePosEx(id, MID, SPEED, ACC);
}

// Center all servos to position 2047
void centerAll() {
  Serial.println("Centering ALL servos to position 2047...");
  for (uint8_t id = 1; id <= 12; ++id) {
    st.WritePosEx(id, MID, SPEED, ACC);
  }
  delay(1000);  // Wait for all servos to reach position
  Serial.println("All servos centered.\n");
}

// Read and display current servo position
void readPosition(uint8_t id) {
  int pos = st.ReadPos(id);
  if (pos < 0) {
    Serial.printf("  ID %2d: Read FAILED\n", id);
  } else {
    int offset = pos - MID;
    const char* status = "";
    if (abs(offset) > 100) status = " <-- NEEDS CALIBRATION";
    else if (abs(offset) > 50) status = " <-- slight offset";
    Serial.printf("  ID %2d (%s): pos=%d (offset: %+d)%s\n",
                  id, JOINT_NAMES[id], pos, offset, status);
  }
}

// Read all servo positions
void readAllPositions() {
  Serial.println("Current positions of all servos:");
  for (uint8_t id = 1; id <= 12; ++id) {
    readPosition(id);
  }
  Serial.println();
}

// Returns -1 if not a number, otherwise the servo ID
int parseServoId(const String& cmd) {
  if (cmd.length() == 0) return -1;
  // Check if it's a number
  for (unsigned int i = 0; i < cmd.length(); i++) {
    if (!isDigit(cmd.charAt(i))) return -1;
  }
  int id = cmd.toInt();
  if (id >= 1 && id <= 12) return id;
  return -1;
}

// Returns the index in SERVO_IDS for the given servo ID, or -1 if not found
int findServoIndex(uint8_t id) {
  for (uint8_t i = 0; i < NUM_SERVOS; ++i) {
    if (SERVO_IDS[i] == id) return i;
  }
  return -1;
}

// Collision-safe operating ranges based on actual robot geometry
// These are MUCH more conservative than servo mechanical limits
void getSafeRange(uint8_t id, int* minPos, int* maxPos) {
  // Units: 4096/360 = 11.38 units per degree
  const float U = 4096.0f / 360.0f;

  if (id == 1 || id == 4 || id == 7 || id == 10) {
    // Shoulders: ±20 degrees (conservative, actual limit is ±30)
    *minPos = MID - (int)(20 * U);
    *maxPos = MID + (int)(20 * U);
  }
  else if (id == 2 || id == 5 || id == 8 || id == 11) {
    // Thighs: -60 to +10 degrees (forward lean only, backward hits body)
    // Based on actual usage: -52° to -10° in walking/pushup
    *minPos = MID - (int)(60 * U);  // Forward (negative direction)
    *maxPos = MID + (int)(10 * U);  // Slight backward only
  }
  else {
    // Calves (id 3,6,9,12): 0 to +90 degrees (bent only, extension hits ground/body)
    // Based on actual usage: +20° to +89° in walking/pushup
    *minPos = MID - (int)(10 * U);  // Slight extension only
    *maxPos = MID + (int)(80 * U);  // Full bend OK
  }
}

// Incremental range test - moves in small steps with confirmation
void testRange(uint8_t id) {
  int minPos, maxPos;
  getSafeRange(id, &minPos, &maxPos);

  const float U = 4096.0f / 360.0f;
  float minDeg = (minPos - MID) / U;
  float maxDeg = (maxPos - MID) / U;

  Serial.printf("  Safe range for %s: %.0f to %.0f degrees\n", JOINT_NAMES[id], minDeg, maxDeg);
  Serial.println("  This test moves in 10-degree steps.");
  Serial.println("  Press Enter to continue each step, 'x' to abort anytime.\n");

  // Start at center
  int currentPos = MID;
  st.WritePosEx(id, currentPos, SPEED, ACC);
  delay(500);
  Serial.println("  At CENTER (0 degrees)");

  int step = (int)(10 * U);  // 10 degrees per step

  // Test toward minimum (usually forward/negative)
  Serial.println("\n  --- Testing toward MINIMUM ---");
  while (currentPos - step >= minPos) {
    Serial.printf("  Move to %+.0f degrees? [Enter=yes, x=abort]: ", (currentPos - step - MID) / U);
    while (!Serial.available()) { delay(10); }
    String input = Serial.readStringUntil('\n');
    if (input.indexOf('x') >= 0 || input.indexOf('X') >= 0) {
      Serial.println("  ABORTED");
      st.WritePosEx(id, MID, SPEED, ACC);
      return;
    }
    currentPos -= step;
    st.WritePosEx(id, currentPos, SPEED, ACC);
    delay(400);
    Serial.println("  OK");
  }
  Serial.printf("  Reached safe minimum: %.0f degrees\n", (currentPos - MID) / U);

  // Return to center
  Serial.println("\n  Returning to center...");
  currentPos = MID;
  st.WritePosEx(id, currentPos, SPEED, ACC);
  delay(500);

  // Test toward maximum (usually backward/positive for thighs)
  Serial.println("\n  --- Testing toward MAXIMUM ---");
  while (currentPos + step <= maxPos) {
    Serial.printf("  Move to %+.0f degrees? [Enter=yes, x=abort]: ", (currentPos + step - MID) / U);
    while (!Serial.available()) { delay(10); }
    String input = Serial.readStringUntil('\n');
    if (input.indexOf('x') >= 0 || input.indexOf('X') >= 0) {
      Serial.println("  ABORTED");
      st.WritePosEx(id, MID, SPEED, ACC);
      return;
    }
    currentPos += step;
    st.WritePosEx(id, currentPos, SPEED, ACC);
    delay(400);
    Serial.println("  OK");
  }
  Serial.printf("  Reached safe maximum: %.0f degrees\n", (currentPos - MID) / U);

  // Return to center
  Serial.println("\n  Returning to center...");
  st.WritePosEx(id, MID, SPEED, ACC);
  delay(500);

  Serial.println("\n  Range test COMPLETE");
  Serial.printf("  Verified range: %.0f to %.0f degrees\n", minDeg, maxDeg);
}

// Calibrate a single servo. Returns next servo ID to calibrate, or -1 to quit, or 0 to continue sequentially
int calibrateServo(uint8_t id) {
  Serial.printf("\n=== Servo ID %d (%s) ===\n", id, JOINT_NAMES[id]);
  Serial.println("Commands: a/d=+/-100  s/w=+/-10  p=save  r=read  t=test range  n=next  q=quit  1-12=jump");

  // Read current position before centering
  Serial.println("Current position:");
  readPosition(id);

  center(id);
  delay(400);
  Serial.println("Centered to 2047. Adjust until joint is at mechanical neutral.");

  int pos = MID;
  while (true) {
    if (!Serial.available()) continue;
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    // Check if it's a servo ID jump command
    int jumpId = parseServoId(cmd);
    if (jumpId > 0) {
      Serial.printf("Jumping to servo ID %d...\n", jumpId);
      return jumpId;
    }

    if      (cmd == "a") pos -= 100;
    else if (cmd == "d") pos += 100;
    else if (cmd == "s") pos -= 10;
    else if (cmd == "w") pos += 10;
    else if (cmd == "r") {
      readPosition(id);
      continue;
    }
    else if (cmd == "t") {
      Serial.println("\n  === RANGE TEST ===");
      Serial.println("  This will move the joint to its limits.");
      Serial.println("  Watch carefully for collisions!");
      testRange(id);
      Serial.printf("  Back to calibrating ID %d. pos=%d\n", id, pos);
      continue;
    }
    else if (cmd == "p") {
      Serial.printf("  Saving offset at pos=%d (offset=%+d)...\n", pos, pos - MID);
      st.CalibrationOfs(id);
      Serial.println("  Offset SAVED to servo EEPROM!");
      Serial.println("  Re-centering to 2047 (should now be at mechanical neutral)...");
      center(id);
      delay(400);
      readPosition(id);
      Serial.println("\n  >>> Recommend: press 't' to test range after calibration <<<");
      continue;
    }
    else if (cmd == "n") { return 0; }  // Continue to next sequential servo
    else if (cmd == "q") { return -1; } // Quit
    else { Serial.println(F("Unknown command. Use: a/d/s/w/p/r/t/n/q or 1-12")); continue; }

    pos = constrain(pos, MID - 1024, MID + 1024);  // ±90 degrees range for calibration
    st.WritePosEx(id, pos, SPEED, ACC);
    Serial.printf("  pos = %d (offset = %+d)\n", pos, pos - MID);
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  Serial1.begin(BUS_BAUD, SERIAL_8N1, S_RXD, S_TXD);
  st.pSerial = &Serial1;
  delay(500);

  Serial.println(F("\n========================================"));
  Serial.println(F("     Harold Mid-point Calibrator"));
  Serial.println(F("========================================"));
  Serial.println(F("This tool calibrates servo zero positions.\n"));

  // Center all servos first
  centerAll();

  // Show current state of all servos
  readAllPositions();

  Serial.println("Enter servo ID (1-12) to calibrate, or press Enter to start from ID 1:");
  Serial.println("(waiting 10 seconds...)");

  // Wait for input with timeout
  unsigned long startTime = millis();
  while (!Serial.available() && (millis() - startTime < 10000)) {
    delay(10);
  }

  int currentIdx = 0;  // Start from first servo by default

  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    int id = parseServoId(input);
    if (id > 0) {
      currentIdx = findServoIndex(id);
      if (currentIdx < 0) currentIdx = 0;
      Serial.printf("Starting from servo ID %d\n", SERVO_IDS[currentIdx]);
    }
  } else {
    Serial.println("No input, starting from ID 1...");
  }

  // Main calibration loop
  while (currentIdx >= 0 && currentIdx < NUM_SERVOS) {
    int result = calibrateServo(SERVO_IDS[currentIdx]);

    if (result == -1) {
      // Quit requested
      Serial.println(F("\nExiting calibration."));
      break;
    } else if (result == 0) {
      // Next sequential servo
      currentIdx++;
    } else {
      // Jump to specific servo ID
      int newIdx = findServoIndex(result);
      if (newIdx >= 0) {
        currentIdx = newIdx;
      } else {
        currentIdx++;  // Fallback to next
      }
    }
  }

  if (currentIdx >= NUM_SERVOS) {
    Serial.println(F("\nAll servos trimmed!"));
  }

  Serial.println(F("\n========================================"));
  Serial.println(F("IMPORTANT: Power-cycle the servo bus to ensure offsets are stored."));
  Serial.println(F("Then test with PushupTestWithFeedback to verify calibration."));
  Serial.println(F("========================================"));

  // Final position readout
  Serial.println(F("\nFinal servo positions:"));
  readAllPositions();
}

void loop() {}
