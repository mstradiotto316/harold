/* Read servo calibration offsets from EEPROM
 * Diagnostic tool to check if offsets are being corrupted
 */
#include <SCServo.h>
SMS_STS st;

#define S_RXD 18
#define S_TXD 19
const uint32_t BUS_BAUD = 1000000;

// Offset registers in EEPROM
#define SMS_STS_OFS_L 31
#define SMS_STS_OFS_H 32

const char* JOINT_NAMES[] = {
  "",
  "FL Shoulder", "FL Thigh", "FL Calf",
  "FR Shoulder", "FR Thigh", "FR Calf",
  "BL Shoulder", "BL Thigh", "BL Calf",
  "BR Shoulder", "BR Thigh", "BR Calf"
};

void readOffset(uint8_t id) {
  // Read the 2-byte offset from EEPROM registers 31-32
  uint8_t buf[2];
  int result = st.Read(id, SMS_STS_OFS_L, buf, 2);

  if (result != 2) {
    Serial.printf("ID %2d (%s): READ FAILED\n", id, JOINT_NAMES[id]);
    return;
  }

  // Combine low and high bytes (little-endian for STS series)
  int16_t offset = buf[0] | (buf[1] << 8);

  // Convert to degrees
  float offset_deg = offset * 360.0f / 4096.0f;

  const char* flag = "";
  if (abs(offset) > 200) flag = " <-- LARGE OFFSET";
  if (id == 2 && abs(offset) > 100) flag = " <-- CHECK THIS (FL Thigh)";

  Serial.printf("ID %2d (%s): offset=%+5d units (%+6.1f deg)%s\n",
                id, JOINT_NAMES[id], offset, offset_deg, flag);
}

void readAllOffsets() {
  Serial.println("\n=== Servo Calibration Offsets (EEPROM registers 31-32) ===\n");
  for (uint8_t id = 1; id <= 12; ++id) {
    readOffset(id);
  }
  Serial.println();
}

void readCurrentPositions() {
  Serial.println("=== Current Positions ===\n");
  for (uint8_t id = 1; id <= 12; ++id) {
    int pos = st.ReadPos(id);
    if (pos < 0) {
      Serial.printf("ID %2d: READ FAILED\n", id);
    } else {
      Serial.printf("ID %2d (%s): pos=%4d (vs mid=2047, diff=%+4d)\n",
                    id, JOINT_NAMES[id], pos, pos - 2047);
    }
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  Serial1.begin(BUS_BAUD, SERIAL_8N1, S_RXD, S_TXD);
  st.pSerial = &Serial1;
  delay(500);

  Serial.println(F("\n========================================"));
  Serial.println(F("    Servo Offset Diagnostic Tool"));
  Serial.println(F("========================================\n"));

  readAllOffsets();
  readCurrentPositions();

  Serial.println("Commands: 'o' = read offsets, 'p' = read positions, 'c' = center all");
  Serial.println("          '2' = detailed servo 2 info");
}

void detailedServo2() {
  Serial.println("\n=== Detailed Servo 2 (FL Thigh) Analysis ===\n");

  // Read offset
  uint8_t buf[2];
  int result = st.Read(2, SMS_STS_OFS_L, buf, 2);
  if (result == 2) {
    int16_t offset = buf[0] | (buf[1] << 8);
    Serial.printf("EEPROM Offset (reg 31-32): %+d units (%+.1f deg)\n",
                  offset, offset * 360.0f / 4096.0f);
  }

  // Read current position
  int pos = st.ReadPos(2);
  Serial.printf("Current Position: %d (diff from 2047: %+d)\n", pos, pos - 2047);

  // Read torque enable register (40) to check its value
  int torque = st.readByte(2, 40);
  Serial.printf("Torque Enable (reg 40): %d", torque);
  if (torque == 128) Serial.print(" <-- CALIBRATION MODE ACTIVE!");
  else if (torque == 0) Serial.print(" (torque disabled)");
  else if (torque == 1) Serial.print(" (torque enabled)");
  Serial.println();

  // Read mode register (33)
  int mode = st.readByte(2, 33);
  Serial.printf("Mode (reg 33): %d\n", mode);

  // Read min/max angle limits
  int minAngle = st.readWord(2, 9);
  int maxAngle = st.readWord(2, 11);
  Serial.printf("Angle Limits: min=%d, max=%d\n", minAngle, maxAngle);

  Serial.println();
}

void centerAll() {
  Serial.println("Centering all servos to 2047...");
  for (uint8_t id = 1; id <= 12; ++id) {
    st.WritePosEx(id, 2047, 800, 100);
  }
  delay(1000);
  Serial.println("Done. Reading positions:");
  readCurrentPositions();
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'o') readAllOffsets();
    else if (c == 'p') readCurrentPositions();
    else if (c == 'c') centerAll();
    else if (c == '2') detailedServo2();
  }
}
