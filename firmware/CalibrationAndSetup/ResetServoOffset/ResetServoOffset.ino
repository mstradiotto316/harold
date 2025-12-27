/*  Reset Servo Offset - Clears calibration offset without affecting ID
 *
 *  This writes 0 to the offset registers (31-32), restoring the servo
 *  to its factory calibration state. The servo ID is NOT affected.
 *
 *  Usage:
 *    1. Flash this sketch
 *    2. Open Serial Monitor @ 115200
 *    3. Enter the servo ID to reset (1-12)
 *    4. Confirm the reset
 *    5. Power-cycle servo bus after reset
 */
#include <SCServo.h>
SMS_STS st;

#define S_RXD 18
#define S_TXD 19
const uint32_t BUS_BAUD = 1000000;

// Feetech STS/SMS servo EEPROM addresses
#define SMS_STS_OFS_L 31      // Offset low byte
#define SMS_STS_OFS_H 32      // Offset high byte
#define SMS_STS_ID 5          // Servo ID register

const char* JOINT_NAMES[] = {
  "",
  "FL Shoulder", "FL Thigh", "FL Calf",
  "FR Shoulder", "FR Thigh", "FR Calf",
  "BL Shoulder", "BL Thigh", "BL Calf",
  "BR Shoulder", "BR Thigh", "BR Calf"
};

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  Serial1.begin(BUS_BAUD, SERIAL_8N1, S_RXD, S_TXD);
  st.pSerial = &Serial1;
  delay(500);

  Serial.println(F("\n========================================"));
  Serial.println(F("   Servo Offset Reset Tool"));
  Serial.println(F("========================================"));
  Serial.println(F("This resets the calibration offset to 0."));
  Serial.println(F("The servo ID will NOT be changed.\n"));

  // Read current state of all servos
  Serial.println("Current servo positions and offsets:");
  for (uint8_t id = 1; id <= 12; ++id) {
    int pos = st.ReadPos(id);

    // Read current offset from EEPROM
    int ofsL = st.readByte(id, SMS_STS_OFS_L);
    int ofsH = st.readByte(id, SMS_STS_OFS_H);
    int16_t offset = 0;
    if (ofsL >= 0 && ofsH >= 0) {
      offset = (int16_t)((ofsH << 8) | ofsL);
    }

    if (pos < 0) {
      Serial.printf("  ID %2d (%s): NO RESPONSE\n", id, JOINT_NAMES[id]);
    } else {
      const char* flag = (abs(offset) > 50) ? " <-- HAS OFFSET" : "";
      Serial.printf("  ID %2d (%s): pos=%d, stored_offset=%d%s\n",
                    id, JOINT_NAMES[id], pos, offset, flag);
    }
  }

  Serial.println(F("\n----------------------------------------"));
  Serial.println(F("Enter servo ID to reset (1-12), or 'q' to quit:"));
}

void resetServoOffset(uint8_t id) {
  Serial.printf("\nResetting offset for servo ID %d (%s)...\n", id, JOINT_NAMES[id]);

  // Read current offset first
  int ofsL = st.readByte(id, SMS_STS_OFS_L);
  int ofsH = st.readByte(id, SMS_STS_OFS_H);
  int16_t oldOffset = 0;
  if (ofsL >= 0 && ofsH >= 0) {
    oldOffset = (int16_t)((ofsH << 8) | ofsL);
  }
  Serial.printf("  Current stored offset: %d\n", oldOffset);

  // Unlock EEPROM for writing
  st.unLockEprom(id);
  delay(10);

  // Write 0 to offset registers
  st.writeByte(id, SMS_STS_OFS_L, 0);
  delay(10);
  st.writeByte(id, SMS_STS_OFS_H, 0);
  delay(10);

  // Lock EEPROM again
  st.LockEprom(id);
  delay(10);

  // Verify the write
  ofsL = st.readByte(id, SMS_STS_OFS_L);
  ofsH = st.readByte(id, SMS_STS_OFS_H);
  int16_t newOffset = 0;
  if (ofsL >= 0 && ofsH >= 0) {
    newOffset = (int16_t)((ofsH << 8) | ofsL);
  }

  if (newOffset == 0) {
    Serial.println(F("  SUCCESS! Offset reset to 0."));
    Serial.printf("  Old offset: %d -> New offset: %d\n", oldOffset, newOffset);
  } else {
    Serial.printf("  WARNING: Offset is now %d (expected 0)\n", newOffset);
    Serial.println(F("  Try power-cycling the servo bus and running again."));
  }

  // Read new position
  delay(100);
  int pos = st.ReadPos(id);
  Serial.printf("  Current position reading: %d\n", pos);

  Serial.println(F("\n  IMPORTANT: Power-cycle the servo bus to apply changes!"));
}

void loop() {
  if (!Serial.available()) return;

  String input = Serial.readStringUntil('\n');
  input.trim();

  if (input == "q" || input == "Q") {
    Serial.println(F("Exiting. Remember to power-cycle servo bus!"));
    while (true) { delay(1000); }
  }

  int id = input.toInt();
  if (id < 1 || id > 12) {
    Serial.println(F("Invalid ID. Enter 1-12, or 'q' to quit:"));
    return;
  }

  // Confirm before reset
  Serial.printf("Reset offset for ID %d (%s)? Enter 'y' to confirm: ", id, JOINT_NAMES[id]);

  while (!Serial.available()) { delay(10); }
  String confirm = Serial.readStringUntil('\n');
  confirm.trim();

  if (confirm == "y" || confirm == "Y") {
    resetServoOffset(id);
  } else {
    Serial.println(F("Cancelled."));
  }

  Serial.println(F("\nEnter another servo ID (1-12), or 'q' to quit:"));
}
