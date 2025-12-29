/*
 * Harold Servo Tuning Utility
 *
 * Allows reading and adjusting servo parameters that affect responsiveness:
 * - Dead zone (deadband) - reduces the "play" before servo responds
 * - Torque limit - already at 100% in streaming firmware
 * - Calibration offset - for position drift issues
 *
 * Commands (via Serial at 115200):
 *   s          - Scan and show all servos
 *   r <id>     - Read all tuning parameters for servo ID
 *   d <id> <v> - Set dead zone (both CW and CCW) to value v (0-10, 0=tightest)
 *   t <id> <v> - Set torque limit (0-1000, 1000=100%)
 *   c <id>     - Calibrate current position as center (saves to EEPROM)
 *   o <id>     - Read current offset
 *   p <id> <pos> - Move servo to position (for testing)
 *   m          - Monitor all servo positions continuously
 */

#include <SCServo.h>

SMS_STS st;

// Pin definitions (must match HaroldStreamingControl)
#define S_RXD 18
#define S_TXD 19
const uint32_t BUS_BAUD = 1000000;

// Register addresses (from SMS_STS.h)
#define SMS_STS_CW_DEAD 26
#define SMS_STS_CCW_DEAD 27
#define SMS_STS_OFS_L 31
#define SMS_STS_OFS_H 32
#define SMS_STS_TORQUE_LIMIT_L 48
#define SMS_STS_TORQUE_LIMIT_H 49
#define SMS_STS_LOCK 55

const int SERVO_IDS[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
const int NUM_SERVOS = 12;

const char* JOINT_NAMES[] = {
  "", // ID 0 unused
  "FL_shoulder", "FL_thigh", "FL_calf",
  "FR_shoulder", "FR_thigh", "FR_calf",
  "BL_shoulder", "BL_thigh", "BL_calf",
  "BR_shoulder", "BR_thigh", "BR_calf"
};

void setup() {
  Serial.begin(115200);
  Serial1.begin(BUS_BAUD, SERIAL_8N1, S_RXD, S_TXD);
  st.pSerial = &Serial1;

  delay(1000);

  Serial.println(F("\n========================================"));
  Serial.println(F("   Harold Servo Tuning Utility"));
  Serial.println(F("========================================\n"));
  Serial.println(F("Commands:"));
  Serial.println(F("  s          - Scan all servos"));
  Serial.println(F("  r <id>     - Read parameters for servo"));
  Serial.println(F("  d <id> <v> - Set dead zone (0-10, lower=tighter)"));
  Serial.println(F("  t <id> <v> - Set torque limit (0-1000)"));
  Serial.println(F("  c <id>     - Calibrate center position"));
  Serial.println(F("  o <id>     - Read offset"));
  Serial.println(F("  p <id> <pos> - Move to position"));
  Serial.println(F("  m          - Monitor positions"));
  Serial.println(F("  a <v>      - Set dead zone for ALL servos"));
  Serial.println();
}

void scanServos() {
  Serial.println(F("\nScanning servos...\n"));
  Serial.println(F("ID   Name           Pos    DeadCW  DeadCCW  Torque  Offset"));
  Serial.println(F("---  -------------  -----  ------  -------  ------  ------"));

  for (int i = 0; i < NUM_SERVOS; i++) {
    int id = SERVO_IDS[i];
    int ping = st.Ping(id);

    if (ping != -1) {
      int pos = st.ReadPos(id);
      int deadCW = st.readByte(id, SMS_STS_CW_DEAD);
      int deadCCW = st.readByte(id, SMS_STS_CCW_DEAD);
      int torqueL = st.readByte(id, SMS_STS_TORQUE_LIMIT_L);
      int torqueH = st.readByte(id, SMS_STS_TORQUE_LIMIT_H);
      int torque = (torqueH << 8) | torqueL;
      int ofsL = st.readByte(id, SMS_STS_OFS_L);
      int ofsH = st.readByte(id, SMS_STS_OFS_H);
      int16_t offset = (int16_t)((ofsH << 8) | ofsL);

      Serial.printf("%2d   %-13s  %5d  %6d  %7d  %6d  %6d\n",
                    id, JOINT_NAMES[id], pos, deadCW, deadCCW, torque, offset);
    } else {
      Serial.printf("%2d   %-13s  -- NOT FOUND --\n", id, JOINT_NAMES[id]);
    }
    delay(10);
  }
  Serial.println();
}

void readServoParams(int id) {
  int ping = st.Ping(id);
  if (ping == -1) {
    Serial.printf("Servo %d not found!\n", id);
    return;
  }

  Serial.printf("\n=== Servo %d (%s) Parameters ===\n", id, JOINT_NAMES[id]);

  int pos = st.ReadPos(id);
  int speed = st.ReadSpeed(id);
  int load = st.ReadLoad(id);
  int voltage = st.ReadVoltage(id);
  int temp = st.ReadTemper(id);
  int current = st.ReadCurrent(id);

  int deadCW = st.readByte(id, SMS_STS_CW_DEAD);
  int deadCCW = st.readByte(id, SMS_STS_CCW_DEAD);
  int torqueL = st.readByte(id, SMS_STS_TORQUE_LIMIT_L);
  int torqueH = st.readByte(id, SMS_STS_TORQUE_LIMIT_H);
  int torque = (torqueH << 8) | torqueL;
  int ofsL = st.readByte(id, SMS_STS_OFS_L);
  int ofsH = st.readByte(id, SMS_STS_OFS_H);
  int16_t offset = (int16_t)((ofsH << 8) | ofsL);

  Serial.printf("Position:      %d (%.2f deg from center)\n", pos, (pos - 2047) * 360.0 / 4096);
  Serial.printf("Speed:         %d\n", speed);
  Serial.printf("Load:          %d (%.1f%%)\n", load, load / 10.0);
  Serial.printf("Voltage:       %d (%.1fV)\n", voltage, voltage / 10.0);
  Serial.printf("Temperature:   %d C\n", temp);
  Serial.printf("Current:       %d mA\n", current);
  Serial.println();
  Serial.printf("Dead Zone CW:  %d steps (%.3f deg)\n", deadCW, deadCW * 360.0 / 4096);
  Serial.printf("Dead Zone CCW: %d steps (%.3f deg)\n", deadCCW, deadCCW * 360.0 / 4096);
  Serial.printf("Torque Limit:  %d/1000 (%.1f%%)\n", torque, torque / 10.0);
  Serial.printf("Offset:        %d steps\n", offset);
  Serial.println();
}

void setDeadZone(int id, int value) {
  if (value < 0 || value > 10) {
    Serial.println("Dead zone must be 0-10");
    return;
  }

  int ping = st.Ping(id);
  if (ping == -1) {
    Serial.printf("Servo %d not found!\n", id);
    return;
  }

  // Unlock EPROM for writing
  st.unLockEprom(id);
  delay(10);

  // Write both CW and CCW dead zones
  st.writeByte(id, SMS_STS_CW_DEAD, value);
  delay(10);
  st.writeByte(id, SMS_STS_CCW_DEAD, value);
  delay(10);

  // Lock EPROM
  st.LockEprom(id);
  delay(10);

  // Verify
  int newCW = st.readByte(id, SMS_STS_CW_DEAD);
  int newCCW = st.readByte(id, SMS_STS_CCW_DEAD);

  Serial.printf("Servo %d dead zone set to %d (CW=%d, CCW=%d)\n", id, value, newCW, newCCW);

  if (value == 0) {
    Serial.println("WARNING: Dead zone 0 may cause servo buzzing at rest!");
  }
}

void setAllDeadZones(int value) {
  Serial.printf("\nSetting dead zone to %d for all servos...\n", value);
  for (int i = 0; i < NUM_SERVOS; i++) {
    int id = SERVO_IDS[i];
    if (st.Ping(id) != -1) {
      setDeadZone(id, value);
    }
    delay(50);
  }
  Serial.println("Done!");
}

void setTorqueLimit(int id, int value) {
  if (value < 0 || value > 1000) {
    Serial.println("Torque limit must be 0-1000");
    return;
  }

  int ping = st.Ping(id);
  if (ping == -1) {
    Serial.printf("Servo %d not found!\n", id);
    return;
  }

  // Torque limit is in SRAM, no EPROM unlock needed
  st.writeByte(id, SMS_STS_TORQUE_LIMIT_L, value & 0xFF);
  delay(5);
  st.writeByte(id, SMS_STS_TORQUE_LIMIT_H, (value >> 8) & 0xFF);
  delay(5);

  // Verify
  int readL = st.readByte(id, SMS_STS_TORQUE_LIMIT_L);
  int readH = st.readByte(id, SMS_STS_TORQUE_LIMIT_H);
  int readVal = (readH << 8) | readL;

  Serial.printf("Servo %d torque limit set to %d/1000 (%.1f%%)\n", id, readVal, readVal / 10.0);
}

void calibrateCenter(int id) {
  int ping = st.Ping(id);
  if (ping == -1) {
    Serial.printf("Servo %d not found!\n", id);
    return;
  }

  int pos = st.ReadPos(id);
  Serial.printf("Servo %d current position: %d\n", id, pos);
  Serial.println("Calibrating current position as center...");

  st.CalibrationOfs(id);
  delay(100);

  // Read new offset
  int ofsL = st.readByte(id, SMS_STS_OFS_L);
  int ofsH = st.readByte(id, SMS_STS_OFS_H);
  int16_t offset = (int16_t)((ofsH << 8) | ofsL);

  int newPos = st.ReadPos(id);
  Serial.printf("Calibration complete! New offset: %d, New position: %d\n", offset, newPos);
  Serial.println("NOTE: Power-cycle the servo to ensure EEPROM write is complete.");
}

void readOffset(int id) {
  int ping = st.Ping(id);
  if (ping == -1) {
    Serial.printf("Servo %d not found!\n", id);
    return;
  }

  int ofsL = st.readByte(id, SMS_STS_OFS_L);
  int ofsH = st.readByte(id, SMS_STS_OFS_H);
  int16_t offset = (int16_t)((ofsH << 8) | ofsL);

  Serial.printf("Servo %d offset: %d steps (%.2f deg)\n", id, offset, offset * 360.0 / 4096);
}

void moveToPosition(int id, int pos) {
  int ping = st.Ping(id);
  if (ping == -1) {
    Serial.printf("Servo %d not found!\n", id);
    return;
  }

  if (pos < 0 || pos > 4095) {
    Serial.println("Position must be 0-4095");
    return;
  }

  Serial.printf("Moving servo %d to position %d...\n", id, pos);
  st.WritePosEx(id, pos, 1500, 50);
}

void monitorPositions() {
  Serial.println("\nMonitoring servo positions (press any key to stop)...\n");

  while (!Serial.available()) {
    Serial.print("Pos: ");
    for (int i = 0; i < NUM_SERVOS; i++) {
      int id = SERVO_IDS[i];
      int pos = st.ReadPos(id);
      if (pos != -1) {
        Serial.printf("%4d ", pos);
      } else {
        Serial.print("---- ");
      }
    }
    Serial.println();
    delay(100);
  }

  // Clear input buffer
  while (Serial.available()) Serial.read();
  Serial.println("\nMonitoring stopped.");
}

void processCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  char c = cmd.charAt(0);

  if (c == 's') {
    scanServos();
  }
  else if (c == 'r') {
    int id = cmd.substring(2).toInt();
    readServoParams(id);
  }
  else if (c == 'd') {
    int space1 = cmd.indexOf(' ');
    int space2 = cmd.indexOf(' ', space1 + 1);
    int id = cmd.substring(space1 + 1, space2).toInt();
    int val = cmd.substring(space2 + 1).toInt();
    setDeadZone(id, val);
  }
  else if (c == 't') {
    int space1 = cmd.indexOf(' ');
    int space2 = cmd.indexOf(' ', space1 + 1);
    int id = cmd.substring(space1 + 1, space2).toInt();
    int val = cmd.substring(space2 + 1).toInt();
    setTorqueLimit(id, val);
  }
  else if (c == 'c') {
    int id = cmd.substring(2).toInt();
    calibrateCenter(id);
  }
  else if (c == 'o') {
    int id = cmd.substring(2).toInt();
    readOffset(id);
  }
  else if (c == 'p') {
    int space1 = cmd.indexOf(' ');
    int space2 = cmd.indexOf(' ', space1 + 1);
    int id = cmd.substring(space1 + 1, space2).toInt();
    int pos = cmd.substring(space2 + 1).toInt();
    moveToPosition(id, pos);
  }
  else if (c == 'm') {
    monitorPositions();
  }
  else if (c == 'a') {
    int val = cmd.substring(2).toInt();
    setAllDeadZones(val);
  }
  else if (c == 'x') {
    // Raw register read: x <id> <reg>
    int space1 = cmd.indexOf(' ');
    int space2 = cmd.indexOf(' ', space1 + 1);
    int id = cmd.substring(space1 + 1, space2).toInt();
    int reg = cmd.substring(space2 + 1).toInt();
    int val = st.readByte(id, reg);
    Serial.printf("Servo %d, Register %d = %d (0x%02X)\n", id, reg, val, val);
  }
  else if (c == 'X') {
    // Dump registers 0-70 for a servo: X <id>
    int id = cmd.substring(2).toInt();
    Serial.printf("\nServo %d Register Dump:\n", id);
    for (int reg = 0; reg <= 70; reg++) {
      int val = st.readByte(id, reg);
      if (reg % 10 == 0) Serial.printf("\n%02d: ", reg);
      Serial.printf("%3d ", val);
      delay(2);
    }
    Serial.println("\n");
  }
  else if (c == 'W') {
    // Raw register write: W <id> <reg> <val>
    int space1 = cmd.indexOf(' ');
    int space2 = cmd.indexOf(' ', space1 + 1);
    int space3 = cmd.indexOf(' ', space2 + 1);
    int id = cmd.substring(space1 + 1, space2).toInt();
    int reg = cmd.substring(space2 + 1, space3).toInt();
    int val = cmd.substring(space3 + 1).toInt();
    st.writeByte(id, reg, val);
    delay(10);
    int readback = st.readByte(id, reg);
    Serial.printf("Wrote %d to servo %d reg %d, readback=%d\n", val, id, reg, readback);
  }
  else {
    Serial.println("Unknown command. Type 's' to scan servos.");
  }
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    processCommand(cmd);
    Serial.print("> ");
  }
}
