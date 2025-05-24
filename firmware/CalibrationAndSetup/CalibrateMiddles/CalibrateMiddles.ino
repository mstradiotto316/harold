/*  Harold mid‑point calibrator  – line‑oriented for Arduino IDE
 *  Keys (type + Enter):
 *    a/d : –/+100   s/w : –/+10
 *    p   : save offset & re‑centre
 *    n   : next servo   q : quit
 */
#include <SCServo.h>
SMS_STS st;

#define S_RXD 18
#define S_TXD 19
const uint32_t BUS_BAUD = 1000000;

const uint8_t SERVO_IDS[] = { 1,2,3,4,5,6, 7,8,9,10,11,12 };
const uint8_t NUM_SERVOS  = sizeof(SERVO_IDS);

const int SPEED = 800, ACC = 100, MID = 2047;

void center(uint8_t id) {
  st.WritePosEx(id, MID, SPEED, ACC);
  delay(400);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  Serial1.begin(BUS_BAUD, SERIAL_8N1, S_RXD, S_TXD);
  st.pSerial = &Serial1;
  delay(500);

  Serial.println(F("\n== Mid‑point Calibrator =="));
  Serial.println(F("a/d = –/+100  s/w = –/+10  p = save  n = next  q = quit\n"));

  for (uint8_t idx = 0; idx < NUM_SERVOS; ++idx) {
    uint8_t id = SERVO_IDS[idx];
    Serial.printf("=== Servo ID %d ===\n", id);
    center(id);

    int pos = MID;
    while (true) {
      if (!Serial.available()) continue;
      String cmd = Serial.readStringUntil('\n');
      cmd.trim();

      if      (cmd == "a") pos -= 100;
      else if (cmd == "d") pos += 100;
      else if (cmd == "s") pos -= 10;
      else if (cmd == "w") pos += 10;
      else if (cmd == "p") {
        st.CalibrationOfs(id);
        Serial.println(F("  Offset SAVED – recentring…"));
        center(id);
        continue;
      }
      else if (cmd == "n") { Serial.println(); break; }
      else if (cmd == "q") { Serial.println(F("Exiting.")); while (true); }
      else                 { Serial.println(F("Un‑recognised key")); continue; }

      pos = constrain(pos, MID - 512, MID + 512);
      st.WritePosEx(id, pos, SPEED, ACC);
      Serial.printf("  pos = %d\n", pos);
    }
  }
  Serial.println(F("All servos trimmed – power‑cycle and test with MovementTest."));
}

void loop() {}
