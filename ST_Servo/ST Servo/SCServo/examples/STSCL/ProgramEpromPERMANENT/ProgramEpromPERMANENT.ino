#include <SCServo.h>
SMS_STS st;

#define S_RXD 18
#define S_TXD 19
const uint8_t OLD_ID = 2;      // current ID on power‑up
const uint8_t NEW_ID = 1;      // permanent ID you want

void setup() {
  Serial.begin(115200); while (!Serial) {}
  Serial1.begin(1000000, SERIAL_8N1, S_RXD, S_TXD);
  st.pSerial = &Serial1;
  delay(300);

  // 1) unlock EEPROM that currently belongs to OLD_ID
  if (st.unLockEprom(OLD_ID) == -1) {
    Serial.println("unlock fail"); return;
  }

  // 2) write NEW_ID into EEPROM address 0x05 (SMS_STS_ID)
  if (st.writeByte(OLD_ID, SMS_STS_ID, NEW_ID) == -1) {
    Serial.println("write fail"); return;
  }
  delay(30);                    // give EEPROM time

  // 3) lock AGAIN **on the OLD_ID** so the packet is accepted
  if (st.LockEprom(OLD_ID) == -1) {
    Serial.println("lock fail"); return;
  }

  Serial.printf("EEPROM ID %d → %d written — pull power now\n",
                OLD_ID, NEW_ID);
}

void loop() { }
