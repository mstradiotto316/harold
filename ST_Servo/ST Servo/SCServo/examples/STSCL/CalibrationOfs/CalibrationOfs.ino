/*
set the current position as middle.
*/

#include <SCServo.h>

SMS_STS st;

// the uart used to control servos.
// GPIO 18 - S_RXD, GPIO 19 - S_TXD, as default.
#define S_RXD 18
#define S_TXD 19
#define SERVO_ID 12               // change to 2,3,… as you go


void setup() {
  Serial.begin(115200);          // USB console
  while (!Serial) {}

  Serial1.begin(1000000, SERIAL_8N1, S_RXD, S_TXD);
  st.pSerial = &Serial1;
  delay(1000);

  if (st.CalibrationOfs(SERVO_ID) != -1) {
    Serial.printf("Offset set on ID %d\n", SERVO_ID);
  } else {
    Serial.printf("Failed on ID %d (check power / wiring)\n", SERVO_ID);
  }
}

void loop()
{
  st.CalibrationOfs(1);
  while(1);
}