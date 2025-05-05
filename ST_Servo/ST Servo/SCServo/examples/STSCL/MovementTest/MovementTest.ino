/*
 * Simple motion check for one ST‑series servo.
 * Sequence: mid → +45° → mid → −45° → mid
 * Works with the same SCServo / SMS_STS objects as Ping / ProgramEprom.
 */
#include <SCServo.h>

SMS_STS st;

#define S_RXD      18          // data pins – unchanged
#define S_TXD      19
#define SERVO_ID    1          // <-- set to the ID you’re testing

// ----- user‑tweakable params -----
const float SERVO_DEG_PER_UNIT = 360.0 / 4096.0;   // ST range
const int   SPEED               = 800;             // ≈ 20 rpm
const int   ACC                 = 100;             // reasonable accel
const int   PAUSE_MS            = 1000;            // pause between moves
// ---------------------------------

static inline int degToPos(float deg, int mid = 2047) {
  return mid + int(deg / SERVO_DEG_PER_UNIT + 0.5f);
}

static void moveAndWait(int pos) {
  st.WritePosEx(SERVO_ID, pos, SPEED, ACC);
  delay(PAUSE_MS);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  Serial1.begin(1000000, SERIAL_8N1, S_RXD, S_TXD);
  st.pSerial = &Serial1;
  delay(500);

  const int MID      = 2047;                // electrical centre
  const int FWD45    = degToPos(+45.0f);    // ≈ 2560
  const int BACK45   = degToPos(-45.0f);    // ≈ 1535

  Serial.printf("Movement test on ID %d …\n", SERVO_ID);

  moveAndWait(MID);
  moveAndWait(FWD45);
  moveAndWait(MID);
  moveAndWait(BACK45);
  moveAndWait(MID);

  Serial.println("…test complete.\n");
}

void loop() { /* nothing */ }
