/* Push‑up with incremental steps so calves (80°) and thighs (40°)
 * finish simultaneously.  Five reps, then hold straight.            */
#include <SCServo.h>
SMS_STS st;

#define S_RXD 18
#define S_TXD 19
const uint32_t BUS_BAUD = 1000000;

const int DIR[13] = { 0,
  /*Front‑Left*/   +1,  /*ID 1*/   /*ID 2*/ +1 ,  /*ID 3*/ +1 ,
  /*Front‑Right*/  -1,  /*ID 4*/   /*ID 5*/ -1 ,  /*ID 6*/ -1 ,
  /*Rear‑Left*/    +1,  /*ID 7*/   /*ID 8*/ +1 ,  /*ID 9*/ +1 ,
  /*Rear‑Right*/   -1,  /*ID 10*/  /*ID 11*/ -1 , /*ID 12*/ -1
};

inline int degToPos(uint8_t id, float deg, int mid=2047) {
  constexpr float U = 4096.0 / 360.0;
  return mid + int(DIR[id] * deg * U + 0.5f);
}

/* shoulder, thigh, calf IDs per leg */
const uint8_t LEG[4][3] = {{1,2,3},{4,5,6},{7,8,9},{10,11,12}};

const int SPEED = 1200, ACC = 100;        // enough speed for 25 ms steps
const int STEP_MS = 25;                   // 25 ms per increment
const int STEPS   = 40;                   // 40 × 1 ° = 40 °

void writeAll(float thigh_angle, float calf_angle) {
  // For each leg
  for (int l=0; l<4; ++l) {
    // Set the new thigh target position
    st.WritePosEx(LEG[l][1], degToPos(LEG[l][1], thigh_angle), SPEED, ACC);
    // Set the new calf target position
    st.WritePosEx(LEG[l][2], degToPos(LEG[l][2], calf_angle), SPEED, ACC);
  }
}

void straightLegs() {
  writeAll(0, 0);
}

void setup() {
  Serial.begin(115200); while (!Serial) {}
  Serial1.begin(BUS_BAUD, SERIAL_8N1, S_RXD, S_TXD); st.pSerial = &Serial1;
  delay(500);

  Serial.println("\n== Incremental Push‑up Test, 5 reps ==");
  straightLegs();                               // start tall
  delay(800);

  // For each pushup repetition
  for (int rep = 1; rep <= 5; ++rep) {
    Serial.printf("Rep %d DOWN…\n", rep);
    /* down phase: thighs -40  =>  -1 °/step ; calves +80  =>  +2 °/step */
    for (int s=1; s<=STEPS; ++s) {
      writeAll(-1.25f * s, 2.5f * s);
      delay(STEP_MS);
    }
    delay(150);                                 // bottom pause

    Serial.println("…UP");
    for (int s=STEPS; s>=0; --s) {              // reverse
      writeAll(-1.25f * s, 2.5f * s);
      delay(STEP_MS);
    }
    delay(250);                                 // top pause
  }

  Serial.println("Done – holding straight pose.");
}

void loop() {}
