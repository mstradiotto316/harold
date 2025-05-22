/* Push‑up with incremental steps so calves (80°) and thighs (40°)
 * finish simultaneously.  Five reps, then hold straight.            */
#include <SCServo.h>
#include <cmath>  // for std::lerp
SMS_STS st;

#define S_RXD 18
#define S_TXD 19
const uint32_t BUS_BAUD = 1000000;

// Safety limits for robot movements
const float MAX_SHOULDER_ANGLE = 30.0f;  // Maximum shoulder angle from center
const float MAX_LEG_ANGLE = 90.0f;       // Maximum thigh/calf angle from center

// Pushup movement angles
const float THIGH_START = 0.0f;    // Starting angle for thighs
const float THIGH_END = -45.0f;    // End angle for thighs
const float CALF_START = 0.0f;     // Starting angle for calves
const float CALF_END = 90.0f;     // End angle for calves

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

const int SPEED = 1200, ACC = 100;        // enough speed for 25 ms steps
const int STEP_MS = 50;                   // 25 ms per increment
const int STEPS   = 40;                   // Number of interpolation steps

// Function to clamp a value between min and max
float clamp(float value, float min_val, float max_val) {
  if (value < min_val) return min_val;
  if (value > max_val) return max_val;
  return value;
}

void writeAll(float thigh_angle, float calf_angle) {
  // Clamp the angles to safety limits
  thigh_angle = clamp(thigh_angle, -MAX_LEG_ANGLE, MAX_LEG_ANGLE);
  calf_angle = clamp(calf_angle, -MAX_LEG_ANGLE, MAX_LEG_ANGLE);
  
  // Debug output for angles
  Serial.printf("Thigh: %.1f°, Calf: %.1f°\n", thigh_angle, calf_angle);
  
  // For each leg
  for (int l=0; l<4; ++l) {
    // Set the new thigh target position
    st.WritePosEx(LEG[l][1], degToPos(LEG[l][1], thigh_angle), SPEED, ACC);
    // Set the new calf target position
    st.WritePosEx(LEG[l][2], degToPos(LEG[l][2], calf_angle), SPEED, ACC);
  }
}

void straightLegs() {
  writeAll(THIGH_START, CALF_START);
}

void setup() {
  Serial.begin(115200); while (!Serial) {}
  Serial1.begin(BUS_BAUD, SERIAL_8N1, S_RXD, S_TXD); st.pSerial = &Serial1;
  delay(2000);

  Serial.println("\n== Incremental Push‑up Test, 5 reps ==");
  straightLegs();                               // start tall
  delay(800);

  // For each pushup repetition
  for (int rep = 1; rep <= 5; ++rep) {
    Serial.printf("\nRep %d DOWN…\n", rep);
    // Down phase: interpolate from start to end positions
    for (int s=1; s<=STEPS; ++s) {
      float t = float(s) / STEPS;  // interpolation factor 0.0 to 1.0
      writeAll(
        std::lerp(THIGH_START, THIGH_END, t),
        std::lerp(CALF_START, CALF_END, t)
      );
      delay(STEP_MS);
    }
    delay(300);                                 // bottom pause

    Serial.println("…UP");
    // Up phase: interpolate from end to start positions
    for (int s=STEPS; s>=0; --s) {
      float t = float(s) / STEPS;  // interpolation factor 1.0 to 0.0
      writeAll(
        std::lerp(THIGH_START, THIGH_END, t),
        std::lerp(CALF_START, CALF_END, t)
      );
      delay(STEP_MS);
    }
    delay(300);                                 // top pause
  }

  Serial.println("Done – holding straight pose.");
}

void loop() {}
