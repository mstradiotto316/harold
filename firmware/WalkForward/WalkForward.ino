#include <SCServo.h>
#include <cmath>

// Simple quasi-static crawl gait for the Harold quadruped.
// Moves one leg at a time while keeping the other three on the ground.

SMS_STS st;

#define S_RXD 18
#define S_TXD 19
const uint32_t BUS_BAUD = 1000000;

const float MAX_SHOULDER_ANGLE = 30.0f; // safety limits
const float MAX_LEG_ANGLE      = 90.0f;

// Servo direction: positive values move as described in README
const int DIR[13] = {0,
  /*Front-Left*/  +1, +1, +1,
  /*Front-Right*/ -1, -1, -1,
  /*Rear-Left*/   +1, +1, +1,
  /*Rear-Right*/  -1, -1, -1
};

const uint8_t LEG[4][3] = {
  {1,2,3},   // Front-Left
  {4,5,6},   // Front-Right
  {7,8,9},   // Rear-Left
  {10,11,12} // Rear-Right
};

const int SPEED = 1000;
const int ACC   = 100;

inline int degToPos(uint8_t id, float deg, int mid = 2047) {
  constexpr float U = 4096.0f / 360.0f; // units per degree
  return mid + int(DIR[id] * deg * U + 0.5f);
}

float clampf(float v, float mn, float mx) {
  if (v < mn) return mn;
  if (v > mx) return mx;
  return v;
}

static float currentPose[12] = {0};

void writePose(const float angles[12]) {
  for (int l = 0; l < 4; ++l) {
    float sh = clampf(angles[l],     -MAX_SHOULDER_ANGLE, MAX_SHOULDER_ANGLE);
    float th = clampf(angles[l+4],   -MAX_LEG_ANGLE,      MAX_LEG_ANGLE);
    float ca = clampf(angles[l+8],   -MAX_LEG_ANGLE,      MAX_LEG_ANGLE);
    st.WritePosEx(LEG[l][0], degToPos(LEG[l][0], sh), SPEED, ACC);
    st.WritePosEx(LEG[l][1], degToPos(LEG[l][1], th), SPEED, ACC);
    st.WritePosEx(LEG[l][2], degToPos(LEG[l][2], ca), SPEED, ACC);
  }
}

void interpolatePose(const float target[12], int steps, int stepDelay) {
  float start[12];
  for (int i=0;i<12;++i) start[i] = currentPose[i];
  for (int s=1; s<=steps; ++s) {
    float interp[12];
    float t = (float)s / steps;
    for (int j=0;j<12;++j) {
      interp[j] = start[j] + (target[j] - start[j]) * t;
    }
    writePose(interp);
    delay(stepDelay);
  }
  for (int i=0;i<12;++i) currentPose[i] = target[i];
}

void resetAllJointsToZero() {
  float zeros[12] = {0};
  interpolatePose(zeros, 40, 25); // smooth transition to neutral
}

void safeShutdown() {
  Serial.println("Safe shutdown triggered - returning to neutral pose.");
  resetAllJointsToZero();
  while (true) { delay(1000); }
}

// Neutral standing pose
const float STANCE[12] = {
  0, 0, 0, 0,       // shoulders
 -5,-5,-5,-5,       // thighs slightly back
 30,30,30,30        // knees bent
};

// Helper to build a pose from current with one leg modified
void copyPose(float dest[12], const float src[12]) {
  for(int i=0;i<12;++i) dest[i]=src[i];
}

void stepLeg(int leg) {
  // Leg index: 0 FL,1 FR,2 RL,3 RR
  int sh = leg;       // shoulder index
  int th = leg + 4;   // thigh index
  int kn = leg + 8;   // knee index
  bool rightSide = (leg == 1) || (leg == 3);

  float pose[12];
  copyPose(pose, currentPose);

  // 1) shift body weight to opposite side using shoulders
  pose[0] = rightSide ? 5  : -5; // FL shoulder
  pose[1] = rightSide ? -5 : 5;  // FR shoulder
  pose[2] = rightSide ? 5  : -5; // RL shoulder
  pose[3] = rightSide ? -5 : 5;  // RR shoulder
  interpolatePose(pose, 20, 30);

  // 2) lift knee
  pose[kn] = 60;
  interpolatePose(pose, 20, 30);

  // 3) swing thigh forward
  pose[th] = 20;
  interpolatePose(pose, 30, 30);

  // 4) lower knee to stance height
  pose[kn] = 30;
  interpolatePose(pose, 20, 30);

  // 5) restore shoulders
  pose[0] = pose[1] = pose[2] = pose[3] = 0;
  interpolatePose(pose, 20, 30);
}

void setup() {
  Serial.begin(115200); while (!Serial) {}
  Serial1.begin(BUS_BAUD, SERIAL_8N1, S_RXD, S_TXD); st.pSerial = &Serial1;
  delay(2000);
  Serial.println("\n== Simple Crawl Walk ==");
  copyPose(currentPose, STANCE);
  resetAllJointsToZero();
  interpolatePose(STANCE, 40, 25);
}

void loop() {
  if (Serial.available()) {
    if (Serial.read() == 's') { safeShutdown(); }
  }

  stepLeg(3); // Rear Right
  stepLeg(1); // Front Right
  stepLeg(2); // Rear Left
  stepLeg(0); // Front Left
}

