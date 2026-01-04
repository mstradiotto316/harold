/* Push-up with incremental steps, pushup start pose, and proportional joint movement.
 * Five reps, then hold the pushup start pose.       */
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
// THIGH_START and CALF_START define the "up" position (pushup start pose)
// Calf range of motion is set to be twice the thigh's range of motion.
const float THIGH_START = -10.0f;   // Thighs slightly back for pushup start pose
const float CALF_START = 20.0f;    // Calves more forward for a deeper knee bend in pushup start pose
const float THIGH_END = -45.0f;    // End angle for thighs (down position). Thigh movement: -35 degrees.
const float CALF_END = 90.0f;     // End angle for calves (down position). Calf movement: +70 degrees.

const int DIR[13] = { 0,
  /*Front‑Left*/   +1,  /*ID 1*/   /*ID 2*/ +1 ,  /*ID 3*/ +1 ,
  /*Front‑Right*/  -1,  /*ID 4*/   /*ID 5*/ -1 ,  /*ID 6*/ -1 ,
  /*Rear‑Left*/    +1,  /*ID 7*/   /*ID 8*/ +1 ,  /*ID 9*/ +1 ,
  /*Rear‑Right*/   -1,  /*ID 10*/  /*ID 11*/ -1 , /*ID 12*/ -1
};

// Function to convert degrees to servo position units
// id: servo ID
// deg: angle in degrees
// mid: servo middle position (default 2047 for 0-4095 range)
inline int degToPos(uint8_t id, float deg, int mid=2047) {
  constexpr float U = 4096.0 / 360.0; // Units per degree
  return mid + int(DIR[id] * deg * U + 0.5f); // Apply direction, convert, and round
}

/* shoulder, thigh, calf IDs per leg */
// LEG[leg_index][joint_index]
// joint_index: 0 for shoulder, 1 for thigh, 2 for calf
const uint8_t LEG[4][3] = {
    {1,2,3},   // Front-Left leg: Shoulder ID 1, Thigh ID 2, Calf ID 3
    {4,5,6},   // Front-Right leg: Shoulder ID 4, Thigh ID 5, Calf ID 6
    {7,8,9},   // Rear-Left leg: Shoulder ID 7, Thigh ID 8, Calf ID 9
    {10,11,12} // Rear-Right leg: Shoulder ID 10, Thigh ID 11, Calf ID 12
};

const int SPEED = 1800, ACC = 100;        // Servo speed and acceleration
const int STEP_MS = 30;                   // Milliseconds per interpolation step
const int STEPS   = 40;                   // Number of interpolation steps for push-up movement

// Function to clamp a value between a minimum and maximum
float clamp(float value, float min_val, float max_val) {
  if (value < min_val) return min_val;
  if (value > max_val) return max_val;
  return value;
}

// Function to write target angles to thigh and calf servos of all legs
void writeAllLegsPushup(float thigh_angle, float calf_angle) {
  // Clamp the angles to safety limits
  float clamped_thigh_angle = clamp(thigh_angle, -MAX_LEG_ANGLE, MAX_LEG_ANGLE);
  float clamped_calf_angle = clamp(calf_angle, -MAX_LEG_ANGLE, MAX_LEG_ANGLE);
  
  // Debug output for angles being sent
  Serial.printf("Targeting Thigh: %.1f° (Clamped: %.1f°), Calf: %.1f° (Clamped: %.1f°)\n", 
                thigh_angle, clamped_thigh_angle, calf_angle, clamped_calf_angle);
  
  // For each leg
  for (int l=0; l<4; ++l) {
    // Set the new thigh target position
    // LEG[l][1] is the thigh servo ID for leg l
    st.WritePosEx(LEG[l][1], degToPos(LEG[l][1], clamped_thigh_angle), SPEED, ACC);
    // Set the new calf target position
    // LEG[l][2] is the calf servo ID for leg l
    st.WritePosEx(LEG[l][2], degToPos(LEG[l][2], clamped_calf_angle), SPEED, ACC);
  }
}

// Function to set all legs to the pushup start position
void setToPushupStartPose() {
  Serial.println("Setting legs to pushup start pose.");
  writeAllLegsPushup(THIGH_START, CALF_START);
}

// Function to reset all 12 joints (shoulders, thighs, calves) to 0 degrees
void resetAllJointsToZero() {
  Serial.println("Resetting all joints to 0 degrees...");
  for (int l = 0; l < 4; ++l) { // Iterate through each leg
    // Reset Shoulder joint (LEG[l][0])
    st.WritePosEx(LEG[l][0], degToPos(LEG[l][0], 0.0f), SPEED, ACC);
    // Reset Thigh joint (LEG[l][1])
    st.WritePosEx(LEG[l][1], degToPos(LEG[l][1], 0.0f), SPEED, ACC);
    // Reset Calf joint (LEG[l][2])
    st.WritePosEx(LEG[l][2], degToPos(LEG[l][2], 0.0f), SPEED, ACC);
  }
  Serial.println("All joints commanded to 0 degrees. Waiting for servos to reach position...");
  delay(1500); // Give some time for servos to move to the zero position. Adjust if necessary.
}

void setup() {
  // Initialize Serial communication for debugging
  Serial.begin(115200); 
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB
  }

  // Initialize Serial1 for SCServo communication
  Serial1.begin(BUS_BAUD, SERIAL_8N1, S_RXD, S_TXD); 
  st.pSerial = &Serial1; // Assign Serial1 to the servo library
  
  Serial.println("System Initializing...");
  delay(2000); // Initial delay for system stabilization and for servos to power up

  // Reset all joints to 0 degrees before starting any specific pose or movement
  resetAllJointsToZero();

  Serial.println("\n== Incremental Push-up Test (Pushup Start Pose, Proportional), 5 reps ==");
  // Set legs to the defined pushup start position.
  setToPushupStartPose();
  delay(800); // Pause after setting initial push-up pose, allowing servos to settle.

  // Perform 5 push-up repetitions
  for (int rep = 1; rep <= 5; ++rep) {
    Serial.printf("\nRep %d DOWN…\n", rep);
    // Down phase: interpolate from pushup start pose (THIGH_START, CALF_START)
    // to bent positions (THIGH_END, CALF_END)
    for (int s=1; s<=STEPS; ++s) {
      float t = float(s) / STEPS;  // t is the interpolation factor, goes from 0.0 to 1.0
      // Calculate current thigh and calf angles using linear interpolation (lerp)
      float current_thigh_angle = std::lerp(THIGH_START, THIGH_END, t);
      float current_calf_angle = std::lerp(CALF_START, CALF_END, t);
      writeAllLegsPushup(current_thigh_angle, current_calf_angle);
      delay(STEP_MS); // Wait for the small incremental move
    }
    delay(300); // Pause at the bottom of the push-up

    Serial.println("…UP");
    // Up phase: interpolate from bent positions back to pushup start pose
    for (int s=STEPS; s>=0; --s) { // Iterate downwards to reverse the motion
      float t = float(s) / STEPS;  // t goes from 1.0 down to 0.0
      float current_thigh_angle = std::lerp(THIGH_START, THIGH_END, t);
      float current_calf_angle = std::lerp(CALF_START, CALF_END, t);
      writeAllLegsPushup(current_thigh_angle, current_calf_angle);
      delay(STEP_MS); // Wait for the small incremental move
    }
    delay(300); // Pause at the top of the push-up (pushup start pose)
  }

  Serial.println("\nPush-up routine complete. Holding pushup start pose.");
  // Ensure legs are in the pushup start pose after the routine.
  setToPushupStartPose();
}

void loop() {
  // The main routine is in setup() and runs once.
  // Loop can be used for continuous operations or left empty if not needed.
}
