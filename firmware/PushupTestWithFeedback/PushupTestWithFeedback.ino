/* Push‑up with incremental steps, athletic stance, and proportional joint movement.
 * Five reps, then hold athletic stance.            */
#include <SCServo.h>
#include <cmath>  // for std::lerp
SMS_STS st;

#define S_RXD 18
#define S_TXD 19
const uint32_t BUS_BAUD = 1000000;

// Safety limits for robot movements
const float MAX_SHOULDER_ANGLE = 30.0f;  // Maximum shoulder angle from center
const float MAX_LEG_ANGLE = 90.0f;       // Maximum thigh/calf angle from center

// Simple torque estimation based on datasheet stall torque
// Assumes ~20 kg·cm stall torque at 12V for STS3215
constexpr float RATED_STALL_TORQUE_KGCM = 20.0f;
constexpr float KGCM_TO_NM = 0.0980665f;
constexpr float RATED_STALL_TORQUE_NM = RATED_STALL_TORQUE_KGCM * KGCM_TO_NM;

// Pushup movement angles
// THIGH_START and CALF_START define the "up" position (athletic stance)
// Calf range of motion is set to be twice the thigh's range of motion.
const float THIGH_START = -10.0f;   // Thighs slightly back for athletic stance
const float CALF_START = 20.0f;    // Calves more forward for a deeper knee bend in athletic stance
const float THIGH_END = -45.0f;    // End angle for thighs (down position). Thigh movement: -35 degrees.
const float CALF_END = 90.0f;     // End angle for calves (down position). Calf movement: +70 degrees.

const int DIR[13] = { 0,
  /*Front‑Left*/   +1,  /*ID 1*/   /*ID 2*/ +1 ,  /*ID 3*/ +1 ,
  /*Front‑Right*/  -1,  /*ID 4*/   /*ID 5*/ -1 ,  /*ID 6*/ -1 ,
  /*Rear‑Left*/    +1,  /*ID 7*/   /*ID 8*/ +1 ,  /*ID 9*/ +1 ,
  /*Rear‑Right*/   -1,  /*ID 10*/  /*ID 11*/ -1 , /*ID 12*/ -1
};

// Diagnostic data structure
struct Diag {
  bool ok;          // whether a full feedback frame was received
  int load;         // signed load value (-1000..+1000)
  int mA;           // motor current (mA)
  int tempC;        // temperature (°C)
  int voltage_dV;   // bus voltage in decivolts
};

// Function to read diagnostic data from a servo using a single feedback frame
Diag readDiag(uint8_t id) {
  Diag d{};
  int n = st.FeedBack(id);
  d.ok = (n >= 0);
  if (!d.ok) {
    // Populate with sentinel values; callers can skip printing when !ok
    d.load = d.mA = d.tempC = d.voltage_dV = -1;
    return d;
  }
  d.load = st.ReadLoad(-1);       // signed: library already decoded sign bit
  d.mA = st.ReadCurrent(-1);
  d.tempC = st.ReadTemper(-1);
  d.voltage_dV = st.ReadVoltage(-1);
  return d;
}

// Function to pretty-print diagnostic data for one servo
void printDiag(uint8_t id, const Diag& d) {
  if (!d.ok) {
    Serial.printf("ID%02u  (no data)\n", id);
    return;
  }
  int mag = d.load < 0 ? -d.load : d.load;
  if (mag > 1000) mag = 1000; // clamp to documented range
  char dir = (d.load < 0) ? '-' : '+';
  float pct = (mag / 1000.0f) * 100.0f;

  // Approximate joint torque from present load percent and torque limit.
  // This is an estimate; actual output depends on speed, friction, and control.
  int torque_lim = st.readWord(id, SMS_STS_TORQUE_LIMIT_L); // 0..1023
  if (torque_lim < 0) torque_lim = 1023; // assume full if unreadable
  float lim_frac = torque_lim / 1023.0f;
  float tau_nm = (d.load / 1000.0f) * RATED_STALL_TORQUE_NM * lim_frac;
  float tau_ncm = tau_nm * 100.0f; // display in N·cm for small robot

  Serial.printf(
    "ID%02u  Load:%c%4d (%.0f%%)  I:%4dmA  T:%3d°C  V:%.1fV  Torq≈%+.1f N·cm\n",
    id, dir, mag, pct, d.mA, d.tempC, d.voltage_dV / 10.0f, tau_ncm
  );
}

// Function to read and print diagnostics from all 12 servos
void dumpAllDiags() {
  Serial.println("=== Servo Diagnostics ===");
  for (uint8_t id = 1; id <= 12; ++id) {
    printDiag(id, readDiag(id));
  }
  Serial.println();
}

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
// Target 20 Hz command updates to match simulation (50 ms period)
const int STEP_MS = 50;                   // Milliseconds per interpolation step (20 Hz)
const int STEPS   = 24;                   // Steps per down/up phase (~1.2 s total at 20 Hz)

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

// Function to set all legs to the athletic (starting/top) position for push-ups
void setToAthleticStance() {
  Serial.println("Setting legs to athletic stance for push-up.");
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

  Serial.println("\n== Incremental Push‑up Test (Athletic Stance, Proportional), 5 reps ==");
  // Set legs to the defined athletic starting position for the push-up.
  setToAthleticStance();                               
  delay(800); // Pause after setting initial push-up pose, allowing servos to settle.

  // Perform 5 push-up repetitions
  for (int rep = 1; rep <= 5; ++rep) {
    Serial.printf("\nRep %d DOWN…\n", rep);
    // Down phase: interpolate from athletic stance (THIGH_START, CALF_START) 
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
    // Diagnostics at the bottom to avoid stalling motion
    dumpAllDiags();

    Serial.println("…UP");
    // Up phase: interpolate from bent positions back to athletic stance
    for (int s=STEPS; s>=0; --s) { // Iterate downwards to reverse the motion
      float t = float(s) / STEPS;  // t goes from 1.0 down to 0.0
      float current_thigh_angle = std::lerp(THIGH_START, THIGH_END, t);
      float current_calf_angle = std::lerp(CALF_START, CALF_END, t);
      writeAllLegsPushup(current_thigh_angle, current_calf_angle);
      delay(STEP_MS); // Wait for the small incremental move
    }
    delay(300); // Pause at the top of the push-up (athletic stance)
    // Diagnostics at the top to avoid stalling motion
    dumpAllDiags();
  }

  Serial.println("\nPush-up routine complete. Holding athletic stance.");
  // Ensure legs are in the athletic stance after the routine.
  setToAthleticStance(); 
  
  // Final diagnostic dump
  delay(100); // Let servos settle in final position
  Serial.println("=== Final Diagnostic State ===");
  dumpAllDiags(); 
}

void loop() {
  // The main routine is in setup() and runs once.
  // Loop can be used for continuous operations or left empty if not needed.
}
