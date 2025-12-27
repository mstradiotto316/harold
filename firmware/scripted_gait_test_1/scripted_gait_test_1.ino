/* Scripted Diagonal Trot Gait Test v3
 * Tests walking gait with parameters ALIGNED TO SIMULATION.
 *
 * Safety features:
 *   - Collision-safe joint limits (not just servo limits)
 *   - Load monitoring with auto-stop
 *
 * Gait parameters match simulation ScriptedGaitCfg exactly:
 *   - 1.0 Hz frequency (1 second per cycle)
 *   - Full amplitude (100%)
 *   - Thigh/calf angles converted from sim radians with sign inversion
 *
 * Gait pattern: Diagonal trot
 *   - FL + BR move together (phase 0)
 *   - FR + BL move together (phase 0.5)
 */

#include <SCServo.h>
#include <cmath>

SMS_STS st;

#define S_RXD 18
#define S_TXD 19
const uint32_t BUS_BAUD = 1000000;

// ============================================================================
// GAIT PARAMETERS - ALIGNED WITH SIMULATION (ScriptedGaitCfg)
// ============================================================================
// Simulation values (radians) converted to hardware (degrees) with sign inversion:
//   hardware_deg = -sim_rad * (180/π)
// Thighs: sim +0.90/+0.40 rad → hw -51.6/-22.9 deg
// Calves: sim -0.90/-1.40 rad → hw +51.6/+80.0 deg
const float BASE_STANCE_THIGH = -51.6f;  // sim: +0.90 rad
const float BASE_SWING_THIGH  = -22.9f;  // sim: +0.40 rad
const float BASE_STANCE_CALF  = 51.6f;   // sim: -0.90 rad (extended during stance)
const float BASE_SWING_CALF   = 80.0f;   // sim: -1.40 rad (bent during swing, at CALF_MAX)
const float BASE_SHOULDER_AMP = 2.9f;    // sim: 0.05 rad

// Active parameters (scaled by amplitude)
float STANCE_THIGH, SWING_THIGH, STANCE_CALF, SWING_CALF, SHOULDER_AMPLITUDE;

// Amplitude scaling (0.0 to 1.0) - use full amplitude for sim comparison
float gaitAmplitude = 1.0f;  // Full amplitude to match simulation

// Timing - MUST match simulation frequency for comparison
float GAIT_FREQUENCY = 0.5f;   // Hz - slower to match real servo response (2 second cycle)
const int UPDATE_RATE_HZ = 20;
const int STEP_MS = 1000 / UPDATE_RATE_HZ;

// Duration
const int WARMUP_SECONDS = 3;
const int WALK_SECONDS = 10;
const int COOLDOWN_SECONDS = 2;

// ============================================================================
// COLLISION-SAFE LIMITS (based on actual robot geometry)
// ============================================================================
const float SHOULDER_MIN = -20.0f;
const float SHOULDER_MAX = 20.0f;
const float THIGH_MIN = -55.0f;   // Forward lean limit
const float THIGH_MAX = 5.0f;     // Backward limit (hits body!)
const float CALF_MIN = -5.0f;     // Extension limit
const float CALF_MAX = 80.0f;     // Bend limit

// ============================================================================
// LOAD MONITORING
// ============================================================================
const int LOAD_WARNING_THRESHOLD = 700;   // 70% load - print warning
const int LOAD_CRITICAL_THRESHOLD = 900;  // 90% load - stop immediately
bool emergencyStop = false;

// ============================================================================
// SERVO CONFIGURATION
// ============================================================================
const int DIR[13] = { 0,
  /*Front-Left*/   +1,  +1,  +1,
  /*Front-Right*/  -1,  -1,  -1,
  /*Rear-Left*/    +1,  +1,  +1,
  /*Rear-Right*/   -1,  -1,  -1
};

const uint8_t LEG[4][3] = {
    {1, 2, 3},     // Front-Left (FL)
    {4, 5, 6},     // Front-Right (FR)
    {7, 8, 9},     // Rear-Left (BL)
    {10, 11, 12}   // Rear-Right (BR)
};

const int LEG_FL = 0;
const int LEG_FR = 1;
const int LEG_BL = 2;
const int LEG_BR = 3;

// Speed settings: 0 = max speed, higher value = faster (steps/second)
// Feetech units: roughly steps/second (4096 steps = 360°)
const int GAIT_SPEED = 0, GAIT_ACC = 0;         // Fast for gait (servo tracks trajectory)
const int TRANSITION_SPEED_THIGH = 400, TRANSITION_ACC = 50;  // Slow for stance changes
const int TRANSITION_SPEED_CALF = 800;  // Calf moves 2x faster to compensate for thigh rotation

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

float clampSafe(float value, float joint_min, float joint_max) {
  if (value < joint_min) return joint_min;
  if (value > joint_max) return joint_max;
  return value;
}

inline int degToPos(uint8_t id, float deg, int mid = 2047) {
  constexpr float U = 4096.0f / 360.0f;
  return mid + int(DIR[id] * deg * U + 0.5f);
}

// Apply amplitude scaling to gait parameters
void updateGaitParameters() {
  float center_thigh = (BASE_STANCE_THIGH + BASE_SWING_THIGH) / 2.0f;
  float range_thigh = (BASE_SWING_THIGH - BASE_STANCE_THIGH) / 2.0f;
  STANCE_THIGH = center_thigh - range_thigh * gaitAmplitude;
  SWING_THIGH = center_thigh + range_thigh * gaitAmplitude;

  float center_calf = (BASE_STANCE_CALF + BASE_SWING_CALF) / 2.0f;
  float range_calf = (BASE_SWING_CALF - BASE_STANCE_CALF) / 2.0f;
  STANCE_CALF = center_calf - range_calf * gaitAmplitude;
  SWING_CALF = center_calf + range_calf * gaitAmplitude;

  SHOULDER_AMPLITUDE = BASE_SHOULDER_AMP * gaitAmplitude;
}

// Check load on all servos, return max load percentage
int checkLoads() {
  int maxLoad = 0;
  for (uint8_t id = 1; id <= 12; ++id) {
    int n = st.FeedBack(id);
    if (n >= 0) {
      int load = abs(st.ReadLoad(-1));
      if (load > maxLoad) maxLoad = load;
    }
  }
  return maxLoad;
}

void printLoadWarning(int load) {
  Serial.printf("WARNING: High servo load detected: %d%% - consider stopping!\n", load / 10);
}

// ============================================================================
// GAIT FUNCTIONS
// ============================================================================

// Fast version for gait (all joints same speed)
void setLegAngles(int leg, float shoulder, float thigh, float calf) {
  // Apply collision-safe limits
  shoulder = clampSafe(shoulder, SHOULDER_MIN, SHOULDER_MAX);
  thigh = clampSafe(thigh, THIGH_MIN, THIGH_MAX);
  calf = clampSafe(calf, CALF_MIN, CALF_MAX);

  st.WritePosEx(LEG[leg][0], degToPos(LEG[leg][0], shoulder), GAIT_SPEED, GAIT_ACC);
  st.WritePosEx(LEG[leg][1], degToPos(LEG[leg][1], thigh), GAIT_SPEED, GAIT_ACC);
  st.WritePosEx(LEG[leg][2], degToPos(LEG[leg][2], calf), GAIT_SPEED, GAIT_ACC);
}

// Slow version for transitions (calf 2x faster to compensate for thigh rotation)
void setLegAnglesTransition(int leg, float shoulder, float thigh, float calf) {
  shoulder = clampSafe(shoulder, SHOULDER_MIN, SHOULDER_MAX);
  thigh = clampSafe(thigh, THIGH_MIN, THIGH_MAX);
  calf = clampSafe(calf, CALF_MIN, CALF_MAX);

  st.WritePosEx(LEG[leg][0], degToPos(LEG[leg][0], shoulder), TRANSITION_SPEED_THIGH, TRANSITION_ACC);
  st.WritePosEx(LEG[leg][1], degToPos(LEG[leg][1], thigh), TRANSITION_SPEED_THIGH, TRANSITION_ACC);
  st.WritePosEx(LEG[leg][2], degToPos(LEG[leg][2], calf), TRANSITION_SPEED_CALF, TRANSITION_ACC);
}

void computeLegTrajectory(float phase, float* shoulder, float* thigh, float* calf) {
  float sin_phase = sinf(phase * 2.0f * M_PI);
  float cos_phase = cosf(phase * 2.0f * M_PI);

  // Thigh trajectory: use -SIN to match simulation's forward walking
  // The key is that thigh should move BACKWARD (toward STANCE) during ground contact
  // Calf uses: calf = mid - amp*cos, so foot DOWN at phase 0, UP at phase 0.5
  // Thigh needs: start at SWING (forward) when foot touches, end at STANCE (back) at liftoff
  // Using -SIN: at phase 0 -> mid, phase 0.25 -> STANCE (back), phase 0.5 -> mid
  // This puts the backward motion during the first half of stance
  float thigh_mid = (STANCE_THIGH + SWING_THIGH) / 2.0f;
  float thigh_amp = (SWING_THIGH - STANCE_THIGH) / 2.0f;
  *thigh = thigh_mid - thigh_amp * sin_phase;  // -SIN for forward walking

  float calf_mid = (STANCE_CALF + SWING_CALF) / 2.0f;
  float calf_amp = (SWING_CALF - STANCE_CALF) / 2.0f;
  *calf = calf_mid - calf_amp * cos_phase;

  *shoulder = SHOULDER_AMPLITUDE * sin_phase;
}

void setAthleticStance() {
  Serial.println("Setting athletic stance (slow transition)...");
  float mid_thigh = (STANCE_THIGH + SWING_THIGH) / 2.0f;
  float mid_calf = (STANCE_CALF + SWING_CALF) / 2.0f;

  for (int leg = 0; leg < 4; ++leg) {
    setLegAnglesTransition(leg, 0.0f, mid_thigh, mid_calf);
  }
  Serial.printf("Athletic stance: thigh=%.1f deg, calf=%.1f deg\n", mid_thigh, mid_calf);
}

void resetAllJointsToZero() {
  Serial.println("Resetting all joints to 0 degrees (slow transition)...");
  for (int leg = 0; leg < 4; ++leg) {
    setLegAnglesTransition(leg, 0.0f, 0.0f, 0.0f);
  }
  delay(2500);  // Longer delay for slow transition
}

void dumpAllDiags() {
  Serial.println("=== Servo Diagnostics ===");
  for (uint8_t id = 1; id <= 12; ++id) {
    int n = st.FeedBack(id);
    if (n >= 0) {
      int load = st.ReadLoad(-1);
      int mA = st.ReadCurrent(-1);
      int temp = st.ReadTemper(-1);
      Serial.printf("ID%02u  Load:%4d (%d%%)  I:%4dmA  T:%d°C\n",
                    id, load, abs(load)/10, mA, temp);
    } else {
      Serial.printf("ID%02u  (no data)\n", id);
    }
  }
  Serial.println();
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  Serial1.begin(BUS_BAUD, SERIAL_8N1, S_RXD, S_TXD);
  st.pSerial = &Serial1;

  Serial.println(F("\n============================================="));
  Serial.println(F("  Harold Scripted Diagonal Trot - SAFE MODE"));
  Serial.println(F("============================================="));
  Serial.println(F("Safety features enabled:"));
  Serial.println(F("  - Collision-safe joint limits"));
  Serial.println(F("  - Load monitoring with auto-stop"));
  Serial.println(F("  - Conservative starting amplitude\n"));

  // Initialize gait parameters
  updateGaitParameters();

  Serial.println(F("Current settings:"));
  Serial.printf("  Amplitude: %.0f%%\n", gaitAmplitude * 100);
  Serial.printf("  Frequency: %.2f Hz (%.1f sec/cycle)\n", GAIT_FREQUENCY, 1.0f/GAIT_FREQUENCY);
  Serial.printf("  Thigh range: %.1f to %.1f deg\n", STANCE_THIGH, SWING_THIGH);
  Serial.printf("  Calf range: %.1f to %.1f deg\n", STANCE_CALF, SWING_CALF);
  Serial.println();

  Serial.println(F("Joint safety limits:"));
  Serial.printf("  Shoulder: %.0f to %.0f deg\n", SHOULDER_MIN, SHOULDER_MAX);
  Serial.printf("  Thigh: %.0f to %.0f deg\n", THIGH_MIN, THIGH_MAX);
  Serial.printf("  Calf: %.0f to %.0f deg\n", CALF_MIN, CALF_MAX);
  Serial.println();

  delay(1000);

  // Initialize to zero
  resetAllJointsToZero();

  // Move to athletic stance
  updateGaitParameters();
  setAthleticStance();
  Serial.printf("Warming up for %d seconds...\n", WARMUP_SECONDS);
  delay(WARMUP_SECONDS * 1000);

  // Initial diagnostics
  dumpAllDiags();

  // Auto-start after brief pause (headless mode)
  Serial.println(F("============================================="));
  Serial.println(F("Starting walking test in 2 seconds..."));
  Serial.println(F("(During walking: press any key to stop)"));
  Serial.println(F("============================================="));
  delay(2000);

  // Walking phase
  Serial.println(F("\n=== STARTING DIAGONAL TROT ==="));
  Serial.printf("Walking for %d seconds (press any key to stop)...\n\n", WALK_SECONDS);

  unsigned long start_time = millis();
  unsigned long walk_duration_ms = WALK_SECONDS * 1000;
  int step_count = 0;
  int loadCheckCounter = 0;

  while (millis() - start_time < walk_duration_ms && !emergencyStop) {
    // Check for user abort
    if (Serial.available()) {
      Serial.readStringUntil('\n');
      Serial.println(F("\n*** USER STOP ***"));
      break;
    }

    // Compute current time in seconds
    float t = (millis() - start_time) / 1000.0f;

    // Compute phase for each diagonal pair
    float phase_A = fmodf(t * GAIT_FREQUENCY, 1.0f);
    float phase_B = fmodf(t * GAIT_FREQUENCY + 0.5f, 1.0f);

    // Compute trajectories
    float shoulder_A, thigh_A, calf_A;
    float shoulder_B, thigh_B, calf_B;

    computeLegTrajectory(phase_A, &shoulder_A, &thigh_A, &calf_A);
    computeLegTrajectory(phase_B, &shoulder_B, &thigh_B, &calf_B);

    // Apply to diagonal pairs
    setLegAngles(LEG_FL, shoulder_A, thigh_A, calf_A);
    setLegAngles(LEG_BR, shoulder_A, thigh_A, calf_A);
    setLegAngles(LEG_FR, shoulder_B, thigh_B, calf_B);
    setLegAngles(LEG_BL, shoulder_B, thigh_B, calf_B);

    // Check loads every 0.5 seconds
    loadCheckCounter++;
    if (loadCheckCounter % (UPDATE_RATE_HZ / 2) == 0) {
      int maxLoad = checkLoads();
      if (maxLoad > LOAD_CRITICAL_THRESHOLD) {
        Serial.printf("\n*** EMERGENCY STOP: Load %d%% exceeds critical threshold! ***\n", maxLoad/10);
        emergencyStop = true;
        break;
      } else if (maxLoad > LOAD_WARNING_THRESHOLD) {
        printLoadWarning(maxLoad);
      }
    }

    // Debug output every 1 second
    step_count++;
    if (step_count % UPDATE_RATE_HZ == 0) {
      Serial.printf("t=%.1fs  thA=%.1f caA=%.1f  thB=%.1f caB=%.1f\n",
                    t, thigh_A, calf_A, thigh_B, calf_B);
    }

    delay(STEP_MS);
  }

  Serial.println(F("\n=== WALKING COMPLETE ==="));

  // Return to athletic stance
  setAthleticStance();
  Serial.printf("Cooling down for %d seconds...\n", COOLDOWN_SECONDS);
  delay(COOLDOWN_SECONDS * 1000);

  // Final diagnostics
  Serial.println(F("=== Final Diagnostic State ==="));
  dumpAllDiags();

  if (emergencyStop) {
    Serial.println(F("\n*** TEST ENDED DUE TO EMERGENCY STOP ***"));
    Serial.println(F("Check servo loads and consider reducing amplitude."));
  } else {
    Serial.println(F("\nTest complete. Observations:"));
    Serial.println(F("1. Did the robot walk forward?"));
    Serial.println(F("2. Was the gait stable?"));
    Serial.println(F("3. Consider increasing amplitude if motion was too small."));
  }

  // Return to zero
  Serial.println(F("\nReturning to zero position..."));
  resetAllJointsToZero();
}

void loop() {
  // Nothing - test runs once in setup()
}
