/* Harold streaming control firmware
 * ---------------------------------
 * - Handshake: host sends "READY?", ESP32 replies "ARDUINO_READY" once initialised
 * - Commands: 12 joint radians in RL order [shoulders 4][thighs 4][calves 4], comma separated, terminated with '#'
 * - Keywords: "START#" enables command execution, "STOP#" disables and returns to nominal stance
 * - Watchdog: if no command within 250 ms, revert to safe stance and pause streaming
 * - Telemetry: every 50 ms send TELEM packet with joint rad (RL sign), load (percent), current (mA), temperature (C)
 */
#include <SCServo.h>
#include <cmath>
#include <cstring>

SMS_STS st;

// --- Hardware configuration -------------------------------------------------
#define S_RXD 18
#define S_TXD 19
const uint32_t BUS_BAUD = 1000000;

// Servo direction table (matches SCServo DIR usage)
const int DIR_TABLE[13] = {0,
  +1, +1, +1,
  -1, -1, -1,
  +1, +1, +1,
  -1, -1, -1
};

const uint8_t SHOULDER_IDS[4] = {1, 4, 7, 10};
const uint8_t THIGH_IDS[4]    = {2, 5, 8, 11};
const uint8_t CALF_IDS[4]     = {3, 6, 9, 12};

// RL -> physical sign correction (thighs & calves inverted vs. sim)
const float JOINT_SIGN[12] = {
  1.0f, 1.0f, 1.0f, 1.0f,
 -1.0f,-1.0f,-1.0f,-1.0f,
 -1.0f,-1.0f,-1.0f,-1.0f
};

// --- Control parameters -----------------------------------------------------
const float DEFAULT_POSE[12] = {
  0.0f, 0.0f, 0.0f, 0.0f,
  0.3f, 0.3f, 0.3f, 0.3f,
 -0.75f,-0.75f,-0.75f,-0.75f
};

const float SHOULDER_LIMIT_DEG = 30.0f;
const float LEG_LIMIT_DEG      = 90.0f;

const uint16_t SERVO_SPEED = 1600;
const uint16_t SERVO_ACC   = 100;
const int SERVO_MID        = 2047;
const float UNITS_PER_DEG  = 4096.0f / 360.0f;

const unsigned long TELEMETRY_PERIOD_MS = 50;   // 20 Hz
const unsigned long COMMAND_TIMEOUT_MS  = 25000;  // watchdog

// --- State ------------------------------------------------------------------
bool streaming_enabled = false;
unsigned long last_command_ms = 0;
unsigned long last_telemetry_ms = 0;
float latest_targets[12];  // stored (RL-oriented) radians

struct ServoDiag {
  bool ok;
  float pos_rad;   // RL-oriented radians
  int load;        // -1000 .. 1000
  int current_mA;
  int temp_C;
};

// --- Helpers ----------------------------------------------------------------
inline int degToPos(uint8_t id, float deg) {
  return SERVO_MID + static_cast<int>(DIR_TABLE[id] * deg * UNITS_PER_DEG + 0.5f);
}

inline float clamp(float val, float min_val, float max_val) {
  return fminf(fmaxf(val, min_val), max_val);
}

void radiansToServoTargets(const float input_rad[12], int16_t out_positions[12]) {
  for (int i = 0; i < 12; ++i) {
    float physical_rad = JOINT_SIGN[i] * input_rad[i];
    float deg = physical_rad * 180.0f / M_PI;
    float clamped_deg;
    if (i < 4) clamped_deg = clamp(deg, -SHOULDER_LIMIT_DEG, SHOULDER_LIMIT_DEG);
    else       clamped_deg = clamp(deg, -LEG_LIMIT_DEG, LEG_LIMIT_DEG);
    if (fabsf(deg - clamped_deg) > 0.05f) {
      Serial.printf("CLAMP joint %d: %.2f deg -> %.2f deg\n", i, deg, clamped_deg);
    }
    uint8_t id;
    if (i < 4) id = SHOULDER_IDS[i];
    else if (i < 8) id = THIGH_IDS[i - 4];
    else id = CALF_IDS[i - 8];
    out_positions[i] = degToPos(id, clamped_deg);
  }
}

void commandPose(const float pose_rad[12]) {
  int16_t targets[12];
  uint16_t speeds[12];
  uint8_t accs[12];
  radiansToServoTargets(pose_rad, targets);
  uint8_t ids[12];
  for (int i = 0; i < 12; ++i) {
    if (i < 4) ids[i] = SHOULDER_IDS[i];
    else if (i < 8) ids[i] = THIGH_IDS[i - 4];
    else ids[i] = CALF_IDS[i - 8];
    speeds[i] = SERVO_SPEED;
    accs[i] = SERVO_ACC;
  }
  st.SyncWritePosEx(ids, 12, targets, speeds, accs);
}

void sendHandshake() {
  Serial.println("ARDUINO_READY");
}

ServoDiag readServoDiag(int index) {
  ServoDiag d{};
  uint8_t id;
  if (index < 4) id = SHOULDER_IDS[index];
  else if (index < 8) id = THIGH_IDS[index - 4];
  else id = CALF_IDS[index - 8];

  int res = st.FeedBack(id);
  d.ok = (res >= 0);
  if (!d.ok) {
    d.pos_rad = latest_targets[index];
    d.load = d.current_mA = d.temp_C = 0;
    return d;
  }
  int pos_units = st.ReadPos(-1);
  int load_raw  = st.ReadLoad(-1);
  int current   = st.ReadCurrent(-1);
  int temp      = st.ReadTemper(-1);

  float deg = DIR_TABLE[id] * (pos_units - SERVO_MID) / UNITS_PER_DEG;
  float rad = deg * M_PI / 180.0f;
  d.pos_rad = rad * JOINT_SIGN[index];
  d.load = load_raw;
  d.current_mA = current;
  d.temp_C = temp;
  return d;
}

void sendTelemetry() {
  unsigned long now = millis();
  if (now - last_telemetry_ms < TELEMETRY_PERIOD_MS) return;
  last_telemetry_ms = now;

  ServoDiag diag[12];
  for (int i = 0; i < 12; ++i) {
    diag[i] = readServoDiag(i);
  }

  Serial.print("TELEM,");
  Serial.print(now);
  Serial.print(',');
  for (int i = 0; i < 12; ++i) {
    Serial.print(diag[i].pos_rad, 4);
    Serial.print(',');
  }
  for (int i = 0; i < 12; ++i) {
    Serial.print(diag[i].load);
    Serial.print(',');
  }
  for (int i = 0; i < 12; ++i) {
    Serial.print(diag[i].current_mA);
    Serial.print(',');
  }
  for (int i = 0; i < 12; ++i) {
    Serial.print(diag[i].temp_C);
    if (i < 11) Serial.print(',');
  }
  Serial.println();
}

void applyStop() {
  streaming_enabled = false;
  commandPose(DEFAULT_POSE);
  memcpy(latest_targets, DEFAULT_POSE, sizeof(latest_targets));
  last_command_ms = millis();
  Serial.println("STATUS,STOPPED");
}

bool parseFloatList(const String& data, float out[12]) {
  int start = data.indexOf('[');
  int end = data.indexOf(']');
  if (start == -1 || end == -1 || end <= start) return false;
  String inner = data.substring(start + 1, end);
  for (int i = 0; i < 12; ++i) {
    int comma = inner.indexOf(',');
    String token;
    if (comma == -1) {
      token = inner;
    } else {
      token = inner.substring(0, comma);
      inner = inner.substring(comma + 1);
    }
    token.trim();
    if (token.length() == 0) return false;
    out[i] = token.toFloat();
  }
  return true;
}

void processMessage(const String& msg) {
  if (msg.equalsIgnoreCase("START")) {
    streaming_enabled = true;
    last_command_ms = millis();
    Serial.println("STATUS,STARTED");
    return;
  }
  if (msg.equalsIgnoreCase("STOP")) {
    applyStop();
    return;
  }
  if (msg.equalsIgnoreCase("PING")) {
    Serial.println("PONG");
    return;
  }
  if (msg.indexOf('[') != -1) {
    float parsed[12];
    if (!parseFloatList(msg, parsed)) {
      Serial.println("ERROR,BAD_FORMAT");
      return;
    }
    memcpy(latest_targets, parsed, sizeof(parsed));
    if (streaming_enabled) {
      commandPose(parsed);
      last_command_ms = millis();
    }
    return;
  }
  if (msg.equalsIgnoreCase("READY?")) {
    sendHandshake();
    return;
  }
  Serial.println("ERROR,UNKNOWN_MSG");
}

void handleSerial() {
  static String buffer = "";
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '#') {
      buffer.trim();
      if (buffer.length() > 0) {
        processMessage(buffer);
      }
      buffer = "";
    } else if (c != '\r' && c != '\n') {
      buffer += c;
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  Serial1.begin(BUS_BAUD, SERIAL_8N1, S_RXD, S_TXD);
  st.pSerial = &Serial1;
  delay(500);

  Serial.println("BOOTING");
  commandPose(DEFAULT_POSE);
  memcpy(latest_targets, DEFAULT_POSE, sizeof(latest_targets));
  last_command_ms = millis();
  last_telemetry_ms = millis();

  sendHandshake();
  Serial.println("STATUS,IDLE");
}

void loop() {
  handleSerial();
  unsigned long now = millis();
  if (streaming_enabled && (now - last_command_ms) > COMMAND_TIMEOUT_MS) {
    Serial.println("WARN,TIMEOUT");
    applyStop();
  }
  sendTelemetry();
}
