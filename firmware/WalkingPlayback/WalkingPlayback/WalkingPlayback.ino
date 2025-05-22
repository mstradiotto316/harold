/* Python-driven action playback that receives joint commands over serial */
#include <SCServo.h>
SMS_STS st;

#define S_RXD 18
#define S_TXD 19
const uint32_t BUS_BAUD = 1000000;
const uint32_t SERIAL_BAUD = 115200;

// Safety limits for robot movements
const float MAX_SHOULDER_ANGLE = 30.0f;  // Maximum shoulder angle from center
const float MAX_LEG_ANGLE = 90.0f;       // Maximum thigh/calf angle from center

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

// Function to clamp a value between min and max
float clamp(float value, float min_val, float max_val) {
  if (value < min_val) return min_val;
  if (value > max_val) return max_val;
  return value;
}

void writeAll(float angles[12]) {
  // For each leg
  for (int l=0; l<4; ++l) {
    // Get the joint indices for this leg
    int shoulder_idx = l * 3;
    int thigh_idx = shoulder_idx + 1;
    int calf_idx = shoulder_idx + 2;
    
    // Clamp and write each joint with appropriate limits
    float shoulder_angle = clamp(angles[shoulder_idx], -MAX_SHOULDER_ANGLE, MAX_SHOULDER_ANGLE);
    float thigh_angle = clamp(angles[thigh_idx], -MAX_LEG_ANGLE, MAX_LEG_ANGLE);
    float calf_angle = clamp(angles[calf_idx], -MAX_LEG_ANGLE, MAX_LEG_ANGLE);
    
    // Debug output for angles
    Serial.printf("Leg %d - Shoulder: %.1f°, Thigh: %.1f°, Calf: %.1f°\n", 
                 l, shoulder_angle, thigh_angle, calf_angle);
    
    // Set the new positions
    st.WritePosEx(LEG[l][0], degToPos(LEG[l][0], shoulder_angle), SPEED, ACC);
    st.WritePosEx(LEG[l][1], degToPos(LEG[l][1], thigh_angle), SPEED, ACC);
    st.WritePosEx(LEG[l][2], degToPos(LEG[l][2], calf_angle), SPEED, ACC);
  }
}

void straightLegs() {
  float zero_angles[12] = {0};
  writeAll(zero_angles);
}

void setup() {
  Serial.begin(SERIAL_BAUD); while (!Serial) {}
  Serial1.begin(BUS_BAUD, SERIAL_8N1, S_RXD, S_TXD); st.pSerial = &Serial1;
  delay(2000);

  Serial.println("\n== Python-driven Action Playback ==");
  Serial.println("Ready to receive commands...");
  straightLegs();  // Start in straight position
}

void loop() {
  if (Serial.available() >= 48) {  // Expect 12 floats (4 bytes each)
    float angles[12];
    
    // Read all angles
    for (int i = 0; i < 12; i++) {
      Serial.readBytes((char*)&angles[i], 4);
    }
    
    // Move to the requested position
    writeAll(angles);
    
    // Send acknowledgment
    Serial.write('A');
  }
} 
