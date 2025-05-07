/*
 * ChainMovementTest.ino
 * ------------------------------------------------------------
 * Runs the standard “mid → +45° → mid → –45° → mid” sequence
 * on every servo in the daisy‑chain, one after another.
 *
 * Library / pin / baud settings are identical to the other
 * STSCL examples that already work on your setup.
 * ------------------------------------------------------------
 */

#include <SCServo.h>

SMS_STS st;

// ---------- hardware settings (unchanged) ----------
#define S_RXD        18          // GPIO for RX
#define S_TXD        19          // GPIO for TX
#define BUS_BAUD 1000000         // UART to the servos
// ----------------------------------------------------

// ---------- chain definition ----------
const uint8_t SERVO_IDS[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};   // edit if you add more
const uint8_t NUM_SERVOS  = sizeof(SERVO_IDS);
// --------------------------------------

// ---------- motion parameters ----------
const float  DEG_PER_UNIT = 360.0 / 4096.0;   // ST‑series resolution
const int    MID_POS      = 2047;             // electrical centre
const int    SPEED        = 800;              // ≈ 20 rpm; dial up/down
const int    ACC          = 100;              // smooth accel
const uint16_t PAUSE_MS   = 800;              // pause between moves
const uint16_t BETWEEN_MS = 1200;             // pause between servos
// ---------------------------------------

static inline int degToPos(float deg)
{
    return MID_POS + int(deg / DEG_PER_UNIT + 0.5f);
}

static void moveAndWait(uint8_t id, int position)
{
    st.WritePosEx(id, position, SPEED, ACC);
    delay(PAUSE_MS);
}

void setup()
{
    Serial.begin(115200);
    while (!Serial) { }

    Serial1.begin(BUS_BAUD, SERIAL_8N1, S_RXD, S_TXD);
    st.pSerial = &Serial1;
    delay(500);

    Serial.println(F("\n--- Chain Movement Test ---"));
    Serial.printf("Testing %d servos at IDs: ", NUM_SERVOS);
    for (uint8_t i = 0; i < NUM_SERVOS; ++i) {
        Serial.printf("%d ", SERVO_IDS[i]);
    }
    Serial.println("\n");
}

void loop()
{
    for (uint8_t idx = 0; idx < NUM_SERVOS; ++idx) {

        uint8_t id   = SERVO_IDS[idx];
        int  fwd45   = degToPos(+45.0f);
        int  back45  = degToPos(-45.0f);

        Serial.printf("ID %d  → centre\n", id);
        moveAndWait(id, MID_POS);

        Serial.printf("ID %d  → +45°\n", id);
        moveAndWait(id, fwd45);

        Serial.printf("ID %d  → centre\n", id);
        moveAndWait(id, MID_POS);

        Serial.printf("ID %d  → –45°\n", id);
        moveAndWait(id, back45);

        Serial.printf("ID %d  → centre\n\n", id);
        moveAndWait(id, MID_POS);

        delay(BETWEEN_MS);               // let everything settle
    }

    Serial.println(F("All servos tested — looping again in 5 s.\n"));
    delay(5000);                         // wait then start over
}
