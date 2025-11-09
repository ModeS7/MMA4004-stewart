/*
  Stewart Platform Controller v2 - Pixy2 + IMU + Servo Integration
  Teensy 4.1

  Combines ball tracking (Pixy2) with IMU orientation sensing for integrated tilt correction.

  Hardware:
  - Pixy2 Camera: Ball position tracking (SPI interface)
  - LSM303DLHC: Accelerometer (~1344 Hz) + Magnetometer (~220 Hz)
  - L3GD20H: Gyroscope (~800 Hz)
  - Pololu Maestro: 6-channel servo controller

  Communication:
  - USB Serial (Teensy ↔ PC): 200000 baud for data streaming
  - Serial1 (Teensy ↔ Maestro): 250000 baud for servo commands

  Serial Output Format:
  - BALL:timestamp,x,y,detected,error_x,error_y  (50Hz Pixy2 ball data)
  - A:timestamp_us,ax,ay,az  (Accelerometer)
  - G:timestamp_us,gx,gy,gz  (Gyroscope)
  - M:timestamp_us,mx,my,mz  (Magnetometer)
  - ACK:OK                   (Command acknowledgment)

  Serial Input Commands:
  - "angle0,angle1,angle2,angle3,angle4,angle5\n"  (Move servos)
  - "SPD:value\n"   (Set servo speed 0-255)
  - "ACC:value\n"   (Set servo acceleration 0-255)
  - "STATS\n"       (Performance statistics)

  Design: control_v1 (Pixy2+servo) + IMU_control (IMU+servo) = control_v2 (all-in-one)
*/

#include <Wire.h>
#include <Pixy2SPI_SS.h>
#include "PololuMaestro.h"

// ===== I2C ADDRESSES =====
#define LSM303_ACCEL_ADDR 0x19
#define LSM303_MAG_ADDR   0x1E
#define L3GD20_ADDR       0x6B

// ===== REGISTER ADDRESSES =====
#define LSM303_OUT_X_L_A  0x28
#define LSM303_OUT_X_H_M  0x03
#define L3GD20_OUT_X_L    0x28

// ===== HARDWARE INTERFACES =====
#define USB_SERIAL Serial
#define MAESTRO_SERIAL Serial1

Pixy2SPI_SS pixy;
MicroMaestro maestro(MAESTRO_SERIAL);

// ===== PIXY2 CONSTANTS =====
const float ORIGIN_X = 145.0;
const float ORIGIN_Y = 102.0;

// ===== IMU DATA BUFFERS =====
int16_t accelData[3];
int16_t gyroData[3];
int16_t magData[3];

// ===== AVERAGING BUFFERS =====
#define ACCEL_AVG_COUNT 5
#define GYRO_AVG_COUNT 5
#define MAG_AVG_COUNT 2

int32_t accelSum[3] = {0, 0, 0};
int32_t gyroSum[3] = {0, 0, 0};
int32_t magSum[3] = {0, 0, 0};
uint8_t accelAvgIndex = 0;
uint8_t gyroAvgIndex = 0;
uint8_t magAvgIndex = 0;

// ===== IMU TIMING =====
unsigned long lastAccelTime = 0;
unsigned long lastGyroTime = 0;
unsigned long lastMagTime = 0;
unsigned long accelCount = 0;
unsigned long gyroCount = 0;
unsigned long magCount = 0;
unsigned long accelStartTime = 0;
unsigned long gyroStartTime = 0;
unsigned long magStartTime = 0;

// Target intervals for native rates
const unsigned long ACCEL_INTERVAL_US = 744;   // ~1344 Hz target
const unsigned long GYRO_INTERVAL_US = 1250;   // ~800 Hz target
const unsigned long MAG_INTERVAL_US = 4545;    // ~220 Hz target

// Rate reporting
unsigned long lastRateReport = 0;
const unsigned long RATE_REPORT_INTERVAL = 2000000; // 2 seconds

// ===== PIXY2 TIMING =====
const unsigned long PIXY_INTERVAL = 20;  // 50Hz ball position updates
unsigned long lastPixyRead = 0;
unsigned long pixyReadCount = 0;

// ===== SERVO CONTROL CONSTANTS =====
// PWM range in quarter-microseconds (Maestro units)
const float abs_0 = 2000;   // 500 µs (0°)
const float abs_90 = 10000;  // 2500 µs (180°)

// Servo ranges (CCW direction) - full 180° range
float range[6][2] = {
  {-90, 90}, {90, -90},
  {-90, 90}, {90, -90},
  {-90, 90}, {90, -90}
};

// Safety limit for angle validation
const float MAX_ANGLE_LIMIT = 80.0;

// Servo offsets (calibration)
float offset[6] = {-1.17, 4.23, -4.34, -0.22, -0.21, 4.55};

// Global offset adjustment
float global_offset = 8.47;

float theta[6] = {0, 0, 0, 0, 0, 0};

int servoSpeed = 0;           // 0 = UNLIMITED
int servoAcceleration = 0;    // 0 = NO RAMPING

// ===== SERIAL INPUT BUFFER =====
const int MAX_CMD_LENGTH = 64;
char inputBuffer[MAX_CMD_LENGTH];
int bufferIndex = 0;

// ===== PERFORMANCE MONITORING =====
elapsedMicros loopTimer;
uint32_t maxLoopTime = 0;
unsigned long servoCommandCount = 0;
unsigned long errorCount = 0;
unsigned long startTime = 0;

// ===== LED INDICATOR =====
const int LED_PIN = LED_BUILTIN;
unsigned long lastBlink = 0;
bool ledState = false;

// ===== SETUP =====
void setup() {
  pinMode(LED_PIN, OUTPUT);

  // USB Serial (200kbaud)
  USB_SERIAL.begin(200000);
  unsigned long waitStart = millis();
  while (!USB_SERIAL && millis() - waitStart < 2000) {
    delay(10);
  }

  // Maestro Serial (250kbaud)
  MAESTRO_SERIAL.begin(250000);

  USB_SERIAL.println("INIT:Controller v2 - Pixy2 + IMU + Servo");

  // Initialize Pixy2
  int pixy_result = pixy.init();
  if (pixy_result < 0) {
    USB_SERIAL.println("ERROR:Pixy2 init failed");
    // Continue anyway - IMU will still work
  } else {
    USB_SERIAL.println("INFO:Pixy2 initialized");
  }

  // Initialize I2C for IMU
  Wire.begin();
  Wire.setClock(400000); // 400kHz I2C

  delay(100);

  // Configure LSM303DLHC Accelerometer: 1344 Hz
  writeRegister(LSM303_ACCEL_ADDR, 0x20, 0x97);

  // Configure LSM303DLHC Magnetometer: 220 Hz, continuous mode
  writeRegister(LSM303_MAG_ADDR, 0x00, 0x1C);
  writeRegister(LSM303_MAG_ADDR, 0x02, 0x00);

  // Configure L3GD20H Gyroscope: 800 Hz, ±2000 dps range
  writeRegister(L3GD20_ADDR, 0x20, 0xFF);
  writeRegister(L3GD20_ADDR, 0x23, 0x20);

  delay(100);

  // Set servo speed/acceleration
  for (int i = 0; i < 6; i++) {
    maestro.setSpeed(i, servoSpeed);
    maestro.setAcceleration(i, servoAcceleration);
  }

  USB_SERIAL.println("READY:All systems online");
  USB_SERIAL.println("FORMAT:BALL:timestamp,x,y,detected,error_x,error_y");
  USB_SERIAL.println("FORMAT:A:timestamp_us,ax,ay,az");
  USB_SERIAL.println("FORMAT:G:timestamp_us,gx,gy,gz");
  USB_SERIAL.println("FORMAT:M:timestamp_us,mx,my,mz");
  USB_SERIAL.println();

  startTime = millis();
  accelStartTime = micros();
  gyroStartTime = micros();
  magStartTime = micros();
  lastRateReport = micros();

  // Clear buffers
  while (USB_SERIAL.available()) {
    USB_SERIAL.read();
  }

  delay(100);
}

// ===== MAIN LOOP =====
void loop() {
  loopTimer = 0;
  unsigned long now_ms = millis();
  unsigned long now_us = micros();

  // Read Pixy2 ball position at 50Hz
  if (now_ms - lastPixyRead >= PIXY_INTERVAL) {
    lastPixyRead = now_ms;
    readAndSendBallPosition(now_ms);
    pixyReadCount++;
  }

  // Read accelerometer at ~1344 Hz, average and send every 5 samples
  if (now_us - lastAccelTime >= ACCEL_INTERVAL_US) {
    readAccel();

    accelSum[0] += accelData[0];
    accelSum[1] += accelData[1];
    accelSum[2] += accelData[2];
    accelAvgIndex++;

    if (accelAvgIndex >= ACCEL_AVG_COUNT) {
      int16_t avgData[3];
      avgData[0] = accelSum[0] / ACCEL_AVG_COUNT;
      avgData[1] = accelSum[1] / ACCEL_AVG_COUNT;
      avgData[2] = accelSum[2] / ACCEL_AVG_COUNT;
      sendAccel(now_us, avgData);

      accelSum[0] = 0;
      accelSum[1] = 0;
      accelSum[2] = 0;
      accelAvgIndex = 0;
      accelCount++;
    }

    lastAccelTime = now_us;
  }

  // Read gyroscope at ~800 Hz, average and send every 5 samples
  if (now_us - lastGyroTime >= GYRO_INTERVAL_US) {
    readGyro();

    gyroSum[0] += gyroData[0];
    gyroSum[1] += gyroData[1];
    gyroSum[2] += gyroData[2];
    gyroAvgIndex++;

    if (gyroAvgIndex >= GYRO_AVG_COUNT) {
      int16_t avgData[3];
      avgData[0] = gyroSum[0] / GYRO_AVG_COUNT;
      avgData[1] = gyroSum[1] / GYRO_AVG_COUNT;
      avgData[2] = gyroSum[2] / GYRO_AVG_COUNT;
      sendGyro(now_us, avgData);

      gyroSum[0] = 0;
      gyroSum[1] = 0;
      gyroSum[2] = 0;
      gyroAvgIndex = 0;
      gyroCount++;
    }

    lastGyroTime = now_us;
  }

  // Read magnetometer at ~220 Hz, average and send every 2 samples
  if (now_us - lastMagTime >= MAG_INTERVAL_US) {
    readMag();

    magSum[0] += magData[0];
    magSum[1] += magData[1];
    magSum[2] += magData[2];
    magAvgIndex++;

    if (magAvgIndex >= MAG_AVG_COUNT) {
      int16_t avgData[3];
      avgData[0] = magSum[0] / MAG_AVG_COUNT;
      avgData[1] = magSum[1] / MAG_AVG_COUNT;
      avgData[2] = magSum[2] / MAG_AVG_COUNT;
      sendMag(now_us, avgData);

      magSum[0] = 0;
      magSum[1] = 0;
      magSum[2] = 0;
      magAvgIndex = 0;
      magCount++;
    }

    lastMagTime = now_us;
  }

  // Report sampling rates every 2 seconds
  if (now_us - lastRateReport >= RATE_REPORT_INTERVAL) {
    reportSamplingRates();
    lastRateReport = now_us;
  }

  // Check for servo commands (non-blocking)
  checkSerialCommandsNonBlocking();

  // LED heartbeat
  if (now_ms - lastBlink > 500) {
    lastBlink = now_ms;
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState);
  }

  // Track max loop time
  uint32_t currentLoopTime = loopTimer;
  if (currentLoopTime > maxLoopTime) {
    maxLoopTime = currentLoopTime;
  }

  // Clear buffer if overfull
  if (USB_SERIAL.available() > 800) {
    USB_SERIAL.clear();
    errorCount++;
  }
}

// ===== PIXY2 BALL READING =====
void readAndSendBallPosition(unsigned long timestamp) {
  if (!USB_SERIAL || USB_SERIAL.availableForWrite() < 64) {
    return;
  }

  int8_t num_blocks = pixy.ccc.getBlocks(false, CCC_SIG1, 1);

  float ball_x, ball_y;
  bool detected;

  if (num_blocks > 0 && pixy.ccc.numBlocks == 1) {
    ball_x = pixy.ccc.blocks[0].m_x;
    ball_y = pixy.ccc.blocks[0].m_y;
    detected = true;
  } else {
    ball_x = 0.0;
    ball_y = 0.0;
    detected = false;
  }

  float error_x = ball_x - ORIGIN_X;
  float error_y = ORIGIN_Y - ball_y;

  // Send data (same format as control_v1)
  USB_SERIAL.print("BALL:");
  USB_SERIAL.print((timestamp - startTime) / 1000.0, 3);
  USB_SERIAL.print(",");
  USB_SERIAL.print(ball_x, 2);
  USB_SERIAL.print(",");
  USB_SERIAL.print(ball_y, 2);
  USB_SERIAL.print(",");
  USB_SERIAL.print(detected ? "1" : "0");
  USB_SERIAL.print(",");
  USB_SERIAL.print(error_x, 2);
  USB_SERIAL.print(",");
  USB_SERIAL.println(error_y, 2);

  USB_SERIAL.flush();
}

// ===== I2C WRITE =====
void writeRegister(uint8_t addr, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

// ===== ACCELEROMETER =====
void readAccel() {
  Wire.beginTransmission(LSM303_ACCEL_ADDR);
  Wire.write(LSM303_OUT_X_L_A | 0x80); // Auto-increment
  Wire.endTransmission();

  Wire.requestFrom(LSM303_ACCEL_ADDR, 6);

  uint8_t xlo = Wire.read();
  uint8_t xhi = Wire.read();
  uint8_t ylo = Wire.read();
  uint8_t yhi = Wire.read();
  uint8_t zlo = Wire.read();
  uint8_t zhi = Wire.read();

  // 12-bit left-justified in 16-bit (shift right 4 bits)
  accelData[0] = (int16_t)(xhi << 8 | xlo) >> 4;
  accelData[1] = (int16_t)(yhi << 8 | ylo) >> 4;
  accelData[2] = (int16_t)(zhi << 8 | zlo) >> 4;
}

void sendAccel(unsigned long timestamp_us, int16_t* data) {
  char buf[64];
  snprintf(buf, sizeof(buf), "A:%lu,%d,%d,%d", timestamp_us, data[0], data[1], data[2]);
  USB_SERIAL.println(buf);
}

// ===== GYROSCOPE =====
void readGyro() {
  Wire.beginTransmission(L3GD20_ADDR);
  Wire.write(L3GD20_OUT_X_L | 0x80); // Auto-increment
  Wire.endTransmission();

  Wire.requestFrom(L3GD20_ADDR, 6);

  uint8_t xlo = Wire.read();
  uint8_t xhi = Wire.read();
  uint8_t ylo = Wire.read();
  uint8_t yhi = Wire.read();
  uint8_t zlo = Wire.read();
  uint8_t zhi = Wire.read();

  gyroData[0] = (int16_t)(xhi << 8 | xlo);
  gyroData[1] = (int16_t)(yhi << 8 | ylo);
  gyroData[2] = (int16_t)(zhi << 8 | zlo);
}

void sendGyro(unsigned long timestamp_us, int16_t* data) {
  char buf[64];
  snprintf(buf, sizeof(buf), "G:%lu,%d,%d,%d", timestamp_us, data[0], data[1], data[2]);
  USB_SERIAL.println(buf);
}

// ===== MAGNETOMETER =====
void readMag() {
  Wire.beginTransmission(LSM303_MAG_ADDR);
  Wire.write(LSM303_OUT_X_H_M); // Start at X high byte
  Wire.endTransmission();

  Wire.requestFrom(LSM303_MAG_ADDR, 6);

  // Magnetometer stores data as big-endian (MSB first)
  // Note: LSM303 has unusual axis order: X, Z, Y
  uint8_t xhi = Wire.read();
  uint8_t xlo = Wire.read();
  uint8_t zhi = Wire.read();
  uint8_t zlo = Wire.read();
  uint8_t yhi = Wire.read();
  uint8_t ylo = Wire.read();

  magData[0] = (int16_t)(xhi << 8 | xlo);
  magData[1] = (int16_t)(yhi << 8 | ylo);
  magData[2] = (int16_t)(zhi << 8 | zlo);
}

void sendMag(unsigned long timestamp_us, int16_t* data) {
  char buf[64];
  snprintf(buf, sizeof(buf), "M:%lu,%d,%d,%d", timestamp_us, data[0], data[1], data[2]);
  USB_SERIAL.println(buf);
}

// ===== RATE REPORTING =====
void reportSamplingRates() {
  unsigned long now = micros();
  unsigned long accelElapsed = now - accelStartTime;
  unsigned long gyroElapsed = now - gyroStartTime;
  unsigned long magElapsed = now - magStartTime;

  float accelHz = (accelCount * 1000000.0) / accelElapsed;
  float gyroHz = (gyroCount * 1000000.0) / gyroElapsed;
  float magHz = (magCount * 1000000.0) / magElapsed;

  USB_SERIAL.print("RATE:Accel=");
  USB_SERIAL.print(accelHz, 2);
  USB_SERIAL.print(" Hz, Gyro=");
  USB_SERIAL.print(gyroHz, 2);
  USB_SERIAL.print(" Hz, Mag=");
  USB_SERIAL.print(magHz, 2);
  USB_SERIAL.println(" Hz");
}

// ===== NON-BLOCKING SERIAL COMMAND PROCESSING =====
void checkSerialCommandsNonBlocking() {
  while (USB_SERIAL.available() > 0) {
    char c = USB_SERIAL.read();

    if (c == '\n' || c == '\r') {
      if (bufferIndex > 0) {
        inputBuffer[bufferIndex] = '\0';
        processCommand(inputBuffer);
        bufferIndex = 0;
      }
    }
    else if (bufferIndex < MAX_CMD_LENGTH - 1) {
      inputBuffer[bufferIndex++] = c;
    }
    else {
      bufferIndex = 0;
      errorCount++;
      USB_SERIAL.println("ERROR:Command too long");
      break;
    }
  }
}

// ===== COMMAND PROCESSING =====
void processCommand(char* cmd) {
  // Stats request
  if (strcmp(cmd, "STATS") == 0) {
    sendPerformanceStats();
    return;
  }

  // Speed command
  if (strncmp(cmd, "SPD:", 4) == 0) {
    servoSpeed = constrain(atoi(cmd + 4), 0, 255);
    for (int i = 0; i < 6; i++) {
      maestro.setSpeed(i, servoSpeed);
    }
    USB_SERIAL.print("ACK:Speed=");
    USB_SERIAL.println(servoSpeed);
    return;
  }

  // Acceleration command
  if (strncmp(cmd, "ACC:", 4) == 0) {
    servoAcceleration = constrain(atoi(cmd + 4), 0, 255);
    for (int i = 0; i < 6; i++) {
      maestro.setAcceleration(i, servoAcceleration);
    }
    USB_SERIAL.print("ACK:Accel=");
    USB_SERIAL.println(servoAcceleration);
    return;
  }

  // Parse servo angles
  parseAndExecuteAngles(cmd);
}

// ===== ANGLE PARSING =====
void parseAndExecuteAngles(char* cmd) {
  float angles[6];
  int angleCount = 0;

  char* token = strtok(cmd, ",");

  while (token != NULL && angleCount < 6) {
    angles[angleCount] = atof(token);
    angleCount++;
    token = strtok(NULL, ",");
  }

  // Validate
  if (angleCount != 6) {
    USB_SERIAL.print("ERROR:Expected 6 angles, got ");
    USB_SERIAL.println(angleCount);
    errorCount++;
    return;
  }

  // Check limits
  for (int i = 0; i < 6; i++) {
    if (abs(angles[i]) > MAX_ANGLE_LIMIT || isnan(angles[i])) {
      USB_SERIAL.print("ERROR:Invalid angle[");
      USB_SERIAL.print(i);
      USB_SERIAL.print("]=");
      USB_SERIAL.println(angles[i]);
      errorCount++;
      return;
    }
  }

  // Update and move immediately
  for (int i = 0; i < 6; i++) {
    theta[i] = angles[i];
  }

  moveServos();
  servoCommandCount++;

  USB_SERIAL.println("ACK:OK");
}

// ===== SERVO MOVEMENT =====
void moveServos() {
  for (int i = 0; i < 6; i++) {
    float pos = theta[i] + offset[i] + global_offset;
    pos = map_float(pos, range[i][0], range[i][1], abs_0, abs_90);
    pos = constrain(pos, abs_0, abs_90);

    maestro.setSpeed(i, servoSpeed);
    maestro.setAcceleration(i, servoAcceleration);
    maestro.setTarget(i, (uint16_t)pos);
  }
}

// ===== UTILITY FUNCTIONS =====
float map_float(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

void sendPerformanceStats() {
  float uptime = (millis() - startTime) / 1000.0;

  USB_SERIAL.println();
  USB_SERIAL.println("======== PERFORMANCE STATS ========");
  USB_SERIAL.print("Controller v2 | USB=200kbaud | Maestro=250kbaud");
  USB_SERIAL.print(" | Speed=");
  USB_SERIAL.print(servoSpeed);
  USB_SERIAL.print(" | Accel=");
  USB_SERIAL.println(servoAcceleration);

  USB_SERIAL.print("Uptime: ");
  USB_SERIAL.print(uptime, 1);
  USB_SERIAL.println(" s");

  USB_SERIAL.print("Pixy reads: ");
  USB_SERIAL.print(pixyReadCount);
  USB_SERIAL.print(" (");
  USB_SERIAL.print(pixyReadCount / uptime, 1);
  USB_SERIAL.println(" Hz)");

  USB_SERIAL.print("IMU samples (A/G/M): ");
  USB_SERIAL.print(accelCount);
  USB_SERIAL.print(" / ");
  USB_SERIAL.print(gyroCount);
  USB_SERIAL.print(" / ");
  USB_SERIAL.println(magCount);

  USB_SERIAL.print("Servo cmds: ");
  USB_SERIAL.print(servoCommandCount);
  USB_SERIAL.print(" (");
  USB_SERIAL.print(servoCommandCount / uptime, 1);
  USB_SERIAL.println(" Hz)");

  USB_SERIAL.print("Errors: ");
  USB_SERIAL.println(errorCount);

  USB_SERIAL.print("Max loop: ");
  USB_SERIAL.print(maxLoopTime);
  USB_SERIAL.println(" µs");

  USB_SERIAL.print("Serial buffer: ");
  USB_SERIAL.print(USB_SERIAL.available());
  USB_SERIAL.println(" bytes");

  USB_SERIAL.println("\nCurrent angles:");
  for (int i = 0; i < 6; i++) {
    USB_SERIAL.print("  S");
    USB_SERIAL.print(i);
    USB_SERIAL.print(": ");
    USB_SERIAL.println(theta[i], 2);
  }
  USB_SERIAL.println("===================================");
  USB_SERIAL.println();
}
