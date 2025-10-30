/*
  Stewart Platform IMU Controller
  Teensy 4.1 - IMU Sensor Fusion Mode

  Reads IMU sensors at native rates:
  - LSM303 Accelerometer: ~1265 Hz
  - LSM303 Magnetometer: ~220 Hz
  - L3GD20 Gyroscope: ~759 Hz

  USB (Teensy ↔ PC): Native USB speed (no baud limit)
  Serial1 (Teensy ↔ Maestro): 250000 baud
*/

#include <Wire.h>
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

MicroMaestro maestro(MAESTRO_SERIAL);

// ===== IMU DATA BUFFERS =====
int16_t accelData[3];
int16_t gyroData[3];
int16_t magData[3];

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

// ===== SERVO CONTROL CONSTANTS =====
// PWM range in quarter-microseconds (Maestro units)
// Full 180° servo range: 500-2500 µs
const float abs_0 = 2000;   // 500 µs (0°)
const float abs_90 = 10000;  // 2500 µs (180°)

// Servo ranges (CCW direction) - full 180° range
float range[6][2] = {
  {-90, 90}, {90, -90},
  {-90, 90}, {90, -90},
  {-90, 90}, {90, -90}
};

// Safety limit for angle validation
const float MAX_ANGLE_LIMIT = 80.0;  // Safety margin (servos can do ±90°)

// Servo offsets (calibration)
float offset[6] = {-5.07, 1.66, -7.22, -1.79, -6.83, 1.61};

// Global offset adjustment (added to all servos)
float global_offset = 8.47;  // Change this value to shift all servos

float theta[6] = {0, 0, 0, 0, 0, 0};

int servoSpeed = 0;           // 0 = UNLIMITED
int servoAcceleration = 0;    // 0 = NO RAMPING

// ===== SERIAL INPUT BUFFER =====
const int MAX_CMD_LENGTH = 64;
char inputBuffer[MAX_CMD_LENGTH];
int bufferIndex = 0;

// ===== PERFORMANCE MONITORING =====
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

  // USB Serial (native speed)
  USB_SERIAL.begin(2000000);
  unsigned long waitStart = millis();
  while (!USB_SERIAL && millis() - waitStart < 2000) {
    delay(10);
  }

  // Maestro Serial
  MAESTRO_SERIAL.begin(250000);

  // Initialize I2C for IMU
  Wire.begin();
  Wire.setClock(400000); // 400kHz I2C

  delay(100);

  USB_SERIAL.println("INIT:Stewart Platform IMU Controller");

  // Configure accelerometer: 1344 Hz
  writeRegister(LSM303_ACCEL_ADDR, 0x20, 0x97);

  // Configure magnetometer: 220 Hz, continuous mode
  writeRegister(LSM303_MAG_ADDR, 0x00, 0x1C);  // CRA_REG_M: 220 Hz
  writeRegister(LSM303_MAG_ADDR, 0x02, 0x00);  // MR_REG_M: Continuous conversion

  // Configure gyroscope: 800 Hz
  writeRegister(L3GD20_ADDR, 0x20, 0xFF);

  delay(100);

  // Set servo speed/acceleration
  for (int i = 0; i < 6; i++) {
    maestro.setSpeed(i, servoSpeed);
    maestro.setAcceleration(i, servoAcceleration);
  }

  USB_SERIAL.println("READY:IMU + Servo control online");
  USB_SERIAL.println("FORMAT:A:timestamp_us,ax,ay,az");
  USB_SERIAL.println("FORMAT:G:timestamp_us,gx,gy,gz");
  USB_SERIAL.println("FORMAT:M:timestamp_us,mx,my,mz");

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
  unsigned long now = micros();

  // Read accelerometer at ~1265 Hz
  if (now - lastAccelTime >= ACCEL_INTERVAL_US) {
    readAccel();
    sendAccel(now);
    lastAccelTime = now;
    accelCount++;
  }

  // Read gyroscope at ~759 Hz
  if (now - lastGyroTime >= GYRO_INTERVAL_US) {
    readGyro();
    sendGyro(now);
    lastGyroTime = now;
    gyroCount++;
  }

  // Read magnetometer at ~220 Hz
  if (now - lastMagTime >= MAG_INTERVAL_US) {
    readMag();
    sendMag(now);
    lastMagTime = now;
    magCount++;
  }

  // Report sampling rates every 2 seconds
  if (now - lastRateReport >= RATE_REPORT_INTERVAL) {
    unsigned long accelElapsed = now - accelStartTime;
    unsigned long gyroElapsed = now - gyroStartTime;
    unsigned long magElapsed = now - magStartTime;

    float accelHz = (accelCount * 1000000.0) / accelElapsed;
    float gyroHz = (gyroCount * 1000000.0) / gyroElapsed;
    float magHz = (magCount * 1000000.0) / magElapsed;

    USB_SERIAL.print("RATE:Accel=");
    USB_SERIAL.print(accelHz, 2);
    USB_SERIAL.print(",Gyro=");
    USB_SERIAL.print(gyroHz, 2);
    USB_SERIAL.print(",Mag=");
    USB_SERIAL.print(magHz, 2);
    USB_SERIAL.println(" Hz");

    lastRateReport = now;
  }

  // Check for servo commands (non-blocking)
  checkSerialCommandsNonBlocking();

  // LED heartbeat (every 500ms)
  unsigned long nowMs = millis();
  if (nowMs - lastBlink > 500) {
    lastBlink = nowMs;
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState);
  }

  // Clear buffer if overfull
  if (USB_SERIAL.available() > 800) {
    USB_SERIAL.clear();
    errorCount++;
  }
}

// ===== IMU I2C FUNCTIONS =====
void writeRegister(uint8_t addr, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

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

  // 12-bit left-justified in 16-bit
  accelData[0] = (int16_t)(xhi << 8 | xlo) >> 4;
  accelData[1] = (int16_t)(yhi << 8 | ylo) >> 4;
  accelData[2] = (int16_t)(zhi << 8 | zlo) >> 4;
}

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

void sendAccel(unsigned long timestamp_us) {
  USB_SERIAL.print("A:");
  USB_SERIAL.print(timestamp_us);
  USB_SERIAL.print(",");
  USB_SERIAL.print(accelData[0]);
  USB_SERIAL.print(",");
  USB_SERIAL.print(accelData[1]);
  USB_SERIAL.print(",");
  USB_SERIAL.println(accelData[2]);
}

void sendGyro(unsigned long timestamp_us) {
  USB_SERIAL.print("G:");
  USB_SERIAL.print(timestamp_us);
  USB_SERIAL.print(",");
  USB_SERIAL.print(gyroData[0]);
  USB_SERIAL.print(",");
  USB_SERIAL.print(gyroData[1]);
  USB_SERIAL.print(",");
  USB_SERIAL.println(gyroData[2]);
}

void readMag() {
  Wire.beginTransmission(LSM303_MAG_ADDR);
  Wire.write(LSM303_OUT_X_H_M); // Start at X high byte (no auto-increment needed for mag)
  Wire.endTransmission();

  Wire.requestFrom(LSM303_MAG_ADDR, 6);

  // Magnetometer stores data as big-endian (MSB first)
  uint8_t xhi = Wire.read();
  uint8_t xlo = Wire.read();
  uint8_t zhi = Wire.read();  // Note: Z comes before Y in LSM303
  uint8_t zlo = Wire.read();
  uint8_t yhi = Wire.read();
  uint8_t ylo = Wire.read();

  magData[0] = (int16_t)(xhi << 8 | xlo);
  magData[1] = (int16_t)(yhi << 8 | ylo);
  magData[2] = (int16_t)(zhi << 8 | zlo);
}

void sendMag(unsigned long timestamp_us) {
  USB_SERIAL.print("M:");
  USB_SERIAL.print(timestamp_us);
  USB_SERIAL.print(",");
  USB_SERIAL.print(magData[0]);
  USB_SERIAL.print(",");
  USB_SERIAL.print(magData[1]);
  USB_SERIAL.print(",");
  USB_SERIAL.println(magData[2]);
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

  // Send servo angles with timestamp for logging
  USB_SERIAL.print("SERVO:");
  USB_SERIAL.print(micros());
  USB_SERIAL.print(",");
  for (int i = 0; i < 6; i++) {
    USB_SERIAL.print(theta[i], 2);
    if (i < 5) USB_SERIAL.print(",");
  }
  USB_SERIAL.println();
}

// ===== UTILITY FUNCTIONS =====
float map_float(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

void sendPerformanceStats() {
  float uptime = (millis() - startTime) / 1000.0;
  unsigned long now = micros();

  float accelHz = (accelCount * 1000000.0) / (now - accelStartTime);
  float gyroHz = (gyroCount * 1000000.0) / (now - gyroStartTime);
  float magHz = (magCount * 1000000.0) / (now - magStartTime);

  USB_SERIAL.println();
  USB_SERIAL.println("======== IMU PERFORMANCE STATS ========");
  USB_SERIAL.print("Uptime: ");
  USB_SERIAL.print(uptime, 1);
  USB_SERIAL.println(" s");

  USB_SERIAL.print("Accel samples: ");
  USB_SERIAL.print(accelCount);
  USB_SERIAL.print(" (");
  USB_SERIAL.print(accelHz, 2);
  USB_SERIAL.println(" Hz)");

  USB_SERIAL.print("Gyro samples: ");
  USB_SERIAL.print(gyroCount);
  USB_SERIAL.print(" (");
  USB_SERIAL.print(gyroHz, 2);
  USB_SERIAL.println(" Hz)");

  USB_SERIAL.print("Mag samples: ");
  USB_SERIAL.print(magCount);
  USB_SERIAL.print(" (");
  USB_SERIAL.print(magHz, 2);
  USB_SERIAL.println(" Hz)");

  USB_SERIAL.print("Servo cmds: ");
  USB_SERIAL.print(servoCommandCount);
  USB_SERIAL.print(" (");
  USB_SERIAL.print(servoCommandCount / uptime, 1);
  USB_SERIAL.println(" Hz)");

  USB_SERIAL.print("Errors: ");
  USB_SERIAL.println(errorCount);

  USB_SERIAL.print("Serial buffer: ");
  USB_SERIAL.print(USB_SERIAL.available());
  USB_SERIAL.println(" bytes");

  USB_SERIAL.println("\nCurrent servo angles:");
  for (int i = 0; i < 6; i++) {
    USB_SERIAL.print("  S");
    USB_SERIAL.print(i);
    USB_SERIAL.print(": ");
    USB_SERIAL.println(theta[i], 2);
  }
  USB_SERIAL.println("=======================================");
  USB_SERIAL.println();
}
