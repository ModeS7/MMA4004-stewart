/*
  Standalone IMU Data Logger
  Teensy 4.1 / Arduino - No Servo Control

  Reads IMU sensors at native rates and streams data over USB Serial.
  Use with rot_core.py for real-time orientation tracking and visualization.

  Sensors:
  - LSM303DLHC Accelerometer: ~1265 Hz
  - LSM303DLHC Magnetometer: ~220 Hz
  - L3GD20H Gyroscope: ~759 Hz

  Serial Output Format:
  - A:timestamp_us,ax,ay,az
  - G:timestamp_us,gx,gy,gz
  - M:timestamp_us,mx,my,mz

  Compatible with rot_core.py from the imu directory.
*/

#include <Wire.h>

// ===== I2C ADDRESSES =====
#define LSM303_ACCEL_ADDR 0x19
#define LSM303_MAG_ADDR   0x1E
#define L3GD20_ADDR       0x6B

// ===== REGISTER ADDRESSES =====
#define LSM303_OUT_X_L_A  0x28
#define LSM303_OUT_X_H_M  0x03
#define L3GD20_OUT_X_L    0x28

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

// ===== LED INDICATOR =====
const int LED_PIN = LED_BUILTIN;
unsigned long lastBlink = 0;
bool ledState = false;

// ===== SETUP =====
void setup() {
  pinMode(LED_PIN, OUTPUT);

  // USB Serial (2 Mbaud)
  Serial.begin(2000000);

  // Wait up to 2 seconds for serial connection
  unsigned long waitStart = millis();
  while (!Serial && millis() - waitStart < 2000) {
    delay(10);
  }

  // Initialize I2C for IMU
  Wire.begin();
  Wire.setClock(400000); // 400kHz I2C

  delay(100);

  Serial.println("INIT:Standalone IMU Data Logger");
  Serial.println("INFO:Compatible with rot_core.py");

  // Configure LSM303DLHC Accelerometer
  // CTRL_REG1_A (0x20): Output Data Rate = 1344 Hz, All axes enabled
  writeRegister(LSM303_ACCEL_ADDR, 0x20, 0x97);

  // Configure LSM303DLHC Magnetometer
  // CRA_REG_M (0x00): Data output rate = 220 Hz
  writeRegister(LSM303_MAG_ADDR, 0x00, 0x1C);
  // MR_REG_M (0x02): Continuous conversion mode
  writeRegister(LSM303_MAG_ADDR, 0x02, 0x00);

  // Configure L3GD20H Gyroscope
  // CTRL_REG1 (0x20): ODR = 800 Hz, All axes enabled
  writeRegister(L3GD20_ADDR, 0x20, 0xFF);

  // Optional: Configure gyroscope full scale
  // CTRL_REG4 (0x23): ±2000 dps range (if you found 8x multiplier works)
  // 0x20 = ±2000 dps, 0x10 = ±500 dps, 0x00 = ±245 dps
  writeRegister(L3GD20_ADDR, 0x23, 0x20); // ±2000 dps

  delay(100);

  Serial.println("READY:IMU streaming at native rates");
  Serial.println("FORMAT:A:timestamp_us,ax,ay,az");
  Serial.println("FORMAT:G:timestamp_us,gx,gy,gz");
  Serial.println("FORMAT:M:timestamp_us,mx,my,mz");
  Serial.println();

  accelStartTime = micros();
  gyroStartTime = micros();
  magStartTime = micros();
  lastRateReport = micros();

  delay(100);
}

// ===== MAIN LOOP =====
void loop() {
  unsigned long now = micros();

  // Read accelerometer at ~1344 Hz, average and send every 5 samples
  if (now - lastAccelTime >= ACCEL_INTERVAL_US) {
    readAccel();

    // Accumulate for averaging
    accelSum[0] += accelData[0];
    accelSum[1] += accelData[1];
    accelSum[2] += accelData[2];
    accelAvgIndex++;

    // Send averaged sample when buffer full
    if (accelAvgIndex >= ACCEL_AVG_COUNT) {
      int16_t avgData[3];
      avgData[0] = accelSum[0] / ACCEL_AVG_COUNT;
      avgData[1] = accelSum[1] / ACCEL_AVG_COUNT;
      avgData[2] = accelSum[2] / ACCEL_AVG_COUNT;
      sendAccel(now, avgData);

      // Reset accumulator
      accelSum[0] = 0;
      accelSum[1] = 0;
      accelSum[2] = 0;
      accelAvgIndex = 0;
      accelCount++;
    }

    lastAccelTime = now;
  }

  // Read gyroscope at ~800 Hz, average and send every 3 samples
  if (now - lastGyroTime >= GYRO_INTERVAL_US) {
    readGyro();

    // Accumulate for averaging
    gyroSum[0] += gyroData[0];
    gyroSum[1] += gyroData[1];
    gyroSum[2] += gyroData[2];
    gyroAvgIndex++;

    // Send averaged sample when buffer full
    if (gyroAvgIndex >= GYRO_AVG_COUNT) {
      int16_t avgData[3];
      avgData[0] = gyroSum[0] / GYRO_AVG_COUNT;
      avgData[1] = gyroSum[1] / GYRO_AVG_COUNT;
      avgData[2] = gyroSum[2] / GYRO_AVG_COUNT;
      sendGyro(now, avgData);

      // Reset accumulator
      gyroSum[0] = 0;
      gyroSum[1] = 0;
      gyroSum[2] = 0;
      gyroAvgIndex = 0;
      gyroCount++;
    }

    lastGyroTime = now;
  }

  // Read magnetometer at ~220 Hz, average and send every 2 samples
  if (now - lastMagTime >= MAG_INTERVAL_US) {
    readMag();

    // Accumulate for averaging
    magSum[0] += magData[0];
    magSum[1] += magData[1];
    magSum[2] += magData[2];
    magAvgIndex++;

    // Send averaged sample when buffer full
    if (magAvgIndex >= MAG_AVG_COUNT) {
      int16_t avgData[3];
      avgData[0] = magSum[0] / MAG_AVG_COUNT;
      avgData[1] = magSum[1] / MAG_AVG_COUNT;
      avgData[2] = magSum[2] / MAG_AVG_COUNT;
      sendMag(now, avgData);

      // Reset accumulator
      magSum[0] = 0;
      magSum[1] = 0;
      magSum[2] = 0;
      magAvgIndex = 0;
      magCount++;
    }

    lastMagTime = now;
  }

  // Report sampling rates every 2 seconds
  if (now - lastRateReport >= RATE_REPORT_INTERVAL) {
    reportSamplingRates();
    lastRateReport = now;
  }

  // LED heartbeat (every 500ms)
  unsigned long nowMs = millis();
  if (nowMs - lastBlink > 500) {
    lastBlink = nowMs;
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState);
  }
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
  Serial.println(buf);
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
  Serial.println(buf);
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
  Serial.println(buf);
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

  Serial.print("RATE:Accel=");
  Serial.print(accelHz, 2);
  Serial.print(" Hz (avg ");
  Serial.print(ACCEL_AVG_COUNT);
  Serial.print("), Gyro=");
  Serial.print(gyroHz, 2);
  Serial.print(" Hz (avg ");
  Serial.print(GYRO_AVG_COUNT);
  Serial.print("), Mag=");
  Serial.print(magHz, 2);
  Serial.print(" Hz (avg ");
  Serial.print(MAG_AVG_COUNT);
  Serial.println(")");
}
