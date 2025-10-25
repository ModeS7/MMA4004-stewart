#include <Wire.h>

// I2C Addresses
#define LSM303_ACCEL_ADDR 0x19
#define LSM303_MAG_ADDR   0x1E
#define L3GD20_ADDR       0x6B

// Register addresses
#define LSM303_STATUS_REG_A 0x27
#define LSM303_OUT_X_L_A    0x28
#define LSM303_SR_REG_M     0x09
#define LSM303_OUT_X_H_M    0x03
#define L3GD20_STATUS_REG   0x27
#define L3GD20_OUT_X_L      0x28

// Raw data buffers
int16_t accelData[3];
int16_t gyroData[3];
int16_t magData[3];

// Performance tracking
unsigned long accelCount = 0;
unsigned long gyroCount = 0;
unsigned long magCount = 0;
unsigned long accelStartTime = 0;
unsigned long gyroStartTime = 0;
unsigned long magStartTime = 0;

// Rate measurement
unsigned long lastRateReport = 0;
const unsigned long rateReportInterval = 2000000; // Report every 2 seconds

void setup() {
  Serial.begin(2000000);

  Wire.begin();
  Wire.setClock(400000); // 400kHz I2C

  delay(100);

  // Configure accelerometer: 1344 Hz
  writeRegister(LSM303_ACCEL_ADDR, 0x20, 0x97);

  // Configure gyroscope: 800 Hz
  writeRegister(L3GD20_ADDR, 0x20, 0xFF);

  // Configure magnetometer: 220 Hz
  writeRegister(LSM303_MAG_ADDR, 0x00, 0x1C);
  writeRegister(LSM303_MAG_ADDR, 0x01, 0x20);
  writeRegister(LSM303_MAG_ADDR, 0x02, 0x00);

  delay(100);
  Serial.println("Sampling started...");
  delay(1000);

  // Initialize timing
  accelStartTime = micros();
  gyroStartTime = micros();
  magStartTime = micros();
  lastRateReport = micros();
}

bool isAccelReady() {
  Wire.beginTransmission(LSM303_ACCEL_ADDR);
  Wire.write(LSM303_STATUS_REG_A);
  Wire.endTransmission();

  Wire.requestFrom(LSM303_ACCEL_ADDR, 1);
  if (Wire.available()) {
    uint8_t status = Wire.read();
    return (status & 0x08) != 0; // Check bit 3 (ZYXDA - new data available)
  }
  return false;
}

bool isGyroReady() {
  Wire.beginTransmission(L3GD20_ADDR);
  Wire.write(L3GD20_STATUS_REG);
  Wire.endTransmission();

  Wire.requestFrom(L3GD20_ADDR, 1);
  if (Wire.available()) {
    uint8_t status = Wire.read();
    return (status & 0x08) != 0; // Check bit 3 (ZYXDA - new data available)
  }
  return false;
}

bool isMagReady() {
  Wire.beginTransmission(LSM303_MAG_ADDR);
  Wire.write(LSM303_SR_REG_M);
  Wire.endTransmission();

  Wire.requestFrom(LSM303_MAG_ADDR, 1);
  if (Wire.available()) {
    uint8_t status = Wire.read();
    return (status & 0x01) != 0; // Check bit 0 (DRDY - data ready)
  }
  return false;
}

void loop() {
  unsigned long now = micros();

  // Check accelerometer DRDY flag (bit 3 of STATUS_REG_A)
  if (isAccelReady()) {
    readAccel();
    printAccel();
    accelCount++;
  }

  // Check gyroscope DRDY flag (bit 3 of STATUS_REG)
  if (isGyroReady()) {
    readGyro();
    printGyro();
    gyroCount++;
  }

  // Check magnetometer DRDY flag (bit 0 of SR_REG_M)
  if (isMagReady()) {
    readMag();
    printMag();
    magCount++;
  }

  // Report sampling rates every 2 seconds
  if (now - lastRateReport >= rateReportInterval) {
    unsigned long accelElapsed = now - accelStartTime;
    unsigned long gyroElapsed = now - gyroStartTime;
    unsigned long magElapsed = now - magStartTime;

    float accelHz = (accelCount * 1000000.0) / accelElapsed;
    float gyroHz = (gyroCount * 1000000.0) / gyroElapsed;
    float magHz = (magCount * 1000000.0) / magElapsed;

    Serial.print("RATE - Accel: ");
    Serial.print(accelHz, 2);
    Serial.print(" Hz | Gyro: ");
    Serial.print(gyroHz, 2);
    Serial.print(" Hz | Mag: ");
    Serial.print(magHz, 2);
    Serial.println(" Hz");

    lastRateReport = now;
  }
}

void writeRegister(uint8_t addr, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

void readAccel() {
  Wire.beginTransmission(LSM303_ACCEL_ADDR);
  Wire.write(LSM303_OUT_X_L_A | 0x80); // Auto-increment bit
  Wire.endTransmission();

  Wire.requestFrom(LSM303_ACCEL_ADDR, 6);

  uint8_t xlo = Wire.read();
  uint8_t xhi = Wire.read();
  uint8_t ylo = Wire.read();
  uint8_t yhi = Wire.read();
  uint8_t zlo = Wire.read();
  uint8_t zhi = Wire.read();

  // Combine bytes (12-bit left-justified in 16-bit)
  accelData[0] = (int16_t)(xhi << 8 | xlo) >> 4;
  accelData[1] = (int16_t)(yhi << 8 | ylo) >> 4;
  accelData[2] = (int16_t)(zhi << 8 | zlo) >> 4;
}

void readGyro() {
  Wire.beginTransmission(L3GD20_ADDR);
  Wire.write(L3GD20_OUT_X_L | 0x80); // Auto-increment bit
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

void readMag() {
  Wire.beginTransmission(LSM303_MAG_ADDR);
  Wire.write(LSM303_OUT_X_H_M);
  Wire.endTransmission();

  Wire.requestFrom(LSM303_MAG_ADDR, 6);

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

void printAccel() {
  Serial.print("A:");
  Serial.print(accelData[0]);
  Serial.print(",");
  Serial.print(accelData[1]);
  Serial.print(",");
  Serial.println(accelData[2]);
}

void printGyro() {
  Serial.print("G:");
  Serial.print(gyroData[0]);
  Serial.print(",");
  Serial.print(gyroData[1]);
  Serial.print(",");
  Serial.println(gyroData[2]);
}

void printMag() {
  Serial.print("M:");
  Serial.print(magData[0]);
  Serial.print(",");
  Serial.print(magData[1]);
  Serial.print(",");
  Serial.println(magData[2]);
}
