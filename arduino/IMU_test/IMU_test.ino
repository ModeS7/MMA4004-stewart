#include <Wire.h>

// I2C Addresses
#define LSM303_ACCEL_ADDR 0x19
#define L3GD20_ADDR       0x6B

// Register addresses
#define LSM303_OUT_X_L_A  0x28
#define L3GD20_OUT_X_L    0x28

// Raw data buffers
int16_t accelData[3];
int16_t gyroData[3];

// Performance tracking
unsigned long accelCount = 0;
unsigned long gyroCount = 0;
unsigned long lastAccelTime = 0;
unsigned long lastGyroTime = 0;
unsigned long accelStartTime = 0;
unsigned long gyroStartTime = 0;

// Rate measurement
unsigned long lastRateReport = 0;
const unsigned long rateReportInterval = 2000000; // Report every 2 seconds (in microseconds)

void setup() {
  Serial.begin(2000000);
  
  Wire.begin();
  Wire.setClock(400000); // 400kHz I2C
  
  delay(100);
  
  // Configure accelerometer: 1344 Hz
  writeRegister(LSM303_ACCEL_ADDR, 0x20, 0x97);
  
  // Configure gyroscope: 800 Hz
  writeRegister(L3GD20_ADDR, 0x20, 0xFF);
  
  delay(100);
  Serial.println("Sampling started...");
  delay(1000);
  
  // Initialize timing
  accelStartTime = micros();
  gyroStartTime = micros();
  lastRateReport = micros();
}

void loop() {
  unsigned long now = micros();
  
  // Read accelerometer every 744us (1344Hz target)
  if (now - lastAccelTime >= 744) {
    readAccel();
    printAccel();
    lastAccelTime = now;
    accelCount++;
  }
  
  // Read gyroscope every 1250us (800Hz target)
  if (now - lastGyroTime >= 1250) {
    readGyro();
    printGyro();
    lastGyroTime = now;
    gyroCount++;
  }
  
  // Report sampling rates every 2 seconds
  if (now - lastRateReport >= rateReportInterval) {
    unsigned long accelElapsed = now - accelStartTime;
    unsigned long gyroElapsed = now - gyroStartTime;
    
    float accelHz = (accelCount * 1000000.0) / accelElapsed;
    float gyroHz = (gyroCount * 1000000.0) / gyroElapsed;
    
    Serial.print("RATE - Accel: ");
    Serial.print(accelHz, 2);
    Serial.print(" Hz | Gyro: ");
    Serial.print(gyroHz, 2);
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