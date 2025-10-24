/*
  Stewart Platform - Optimized IMU Streaming
  
  Receives servo commands: "theta0,theta1,theta2,theta3,theta4,theta5\n"
  Sends IMU: "IMU,imu_us,cmd_us,ax,ay,az,gx,gy,gz\n"
  Sends CMD: "CMD,cmd_us,theta0,theta1,theta2,theta3,theta4,theta5\n" (only on change)
*/

#include "PololuMaestro.h"
#include <Wire.h>

#define maestroSerial SERIAL_PORT_HARDWARE_OPEN

MicroMaestro maestro(maestroSerial);

// I2C Addresses
#define LSM303_ACCEL_ADDR 0x19
#define L3GD20_ADDR       0x6B
#define LSM303_OUT_X_L_A  0x28
#define L3GD20_OUT_X_L    0x28

// Servo configuration
float abs_0 = 4000;
float abs_90 = 8000;

float range[6][2] = {
  {-45, 45}, {45, -45},
  {-45, 45}, {45, -45},
  {-45, 45}, {45, -45}
};

float offset[6] = {-0.5, 7, -2, 4, -2.5, 5.5};
float theta[6] = {0, 0, 0, 0, 0, 0};

// Track when command was received
unsigned long commandTimestamp = 0;

// IMU data
int16_t accelData[3] = {0, 0, 0};
int16_t gyroData[3] = {0, 0, 0};
unsigned long lastAccelTime = 0;
unsigned long lastGyroTime = 0;

bool imuInitialized = false;

// Buffer for faster serial output
char msgBuffer[128];

void setup() {
  Serial.begin(2000000);
  
  // Initialize Maestro first
  maestroSerial.begin(9600);
  delay(100);
  
  // Initialize I2C for IMU
  Wire.begin();
  Wire.setClock(400000);
  delay(100);
  
  // Configure accelerometer: 1344 Hz, normal mode, all axes enabled
  writeRegister(LSM303_ACCEL_ADDR, 0x20, 0x97);
  delay(10);
  
  // Configure gyroscope: 800 Hz, normal mode, all axes enabled
  writeRegister(L3GD20_ADDR, 0x20, 0xFF);
  delay(10);
  
  // Verify IMU is responding
  Wire.beginTransmission(LSM303_ACCEL_ADDR);
  if (Wire.endTransmission() == 0) {
    Wire.beginTransmission(L3GD20_ADDR);
    if (Wire.endTransmission() == 0) {
      imuInitialized = true;
      Serial.println("IMU_OK");
    } else {
      Serial.println("IMU_ERROR:Gyro not found");
    }
  } else {
    Serial.println("IMU_ERROR:Accel not found");
  }
  
  Serial.println("READY");
  
  lastAccelTime = micros();
  lastGyroTime = micros();
  commandTimestamp = micros();
}

void loop() {
  unsigned long now = micros();
  
  // Read and send IMU data at high frequency
  if (imuInitialized) {
    // Read accelerometer every 744us (1344Hz target)
    if (now - lastAccelTime >= 744) {
      readAccel();
      sendIMU(now);
      lastAccelTime = now;
    }
    
    // Read gyroscope every 1250us (800Hz target)
    if (now - lastGyroTime >= 1250) {
      readGyro();
      lastGyroTime = now;
    }
  }
  
  // Check for servo commands (non-blocking)
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input.length() == 0) {
      return;
    }
    
    if (input.startsWith("SPD:")) {
      int spd = input.substring(4).toInt();
      for (int i = 0; i < 6; i++) {
        maestro.setSpeed(i, spd);
        maestro.setAcceleration(i, 0);
      }
      Serial.println("OK_SPD");
      return;
    }
    
    // Parse servo angles
    float angles[6];
    int angleIndex = 0;
    int startIndex = 0;
    
    for (int i = 0; i <= input.length(); i++) {
      if (i == input.length() || input.charAt(i) == ',') {
        if (angleIndex < 6) {
          angles[angleIndex] = input.substring(startIndex, i).toFloat();
          angleIndex++;
          startIndex = i + 1;
        }
      }
    }
    
    if (angleIndex == 6) {
      bool valid = true;
      for (int i = 0; i < 6; i++) {
        if (abs(angles[i]) > 40 || isnan(angles[i])) {
          valid = false;
          break;
        }
      }
      
      if (valid) {
        commandTimestamp = micros();
        
        for (int i = 0; i < 6; i++) {
          theta[i] = angles[i];
        }
        
        moveServos();
        
        // Send command update message
        sendCommand(commandTimestamp);
        
        Serial.println("OK");
      } else {
        Serial.println("ERROR");
      }
    }
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
  Wire.write(LSM303_OUT_X_L_A | 0x80);
  Wire.endTransmission();
  
  Wire.requestFrom(LSM303_ACCEL_ADDR, 6);
  
  if (Wire.available() >= 6) {
    uint8_t xlo = Wire.read();
    uint8_t xhi = Wire.read();
    uint8_t ylo = Wire.read();
    uint8_t yhi = Wire.read();
    uint8_t zlo = Wire.read();
    uint8_t zhi = Wire.read();
    
    accelData[0] = (int16_t)(xhi << 8 | xlo) >> 4;
    accelData[1] = (int16_t)(yhi << 8 | ylo) >> 4;
    accelData[2] = (int16_t)(zhi << 8 | zlo) >> 4;
  }
}

void readGyro() {
  Wire.beginTransmission(L3GD20_ADDR);
  Wire.write(L3GD20_OUT_X_L | 0x80);
  Wire.endTransmission();
  
  Wire.requestFrom(L3GD20_ADDR, 6);
  
  if (Wire.available() >= 6) {
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
}

void sendIMU(unsigned long timestamp) {
  // Optimized: use sprintf for speed (single serial write)
  // Format: IMU,imu_us,cmd_us,ax,ay,az,gx,gy,gz
  sprintf(msgBuffer, "IMU,%lu,%lu,%d,%d,%d,%d,%d,%d",
          timestamp, commandTimestamp,
          accelData[0], accelData[1], accelData[2],
          gyroData[0], gyroData[1], gyroData[2]);
  Serial.println(msgBuffer);
}

void sendCommand(unsigned long timestamp) {
  // Format: CMD,cmd_us,theta0,theta1,theta2,theta3,theta4,theta5
  // Use dtostrf for float formatting
  char t0[8], t1[8], t2[8], t3[8], t4[8], t5[8];
  dtostrf(theta[0], 6, 3, t0);
  dtostrf(theta[1], 6, 3, t1);
  dtostrf(theta[2], 6, 3, t2);
  dtostrf(theta[3], 6, 3, t3);
  dtostrf(theta[4], 6, 3, t4);
  dtostrf(theta[5], 6, 3, t5);
  
  sprintf(msgBuffer, "CMD,%lu,%s,%s,%s,%s,%s,%s",
          timestamp, t0, t1, t2, t3, t4, t5);
  Serial.println(msgBuffer);
}

void moveServos() {
  for (int i = 0; i < 6; i++) {
    float pos = theta[i] + offset[i];
    pos = map(pos, range[i][0], range[i][1], abs_0, abs_90);
    maestro.setSpeed(i, 0);
    maestro.setAcceleration(i, 0);
    maestro.setTarget(i, pos);
  }
}