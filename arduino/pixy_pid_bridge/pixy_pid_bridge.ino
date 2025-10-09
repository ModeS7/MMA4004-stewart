/*
  Stewart Platform Ball Balance Controller

  Hardware:
  - Pixy2 camera on SPI (with SS pin)
  - Pololu Maestro on Serial1 (pins 0/1)
  - USB Serial for communication with Python

  Communication protocol:
  - SENDS to PC: "timestamp,x,y,detected,error_x,error_y\n"
  - RECEIVES from PC: "theta0,theta1,theta2,theta3,theta4,theta5\n"
*/

#include <Pixy2SPI_SS.h>
#include "PololuMaestro.h"

// Pixy2 on SPI
Pixy2SPI_SS pixy;

// Maestro on Serial1 (hardware serial)
#define maestroSerial Serial1
MicroMaestro maestro(maestroSerial);

// Pixy2 constants
const float ORIGIN_X = 158.0;  // Center X (316/2)
const float ORIGIN_Y = 104.0;  // Center Y (208/2)

// Servo control constants
const float abs_0 = 4000;
const float abs_90 = 8000;

// Servo ranges (CCW direction)
float range[6][2] = {
  {-45, 45}, {45, -45},
  {-45, 45}, {45, -45},
  {-45, 45}, {45, -45}
};

// Servo offsets (calibration)
float offset[6] = {-0.5, 7, -2, 4, -2.5, 5.5};

// Current servo angles
float theta[6] = {0, 0, 0, 0, 0, 0};
int servoSpeed = 20;
int servoAcceleration = 20;

// Timing
unsigned long lastPixyRead = 0;
unsigned long lastSerialCheck = 0;
const unsigned long PIXY_INTERVAL = 50;  // 20Hz
const unsigned long SERIAL_CHECK_INTERVAL = 5;  // Check serial often

void setup() {
  // USB Serial for Python communication
  Serial.begin(115200);

  // Maestro servo controller
  maestroSerial.begin(9600);

  // Initialize Pixy2
  Serial.println("INIT:Initializing Pixy2...");
  int result = pixy.init();

  if (result < 0) {
    Serial.println("ERROR:Pixy2 init failed");
    while(1) {
      delay(1000);
      Serial.println("ERROR:Pixy2 init failed");
    }
  }

  Serial.println("READY:Ball balance controller online");
  Serial.println("FORMAT:timestamp,x,y,detected,error_x,error_y");

  delay(500);
}

void loop() {
  unsigned long now = millis();

  // Read ball position at 20Hz
  if (now - lastPixyRead >= PIXY_INTERVAL) {
    lastPixyRead = now;
    readAndSendBallPosition(now);
  }

  // Check for incoming servo commands frequently
  if (now - lastSerialCheck >= SERIAL_CHECK_INTERVAL) {
    lastSerialCheck = now;
    checkSerialCommands();
  }
}

void readAndSendBallPosition(unsigned long timestamp) {
  // Get blocks from Pixy2 (signature 1, max 1 block)
  int8_t num_blocks = pixy.ccc.getBlocks(false, CCC_SIG1, 1);

  float ball_x, ball_y;
  bool detected;

  if (num_blocks > 0 && pixy.ccc.numBlocks == 1) {
    // Ball detected
    ball_x = pixy.ccc.blocks[0].m_x;
    ball_y = pixy.ccc.blocks[0].m_y;
    detected = true;
  } else {
    // No ball detected - send zeros
    ball_x = 0.0;
    ball_y = 0.0;
    detected = false;
  }

  // Calculate error from center
  float error_x = ball_x - ORIGIN_X;
  float error_y = ORIGIN_Y - ball_y;  // Inverted Y

  // Send to Python: timestamp,x,y,detected,error_x,error_y
  Serial.print("BALL:");
  Serial.print(timestamp / 1000.0, 3);
  Serial.print(",");
  Serial.print(ball_x, 2);
  Serial.print(",");
  Serial.print(ball_y, 2);
  Serial.print(",");
  Serial.print(detected ? "1" : "0");
  Serial.print(",");
  Serial.print(error_x, 2);
  Serial.print(",");
  Serial.println(error_y, 2);
}

void checkSerialCommands() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    // Check for speed command
    if (input.startsWith("SPD:")) {
      servoSpeed = input.substring(4).toInt();
      Serial.print("ACK:Speed=");
      Serial.println(servoSpeed);
      return;
    }

    // Check for acceleration command
    if (input.startsWith("ACC:")) {
      servoAcceleration = input.substring(4).toInt();
      Serial.print("ACK:Accel=");
      Serial.println(servoAcceleration);
      return;
    }

    // Parse comma-separated servo angles
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

    // Validate: must have 6 angles
    if (angleIndex == 6) {
      bool valid = true;

      // Check angle limits
      for (int i = 0; i < 6; i++) {
        if (abs(angles[i]) > 40 || isnan(angles[i])) {
          Serial.print("ERROR:Invalid angle[");
          Serial.print(i);
          Serial.print("]=");
          Serial.println(angles[i]);
          valid = false;
          break;
        }
      }

      if (valid) {
        // Update theta array
        for (int i = 0; i < 6; i++) {
          theta[i] = angles[i];
        }

        // Move servos
        moveServos(servoSpeed, servoAcceleration);

        // Acknowledge
        Serial.println("ACK:Servos updated");
      }
    } else {
      Serial.print("ERROR:Expected 6 angles, got ");
      Serial.println(angleIndex);
    }
  }
}

void moveServos(int spd, int acc) {
  for (int i = 0; i < 6; i++) {
    float pos = theta[i] + offset[i];
    pos = map(pos, range[i][0], range[i][1], abs_0, abs_90);
    maestro.setSpeed(i, spd);
    maestro.setAcceleration(i, acc);
    maestro.setTarget(i, pos);
  }
}