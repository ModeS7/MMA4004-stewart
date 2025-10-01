/*
  Stewart Platform - Simple Servo Controller

  Receives 6 servo angles over serial and moves servos.
  Expected serial format: "theta0,theta1,theta2,theta3,theta4,theta5\n"
  Example: "0.0,0.0,0.0,0.0,0.0,0.0\n"

  Before using:
  - Serial mode: UART, fixed baud rate
  - Baud rate: 9600 (maestroSerial)
  - Serial monitor: 115200 (USB Serial)
*/

#include "PololuMaestro.h"

#define maestroSerial SERIAL_PORT_HARDWARE_OPEN

MicroMaestro maestro(maestroSerial);

// CONSTANTS
float abs_0 = 4000;   // 0 degrees position in microseconds
float abs_90 = 8000;  // 90 degrees position in microseconds

// Servo ranges (CCW direction)
float range[6][2] = {
  {-45, 45}, {45, -45},   // a1, a2
  {-45, 45}, {45, -45},   // b1, b2
  {-45, 45}, {45, -45}    // c1, c2
};

// Servo offsets (calibration)
float offset[6] = {-0.5, 7, -2, 4, -2.5, 5.5};

float theta[6] = {0, 0, 0, 0, 0, 0};  // Current servo angles

void setup() {
  Serial.begin(115200);
  maestroSerial.begin(9600);

  Serial.println("Stewart Platform Servo Controller");
  Serial.println("Send format: theta0,theta1,theta2,theta3,theta4,theta5");
  Serial.println("Ready.");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    // Parse comma-separated angles
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

    // Check if we received all 6 angles
    if (angleIndex == 6) {
      // Validate angles
      bool valid = true;
      for (int i = 0; i < 6; i++) {
        if (abs(angles[i]) > 40) {
          Serial.print("ERROR: Angle ");
          Serial.print(i);
          Serial.print(" exceeds limit: ");
          Serial.println(angles[i]);
          valid = false;
          break;
        }
        if (isnan(angles[i])) {
          Serial.print("ERROR: Angle ");
          Serial.print(i);
          Serial.println(" is NaN");
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
        moveServos(0, 0);  // Max speed and acceleration

        Serial.print("OK: ");
        for (int i = 0; i < 6; i++) {
          Serial.print(theta[i]);
          if (i < 5) Serial.print(",");
        }
        Serial.println();
      }
    } else {
      Serial.println("ERROR: Expected 6 angles");
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