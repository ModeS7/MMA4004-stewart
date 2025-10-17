/*
  Stewart Platform - Servo Controller with Speed Control

  Receives 6 servo angles over serial and moves servos.
  Expected serial format: "theta0,theta1,theta2,theta3,theta4,theta5\n"
  Optional speed control: "SPD:value\n" where value is 0-255 (0=unlimited)
  Example: "0.0,0.0,0.0,0.0,0.0,0.0\n" or "SPD:50\n"
*/

#include "PololuMaestro.h"

#define maestroSerial SERIAL_PORT_HARDWARE_OPEN

MicroMaestro maestro(maestroSerial);

// CONSTANTS
float abs_0 = 4000;
float abs_90 = 8000;

// Servo ranges (CCW direction)
float range[6][2] = {
  {-45, 45}, {45, -45},
  {-45, 45}, {45, -45},
  {-45, 45}, {45, -45}
};

// Servo offsets (calibration)
float offset[6] = {-0.5, 7, -2, 4, -2.5, 5.5};

float theta[6] = {0, 0, 0, 0, 0, 0};
int servoSpeed = 20;        // 0 = unlimited (fastest)
int servoAcceleration = 20; // 0 = unlimited (fastest)

void setup() {
  Serial.begin(115200);
  maestroSerial.begin(250000);

  Serial.println("Stewart Platform Servo Controller");
  Serial.println("Commands:");
  Serial.println("  Angles: theta0,theta1,theta2,theta3,theta4,theta5");
  Serial.println("  Speed:  SPD:value (0-255, 0=unlimited)");
  Serial.println("  Accel:  ACC:value (0-255, 0=unlimited)");
  Serial.println("Ready.");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    // Check for speed command
    if (input.startsWith("SPD:")) {
      servoSpeed = input.substring(4).toInt();
      Serial.print("Speed set to: ");
      Serial.println(servoSpeed);
      return;
    }

    // Check for acceleration command
    if (input.startsWith("ACC:")) {
      servoAcceleration = input.substring(4).toInt();
      Serial.print("Acceleration set to: ");
      Serial.println(servoAcceleration);
      return;
    }

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
        moveServos(servoSpeed, servoAcceleration);

        // Minimal feedback to reduce serial overhead
        Serial.println("OK");
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