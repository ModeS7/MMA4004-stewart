/*
 * Servo Calibration for Teensy + Pololu Maestro
 *
 * Direct passthrough - no calibration offsets applied
 * Uses same communication pattern as control_v1.ino
 *
 * USB (Teensy ↔ PC): 200000 baud
 * Serial1 (Teensy ↔ Maestro): 250000 baud
 */

#include "PololuMaestro.h"

// Hardware interfaces
#define USB_SERIAL Serial
#define MAESTRO_SERIAL Serial1

MicroMaestro maestro(MAESTRO_SERIAL);

// Servo configuration
const int NUM_SERVOS = 6;
const float abs_0 = 4000;   // Corresponds to -90 degrees pulse width
const float abs_90 = 8000;  // Corresponds to +90 degrees pulse width

// Servo direction mapping (from control_v1.ino)
// Servos 0,2,4 are normal, Servos 1,3,5 are reversed
float range[6][2] = {
  {-45, 45}, {45, -45},
  {-45, 45}, {45, -45},
  {-45, 45}, {45, -45}
};

// Current servo angles (degrees)
float servoAngles[6] = {0, 0, 0, 0, 0, 0};

// Serial input buffer
const int MAX_CMD_LENGTH = 64;
char inputBuffer[MAX_CMD_LENGTH];
int bufferIndex = 0;

// LED indicator
const int LED_PIN = LED_BUILTIN;
unsigned long lastBlink = 0;
bool ledState = false;

void setup() {
  pinMode(LED_PIN, OUTPUT);

  // USB Serial (Teensy ↔ PC): 200000 baud (match control_v1)
  USB_SERIAL.begin(200000);
  unsigned long waitStart = millis();
  while (!USB_SERIAL && millis() - waitStart < 2000) {
    delay(10);
  }

  // Maestro Serial: 250000 baud (match control_v1)
  MAESTRO_SERIAL.begin(250000);

  // Set unlimited speed (no ramping)
  for (int i = 0; i < NUM_SERVOS; i++) {
    maestro.setSpeed(i, 0);
    maestro.setAcceleration(i, 0);
  }

  USB_SERIAL.println("READY:Servo Calibration Mode");
  USB_SERIAL.println("FORMAT:angle0,angle1,angle2,angle3,angle4,angle5");

  // Initialize to neutral
  setAllServosNeutral();

  delay(100);
}

void loop() {
  // Check for serial commands
  checkSerialCommands();

  // LED heartbeat
  if (millis() - lastBlink > 500) {
    lastBlink = millis();
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState);
  }
}

void checkSerialCommands() {
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
      USB_SERIAL.println("ERROR:Command too long");
      break;
    }
  }
}

void processCommand(char* cmd) {
  // Parse comma-separated angles
  float angles[NUM_SERVOS];
  int angleCount = 0;

  char* token = strtok(cmd, ",");

  while (token != NULL && angleCount < NUM_SERVOS) {
    angles[angleCount] = atof(token);
    angleCount++;
    token = strtok(NULL, ",");
  }

  // Validate count
  if (angleCount != NUM_SERVOS) {
    USB_SERIAL.print("ERROR:Expected 6 angles, got ");
    USB_SERIAL.println(angleCount);
    return;
  }

  // No angle limits - allow full calibration range
  // Update angles and move servos
  for (int i = 0; i < NUM_SERVOS; i++) {
    servoAngles[i] = angles[i];
  }

  moveServos();

  // Send acknowledgment (match control_v1 format)
  USB_SERIAL.println("ACK:OK");
}

void moveServos() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    // Apply directional mapping (match control_v1.ino)
    // No offsets - we're calibrating to find those
    float pos = map_float(servoAngles[i], range[i][0], range[i][1], abs_0, abs_90);

    // Clamp to safe range
    pos = constrain(pos, 3000, 9000);

    // Send to Maestro
    maestro.setTarget(i, (uint16_t)pos);
  }
}

void setAllServosNeutral() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    servoAngles[i] = 0.0;
  }
  moveServos();
  delay(100);
}

float map_float(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}
