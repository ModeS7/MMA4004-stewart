/*
  Stewart Platform Ball Balance Controller
  Teensy 4.1 - All speed restrictions removed
*/

#include <Pixy2SPI_SS.h>
#include "PololuMaestro.h"

// ===== HARDWARE INTERFACES =====
#define USB_SERIAL Serial
#define MAESTRO_SERIAL Serial1

Pixy2SPI_SS pixy;
MicroMaestro maestro(MAESTRO_SERIAL);

// ===== PIXY2 CONSTANTS =====
const float ORIGIN_X = 158.0;
const float ORIGIN_Y = 104.0;

// ===== SERVO CONTROL CONSTANTS =====
const float abs_0 = 4000;
const float abs_90 = 8000;

float range[6][2] = {
  {-45, 45}, {45, -45},
  {-45, 45}, {45, -45},
  {-45, 45}, {45, -45}
};

float offset[6] = {-0.5, 7, -2, 4, -2.5, 5.5};
float theta[6] = {0, 0, 0, 0, 0, 0};

// **SPEED SETTINGS**
int servoSpeed = 0;           // 0 = UNLIMITED
int servoAcceleration = 0;    // 0 = NO RAMPING

// ===== TIMING CONTROL =====
const unsigned long PIXY_INTERVAL = 20;  // 50Hz ball position updates
const unsigned long SERIAL_CHECK_INTERVAL = 1;  // Check serial every 1ms

unsigned long lastPixyRead = 0;
unsigned long lastSerialCheck = 0;
unsigned long startTime = 0;

// ===== SERIAL INPUT BUFFER (OPTIMIZED) =====
const int MAX_CMD_LENGTH = 64;  // Reduced from 128
char inputBuffer[MAX_CMD_LENGTH];
int bufferIndex = 0;

// Pre-parsed command storage
float pendingAngles[6];
bool hasNewCommand = false;

// ===== PERFORMANCE MONITORING =====
elapsedMicros loopTimer;
uint32_t maxLoopTime = 0;
uint32_t pixyReadCount = 0;
uint32_t servoCommandCount = 0;
uint32_t errorCount = 0;

// ===== LED INDICATOR =====
const int LED_PIN = LED_BUILTIN;
unsigned long lastBlink = 0;
bool ledState = false;

// ===== SETUP =====
void setup() {
  pinMode(LED_PIN, OUTPUT);
  
  // USB Serial
  USB_SERIAL.begin(115200);
  unsigned long waitStart = millis();
  while (!USB_SERIAL && millis() - waitStart < 2000) {
    delay(10);
  }

  // Maestro servo controller
  MAESTRO_SERIAL.begin(115200);

  // Initialize Pixy2
  USB_SERIAL.println("INIT:Starting MAX SPEED mode...");
  int result = pixy.init();

  if (result < 0) {
    USB_SERIAL.println("ERROR:Pixy2 init failed");
    while(1) {
      digitalWrite(LED_PIN, (millis() / 200) % 2);
      delay(100);
    }
  }

  // **SET ALL SERVOS SPEED**
  for (int i = 0; i < 6; i++) {
    maestro.setSpeed(i, 0);
    maestro.setAcceleration(i, 0);
  }

  USB_SERIAL.println("READY:Teensy online");
  USB_SERIAL.println("FORMAT:BALL:timestamp,x,y,detected,error_x,error_y");
  
  startTime = millis();
  
  // Clear buffers
  while (USB_SERIAL.available()) {
    USB_SERIAL.read();
  }
  
  delay(100);
}

// ===== MAIN LOOP =====
void loop() {
  loopTimer = 0;
  
  unsigned long now = millis();

  // Read ball position at 50Hz
  if (now - lastPixyRead >= PIXY_INTERVAL) {
    lastPixyRead = now;
    readAndSendBallPosition(now);
    pixyReadCount++;
  }

  // Check for commands
  if (now - lastSerialCheck >= SERIAL_CHECK_INTERVAL) {
    lastSerialCheck = now;
    checkSerialCommandsNonBlocking();
  }

  // LED heartbeat
  if (now - lastBlink > 500) {
    lastBlink = now;
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

  // Send data
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

  // Speed command (allows user override if needed)
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
    if (abs(angles[i]) > 40 || isnan(angles[i])) {
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
    float pos = theta[i] + offset[i];
    pos = map_float(pos, range[i][0], range[i][1], abs_0, abs_90);
    pos = constrain(pos, 3000, 9000);
    
    // Always use current speed settings
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
  USB_SERIAL.print("Mode: MAX SPEED");
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