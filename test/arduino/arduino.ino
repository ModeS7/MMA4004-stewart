/*
  Arduino Bidirectional USB Test - Mimic Teensy Workload
  
  This sketch mimics the REAL Stewart platform workload:
  - Send ball position data at 50Hz (like Pixy2)
  - Receive servo angle commands at 100Hz (like Maestro)
  - Process both simultaneously to stress Windows USB drivers
  
  Upload to Arduino Uno, then run Python test script
*/

const unsigned long SEND_INTERVAL = 20;  // 20ms = 50Hz ball data
unsigned long lastSend = 0;
unsigned long startTime = 0;

// Simulated ball position
float ball_x = 158.0;
float ball_y = 104.0;

// Received servo angles
float servo_angles[6] = {0, 0, 0, 0, 0, 0};

// Statistics
unsigned long ballMessagesSent = 0;
unsigned long servoCommandsReceived = 0;
unsigned long errorCount = 0;

// Serial buffer
const int MAX_CMD_LENGTH = 64;
char inputBuffer[MAX_CMD_LENGTH];
int bufferIndex = 0;

void setup() {
  Serial.begin(200000);  // Same baud as Teensy
  
  while (!Serial && millis() < 2000) {
    delay(10);
  }
  
  Serial.println("READY:Arduino Bidirectional Test");
  Serial.println("FORMAT:BALL:timestamp,x,y,detected,error_x,error_y");
  
  startTime = millis();
  randomSeed(analogRead(0));
}

void loop() {
  unsigned long now = millis();
  
  // Send ball data at 50Hz
  if (now - lastSend >= SEND_INTERVAL) {
    lastSend = now;
    sendBallData(now);
    ballMessagesSent++;
  }
  
  // Process incoming servo commands (high priority)
  // This mimics the real system where commands arrive at 100Hz
  processIncomingCommands();
}

void sendBallData(unsigned long timestamp) {
  // Simulate ball movement (random walk)
  ball_x += random(-10, 11) * 0.5;
  ball_y += random(-10, 11) * 0.5;
  
  // Keep in bounds
  ball_x = constrain(ball_x, 50, 266);
  ball_y = constrain(ball_y, 30, 178);
  
  float error_x = ball_x - 158.0;
  float error_y = 104.0 - ball_y;
  
  // Same format as Teensy
  Serial.print("BALL:");
  Serial.print((timestamp - startTime) / 1000.0, 3);
  Serial.print(",");
  Serial.print(ball_x, 2);
  Serial.print(",");
  Serial.print(ball_y, 2);
  Serial.print(",1,");  // Always detected
  Serial.print(error_x, 2);
  Serial.print(",");
  Serial.println(error_y, 2);
  
  // Explicitly flush (like Teensy)
  Serial.flush();
}

void processIncomingCommands() {
  // Process ALL available bytes (non-blocking)
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    if (c == '\n' || c == '\r') {
      if (bufferIndex > 0) {
        inputBuffer[bufferIndex] = '\0';
        handleCommand(inputBuffer);
        bufferIndex = 0;
      }
    }
    else if (bufferIndex < MAX_CMD_LENGTH - 1) {
      inputBuffer[bufferIndex++] = c;
    }
    else {
      // Buffer overflow
      bufferIndex = 0;
      errorCount++;
    }
  }
}

void handleCommand(char* cmd) {
  // Stats request
  if (strcmp(cmd, "STATS") == 0) {
    sendStats();
    return;
  }
  
  // Parse servo angles (6 comma-separated floats)
  float angles[6];
  int angleCount = 0;
  
  char* token = strtok(cmd, ",");
  while (token != NULL && angleCount < 6) {
    angles[angleCount] = atof(token);
    angleCount++;
    token = strtok(NULL, ",");
  }
  
  // Validate
  if (angleCount == 6) {
    bool valid = true;
    for (int i = 0; i < 6; i++) {
      if (abs(angles[i]) > 40 || isnan(angles[i])) {
        valid = false;
        break;
      }
    }
    
    if (valid) {
      // Store angles
      for (int i = 0; i < 6; i++) {
        servo_angles[i] = angles[i];
      }
      servoCommandsReceived++;
      
      // Send ACK (mimics Teensy behavior)
      Serial.println("ACK:OK");
    }
    else {
      Serial.println("ERROR:Invalid angles");
      errorCount++;
    }
  }
  else {
    Serial.print("ERROR:Expected 6 angles, got ");
    Serial.println(angleCount);
    errorCount++;
  }
}

void sendStats() {
  float uptime = (millis() - startTime) / 1000.0;
  
  Serial.println();
  Serial.println("======== ARDUINO STATS ========");
  Serial.print("Uptime: ");
  Serial.print(uptime, 1);
  Serial.println(" s");
  
  Serial.print("Ball messages sent: ");
  Serial.print(ballMessagesSent);
  Serial.print(" (");
  Serial.print(ballMessagesSent / uptime, 1);
  Serial.println(" Hz)");
  
  Serial.print("Servo commands received: ");
  Serial.print(servoCommandsReceived);
  Serial.print(" (");
  Serial.print(servoCommandsReceived / uptime, 1);
  Serial.println(" Hz)");
  
  Serial.print("Errors: ");
  Serial.println(errorCount);
  
  Serial.print("Current servo angles: ");
  for (int i = 0; i < 6; i++) {
    Serial.print(servo_angles[i], 2);
    if (i < 5) Serial.print(", ");
  }
  Serial.println();
  
  Serial.println("===============================");
  Serial.println();
}