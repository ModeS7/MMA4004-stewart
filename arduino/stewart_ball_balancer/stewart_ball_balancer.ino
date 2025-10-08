# Include necessary libraries
#include <Pixy2SPI_SS.h>
#include <PololuMaestro.h>

// Constants
const int PIXY_CENTER_X = 158;
const int PIXY_CENTER_Y = 104;
const int SERIAL_BAUD = 9600;

// Global variables
Pixy2 pixy;
Maestro maestro;

void setup() {
    Serial.begin(SERIAL_BAUD);
    pixy.init();
    maestro.init();
}

void loop() {
    pixy.ccc.getBlocks(); // Get blocks from Pixy2
    if (pixy.ccc.numBlocks > 0) {
        int ball_x = pixy.ccc.blocks[0].x;
        int ball_y = pixy.ccc.blocks[0].y;
        int error_x = PIXY_CENTER_X - ball_x;
        int error_y = PIXY_CENTER_Y - ball_y;
        unsigned long timestamp = millis();
        // Output CSV
        Serial.print(timestamp);
        Serial.print(",");
        Serial.print(ball_x);
        Serial.print(",");
        Serial.print(ball_y);
        Serial.print(",");
        Serial.print(pixy.ccc.numBlocks);
        Serial.print(",");
        Serial.print(error_x);
        Serial.print(",");
        Serial.println(error_y);
    }
    // Listen for incoming servo commands
    if (Serial.available()) {
        String command = Serial.readStringUntil('\n');
        // Send commands to Maestro
        maestro.setTarget(0, command.toInt()); // Add logic for all servos
    }
    delay(50); // Loop rate
}