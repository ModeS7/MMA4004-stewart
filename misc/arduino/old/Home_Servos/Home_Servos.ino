/* Before using this, go to the Serial Settings
  tab in the Maestro Control Center and apply these settings:

   Serial mode: UART, fixed baud rate
   Baud rate: 9600
   CRC disabled

  Be sure to click "Apply Settings" after making any changes.

  Array Indexes
  -------------
  a1 = 0
  a2 = 1
  b1 = 2
  b2 = 3
  c1 = 4
  c2 = 5

*/
  
#include "PololuMaestro.h"
#define maestroSerial SERIAL_PORT_HARDWARE_OPEN //for teensy
MicroMaestro maestro(maestroSerial);  // if using the Maestro Micro controller 


//**CONSTANTS**
//-------------
float abs_0 = 4000;   //ms position of absolute 0 degrees
float abs_90 = 8000;  //ms position of absolute 90 degrees

//**USER DEFINED VALUES**
//----------------------

// angle range for each servo going CCW
// each servo can move from absolute 0 to absolute 90 degrees
// this defines a different range such as 90 to 180 degrees 
float range[6][2] = { { -45, 45 }, { 45, -45 }, // a1, a2
                      { -45, 45 }, { 45, -45 }, // b1, b2
                      { -45, 45 }, { 45, -45 }}; // c1, c2

// angle offset value for each servo (should be less than 14 degrees since there are 25 teeth on the servo)
// redefines the position of the lower range of each servo
// Example: a servo with a range [0, 90] and an offset value of 5 would have its new 0 degree position where the old 5 degree position was

float offset[6] = {-0.5, 7, // channel #0 and channel #1
                   -2, 4, // channel #2 and channel #3
                   -2.5, 5.5}; // channel #4 and channel #5

float X, Y; // X and Y co-ords of the ball

void setup() {
  Serial.begin(115200);
  maestroSerial.begin(9600);

  // moves servos to 0 degree position
  int pos;
  for(int i = 0; i < 6; i++){
    maestro.setSpeed(0, 0); //sets speed to servo i
    maestro.setAcceleration(0, 0); //setsacceleration to servo i
    pos = map(offset[i], range[i][0], range[i][1], abs_0, abs_90);  //finds servo 0 degree position in microseconds
    maestro.setTarget(i, pos); //drives motor to calculated position
  }
}

void loop() {  
}
