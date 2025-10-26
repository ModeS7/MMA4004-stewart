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

// libraries
#include "PololuMaestro.h"
#include <Pixy2SPI_SS.h>

#define maestroSerial SERIAL_PORT_HARDWARE_OPEN  //for teensy

// objects
Pixy2SPI_SS pixy;
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
float range[6][2] = { { -45, 45 }, 
                      { 45, -45 },  // a1, a2
                      { -45, 45 },
                      { 45, -45 },  // b1, b2
                      { -45, 45 },
                      { 45, -45 } };  // c1, c2

// angle offset value for each servo (should be less than 14 degrees since there are 25 teeth on the servo)
// redefines the position of the lower range of each servo
// Example: a servo with a range [0, 90] and an offset value of 5 would have its new 0 degree position where the old 5 degree position was

float offset[6] = {-0.5, 7, // channel #0 and channel #1
                   -2, 4, // channel #2 and channel #3
                   -2.5, 5.5}; // channel #4 and channel #5

float X, Y;        // X and Y co-ords of the markers
float r_platform;  //max radius of the platform

float X_total = 0, Y_total = 0;  // the total summation of X and Y values
float r_platform_total = 0;           // the total summation of r_max values
int j = 0;

void setup() {
  Serial.begin(115200);
  maestroSerial.begin(9600);
  pixy.init();

  // moves servos to 0 degree position
  int pos;
  for (int i = 0; i < 6; i++) {
    maestro.setSpeed(0, 0);                                         //sets speed to servo i
    maestro.setAcceleration(0, 0);                                  //setsacceleration to servo i
    pos = map(offset[i], range[i][0], range[i][1], abs_0, abs_90);  //finds servo 0 degree position in microseconds
    maestro.setTarget(i, pos);                                      //drives motor to calculated position
  }
  delay(1000);
}

void loop() {

  for (int k = 0; k < 100; k++) { // runs 100 loops and gets average center location and r platform value
    // grab blocks!
    pixy.ccc.getBlocks();

    // If there are 4 markers detected, then collect and print the data
    if (pixy.ccc.numBlocks == 4) {
      //sets current X and Y co-ords of the center
      X = 0;
      Y = 0;
      for (int i = 0; i < 4; i++) {
        X = X + pixy.ccc.blocks[i].m_x;  // absolute X location of the center
        Y = Y + pixy.ccc.blocks[i].m_y;  // absolute Y location of the center
      }

      X = X / 4;
      Y = Y / 4;

      r_platform = 0;
      for (int i = 0; i < 4; i++) {
        // adds magnitude of vector pointing from the center of the platform to the center of the marker
        r_platform = r_platform + sqrt(pow(pixy.ccc.blocks[i].m_x - X, 2) + pow(pixy.ccc.blocks[i].m_y - Y, 2));  
        // adds magnitude of vector pointing from the center of the marker to the corner of the marker
        r_platform = r_platform + sqrt(pow(pixy.ccc.blocks[i].m_width, 2) + pow(pixy.ccc.blocks[i].m_height, 2))*0.5;
      }

      r_platform = r_platform / 4;  //calculates average r_max

      X_total = X_total + X;
      Y_total = Y_total + Y;
      r_platform_total = r_platform_total + r_platform;

      j++;

    }
    // If there are not 4 balls detected, then print so
    else {
    }
  }

  X_total = X_total / j;
  Y_total = Y_total / j;
  r_platform_total = r_platform_total / j;

  Serial.println((String) "CENTER: [" + int(X_total) + ", " + int(Y_total) + "]");  //prints location of the center
  Serial.println((String) "r platform: " + int(r_platform_total));                  //prints location of the center

  while (1) {}
}
