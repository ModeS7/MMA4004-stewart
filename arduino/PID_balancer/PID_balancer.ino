/*
  Stewart Platform - Ball Balancing (PERFORMANCE OPTIMIZED)
  Fixes blocking Pixy2 calls
*/

#include "PololuMaestro.h"
#include <Pixy2SPI_SS.h>
#include <math.h>

#define maestroSerial SERIAL_PORT_HARDWARE_OPEN

Pixy2SPI_SS pixy;
MicroMaestro maestro(maestroSerial);

// ============================================================================
// SERVO CONFIGURATION
// ============================================================================

const float abs_0 = 4000.0f;
const float abs_90 = 8000.0f;

const float range[6][2] = {
  {-45, 45}, {45, -45},
  {-45, 45}, {45, -45},
  {-45, 45}, {45, -45}
};

const float offset[6] = {-0.5, 7, -2, 4, -2.5, 5.5};

// ============================================================================
// SERVO CONTROL SETTINGS
// ============================================================================

const int SERVO_SPEED_BALANCING = 0;    // 0 = unlimited (fastest)
const int SERVO_ACCEL_BALANCING = 0;
const int SERVO_SPEED_HOME = 20;
const int SERVO_ACCEL_HOME = 20;

// ============================================================================
// TIMING CONTROL
// ============================================================================

const unsigned long CONTROL_LOOP_INTERVAL = 20;  // Target 50Hz (20ms)
unsigned long last_control_update = 0;

// ============================================================================
// STEWART PLATFORM GEOMETRY
// ============================================================================

const float HORN_LENGTH = 31.75f;
const float ROD_LENGTH = 145.0f;
const float BASE_RADIUS = 73.025f;
const float BASE_ANCHORS_OFFSET = 36.8893f;
const float PLATFORM_RADIUS = 67.775f;
const float PLATFORM_ANCHORS_OFFSET = 12.7f;

float base_anchors[6][3];
float platform_anchors[6][3];
float beta_angles[6];
float home_height_anchor_center;

// ============================================================================
// PIXY2 CONFIGURATION
// ============================================================================

const float pixy_origin[2] = {144.0f, 108.0f};
const float r_platform = 121.0f;
float ball[2] = {0, 0};
bool ball_detected = false;

// ============================================================================
// PID CONFIGURATION
// ============================================================================

float error[2] = {0, 0};
float error_prev[2] = {0, 0};
float integral[2] = {0, 0};
float deriv[2] = {0, 0};

const float kp = 0.5f;
const float ki = 0.0f;
const float kd = 5.0f;
const float r_max = 20.0f;
const float integral_limit = 50.0f;

float out[2] = {0, 0};
unsigned long time_prev = 0;

// Performance monitoring
unsigned long loop_count = 0;
unsigned long last_perf_print = 0;

// ============================================================================
// QUATERNION AND VECTOR MATH
// ============================================================================

struct Quaternion {
  float w, x, y, z;
};

void euler_to_quaternion(float rx, float ry, float rz, Quaternion& q) {
  float cy = cos(rz * 0.5f);
  float sy = sin(rz * 0.5f);
  float cp = cos(ry * 0.5f);
  float sp = sin(ry * 0.5f);
  float cr = cos(rx * 0.5f);
  float sr = sin(rx * 0.5f);

  q.w = cr * cp * cy + sr * sp * sy;
  q.x = sr * cp * cy - cr * sp * sy;
  q.y = cr * sp * cy + sr * cp * sy;
  q.z = cr * cp * sy - sr * sp * cy;
}

void rotate_vector(const float v[3], const Quaternion& q, float result[3]) {
  float w = q.w, x = q.x, y = q.y, z = q.z;
  float vx = v[0], vy = v[1], vz = v[2];

  result[0] = vx * (w*w + x*x - y*y - z*z) + vy * (2*x*y - 2*w*z) + vz * (2*x*z + 2*w*y);
  result[1] = vx * (2*x*y + 2*w*z) + vy * (w*w - x*x + y*y - z*z) + vz * (2*y*z - 2*w*x);
  result[2] = vx * (2*x*z - 2*w*y) + vy * (2*y*z + 2*w*x) + vz * (w*w - x*x - y*y + z*z);
}

float dot_product(const float a[3], const float b[3]) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

// ============================================================================
// STEWART PLATFORM INITIALIZATION
// ============================================================================

void calculate_home_coordinates(float l, float d, const float phi[3], float xy[6][3]) {
  float angels[2] = {-PI/2, PI/2};

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      int idx = i * 2 + j;
      xy[idx][0] = l * cos(phi[i]) + d * cos(phi[i] + angels[j]);
      xy[idx][1] = l * sin(phi[i]) + d * sin(phi[i] + angels[j]);
      xy[idx][2] = 0;
    }
  }
}

void init_stewart_geometry() {
  float base_angles[3] = {-PI/2, PI/6, PI*5/6};
  float temp_base[6][3];
  calculate_home_coordinates(BASE_RADIUS, BASE_ANCHORS_OFFSET, base_angles, temp_base);
  memcpy(base_anchors, temp_base, sizeof(base_anchors));

  float platform_angles[3] = {-PI*5/6, -PI/6, PI/2};
  float temp_platform[6][3];
  calculate_home_coordinates(PLATFORM_RADIUS, PLATFORM_ANCHORS_OFFSET, platform_angles, temp_platform);

  for (int i = 0; i < 6; i++) {
    int src_idx = (i + 1) % 6;
    platform_anchors[i][0] = temp_platform[src_idx][0];
    platform_anchors[i][1] = temp_platform[src_idx][1];
    platform_anchors[i][2] = temp_platform[src_idx][2];
  }

  beta_angles[0] = 0;
  beta_angles[1] = PI;

  float dx_23 = base_anchors[3][0] - base_anchors[2][0];
  float dy_23 = base_anchors[3][1] - base_anchors[2][1];
  float angle_23 = atan2(dy_23, dx_23);
  beta_angles[2] = angle_23;
  beta_angles[3] = angle_23 + PI;

  float dx_54 = base_anchors[4][0] - base_anchors[5][0];
  float dy_54 = base_anchors[4][1] - base_anchors[5][1];
  float angle_54 = atan2(dy_54, dx_54);
  beta_angles[5] = angle_54;
  beta_angles[4] = angle_54 + PI;

  float base_pos[3] = {base_anchors[0][0], base_anchors[0][1], base_anchors[0][2]};
  float platform_pos[3] = {platform_anchors[0][0], platform_anchors[0][1], platform_anchors[0][2]};

  float horn_end_x = base_pos[0] + HORN_LENGTH * cos(beta_angles[0]);
  float horn_end_y = base_pos[1] + HORN_LENGTH * sin(beta_angles[0]);

  float dx = platform_pos[0] - horn_end_x;
  float dy = platform_pos[1] - horn_end_y;
  float horiz_dist_sq = dx*dx + dy*dy;

  home_height_anchor_center = sqrt(ROD_LENGTH*ROD_LENGTH - horiz_dist_sq);
}

// ============================================================================
// INVERSE KINEMATICS
// ============================================================================

bool calculate_servo_angles(float tx, float ty, float tz,
                            float rx_deg, float ry_deg, float rz_deg,
                            float angles[6]) {
  float rx = rx_deg * PI / 180.0f;
  float ry = ry_deg * PI / 180.0f;
  float rz = rz_deg * PI / 180.0f;

  Quaternion quat;
  euler_to_quaternion(rx, ry, rz, quat);

  float translation[3] = {tx, ty, tz};

  for (int k = 0; k < 6; k++) {
    float rotated_anchor[3];
    rotate_vector(platform_anchors[k], quat, rotated_anchor);

    float p_world[3] = {
      translation[0] + rotated_anchor[0],
      translation[1] + rotated_anchor[1],
      translation[2] + rotated_anchor[2]
    };

    float leg[3] = {
      p_world[0] - base_anchors[k][0],
      p_world[1] - base_anchors[k][1],
      p_world[2] - base_anchors[k][2]
    };

    float leg_length_sq = dot_product(leg, leg);

    float e_k = 2 * HORN_LENGTH * leg[2];
    float f_k = 2 * HORN_LENGTH * (
      cos(beta_angles[k]) * leg[0] +
      sin(beta_angles[k]) * leg[1]
    );
    float g_k = leg_length_sq - (ROD_LENGTH*ROD_LENGTH - HORN_LENGTH*HORN_LENGTH);

    float sqrt_term = e_k*e_k + f_k*f_k;
    if (sqrt_term < 1e-6f) return false;

    float ratio = g_k / sqrt(sqrt_term);
    if (fabs(ratio) > 1.0f) return false;

    float alpha_k = asin(ratio) - atan2(f_k, e_k);
    angles[k] = -alpha_k * 180.0f / PI;

    if (fabs(angles[k]) > 40.0f) return false;
  }

  return true;
}

// ============================================================================
// SERVO CONTROL
// ============================================================================

void move_servos(const float angles[6], int spd, int acc) {
  for (int i = 0; i < 6; i++) {
    float pos = angles[i] + offset[i];
    pos = map(pos, range[i][0], range[i][1], abs_0, abs_90);
    maestro.setSpeed(i, spd);
    maestro.setAcceleration(i, acc);
    maestro.setTarget(i, pos);
  }
}

void move_to_home() {
  float angles[6];
  if (calculate_servo_angles(0, 0, home_height_anchor_center, 0, 0, 0, angles)) {
    move_servos(angles, SERVO_SPEED_HOME, SERVO_ACCEL_HOME);
  }
}

// ============================================================================
// PIXY2 BALL TRACKING (NON-BLOCKING)
// ============================================================================

void find_ball() {
  // Get blocks without blocking
  int8_t result = pixy.ccc.getBlocks(false, CCC_SIG1, 1);  // Non-blocking, signature 1, max 1 block

  if (result > 0 && pixy.ccc.numBlocks == 1) {
    ball[0] = pixy.ccc.blocks[0].m_x;
    ball[1] = pixy.ccc.blocks[0].m_y;
    ball_detected = true;
  } else {
    ball[0] = 4004;
    ball[1] = 4004;
    ball_detected = false;
  }
}

// ============================================================================
// PID CONTROL
// ============================================================================

void run_pid() {
  unsigned long time_now = millis();
  float delta_time_ms = time_now - time_prev;
  time_prev = time_now;

  if (delta_time_ms < 1) delta_time_ms = 1;
  float delta_time_sec = delta_time_ms / 1000.0f;

  error[0] = pixy_origin[0] - ball[0];
  error[1] = pixy_origin[1] - ball[1];

  float r_ball = sqrt(error[0]*error[0] + error[1]*error[1]);

  if (r_ball > r_platform) {
    integral[0] = 0;
    integral[1] = 0;
    move_to_home();
    return;
  }

  deriv[0] = (error[0] - error_prev[0]) / delta_time_ms;
  deriv[1] = (error[1] - error_prev[1]) / delta_time_ms;

  if (isnan(deriv[0]) || isinf(deriv[0])) deriv[0] = 0;
  if (isnan(deriv[1]) || isinf(deriv[1])) deriv[1] = 0;

  integral[0] += error[0] * delta_time_sec;
  integral[1] += error[1] * delta_time_sec;

  if (integral[0] > integral_limit) integral[0] = integral_limit;
  if (integral[0] < -integral_limit) integral[0] = -integral_limit;
  if (integral[1] > integral_limit) integral[1] = integral_limit;
  if (integral[1] < -integral_limit) integral[1] = -integral_limit;

  error_prev[0] = error[0];
  error_prev[1] = error[1];

  out[0] = error[0] * kp + integral[0] * ki + deriv[0] * kd;
  out[1] = error[1] * kp + integral[1] * ki + deriv[1] * kd;

  float r_out = sqrt(out[0]*out[0] + out[1]*out[1]);
  if (r_out > r_max) {
    out[0] = out[0] * (r_max / r_out);
    out[1] = out[1] * (r_max / r_out);
  }

  // Coordinate transformation
  float platform_tilt_x = out[1];
  float platform_tilt_y = out[0];

  float angles[6];
  if (calculate_servo_angles(0, 0, home_height_anchor_center,
                             platform_tilt_x, platform_tilt_y, 0, angles)) {
    move_servos(angles, SERVO_SPEED_BALANCING, SERVO_ACCEL_BALANCING);
  } else {
    move_to_home();
  }
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================

void setup() {
  Serial.begin(115200);
  maestroSerial.begin(9600);

  Serial.println("Initializing...");
  pixy.init();

  init_stewart_geometry();
  move_to_home();
  delay(1000);

  time_prev = millis();
  last_control_update = millis();
  last_perf_print = millis();

  Serial.println("Running!");
}

void loop() {
  unsigned long loop_start = micros();

  // Always read camera (fast, non-blocking)
  find_ball();

  // Rate-limited control updates
  unsigned long now = millis();
  if (now - last_control_update >= CONTROL_LOOP_INTERVAL) {
    last_control_update = now;

    if (ball_detected) {
      run_pid();
    } else {
      move_to_home();
    }
  }

  loop_count++;

  // Performance stats every 5 seconds
  if (now - last_perf_print >= 5000) {
    float loops_per_sec = loop_count / 5.0f;
    Serial.print("Loop rate: ");
    Serial.print(loops_per_sec);
    Serial.println(" Hz");
    loop_count = 0;
    last_perf_print = now;
  }

  // Small delay to prevent CPU hammering
  delayMicroseconds(500);
}