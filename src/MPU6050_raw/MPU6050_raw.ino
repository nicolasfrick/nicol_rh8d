/*
  MPU6050 Raw with Static Rotation
*/
#include "I2Cdev.h"
#include "MPU6050.h"

MPU6050 mpu;

#define OUTPUT_READABLE_ACCELGYRO
//#define OUTPUT_BINARY_ACCELGYRO

int16_t ax, ay, az;
int16_t gx, gy, gz;
int16_t ax_rot, ay_rot, az_rot;  // Rotated values
int16_t gx_rot, gy_rot, gz_rot;
bool blinkState;

void setup() {
  #if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
    Wire.begin(); 
  #elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
    Fastwire::setup(400, true);
  #endif

  Serial.begin(38400);
  mpu.initialize();
  if (!mpu.testConnection()) {
//     Serial.println("MPU6050 connection failed");
    while (true);
  }

  // Set calibration offsets (for bias, not rotation)
  mpu.setXAccelOffset(0);
  mpu.setYAccelOffset(0);
  mpu.setZAccelOffset(0);
  mpu.setXGyroOffset(0);
  mpu.setYGyroOffset(0);
  mpu.setZGyroOffset(0);

  pinMode(LED_BUILTIN, OUTPUT);
//  Serial.println("Done setup");
}

void apply_static_rotation(int16_t &ax_in, int16_t &ay_in, int16_t &az_in,
                          int16_t &gx_in, int16_t &gy_in, int16_t &gz_in,
                          int16_t &ax_out, int16_t &ay_out, int16_t &az_out,
                          int16_t &gx_out, int16_t &gy_out, int16_t &gz_out) {
  // Apply -90° X-axis rotation
  // R = [1  0  0]   New X = X
  //     [0  0  1]   New Y = Z
  //     [0 -1  0]   New Z = -Y
  ax_out = ax_in;       // X unchanged
  ay_out = az_in;       // Y = Z
  az_out = -ay_in;      // Z = -Y
  
  gx_out = gx_in;       // Angular rate X unchanged
  gy_out = gz_in;       // Angular rate Y = Z
  gz_out = -gy_in;      // Angular rate Z = -Y
}

void loop() {
//  Serial.println("Running");
//  delay(100);
  if (Serial.available()) 
  { 
    if (Serial.readString() == "r")
    {
      // Read raw accel/gyro data
      mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

      // Apply static rotation
      apply_static_rotation(ax, ay, az, gx, gy, gz,
                            ax_rot, ay_rot, az_rot, gx_rot, gy_rot, gz_rot);

      // Output rotated data
      #ifdef OUTPUT_READABLE_ACCELGYRO
        Serial.print("a/g:\t");
        Serial.print(ax_rot); Serial.print("\t");
        Serial.print(ay_rot); Serial.print("\t");
        Serial.print(az_rot); Serial.print("\t");
        Serial.print(gx_rot); Serial.print("\t");
        Serial.print(gy_rot); Serial.print("\t");
        Serial.println(gz_rot);
      #endif

      #ifdef OUTPUT_BINARY_ACCELGYRO
        Serial.write((uint8_t)(ax_rot >> 8)); Serial.write((uint8_t)(ax_rot & 0xFF));
        Serial.write((uint8_t)(ay_rot >> 8)); Serial.write((uint8_t)(ay_rot & 0xFF));
        Serial.write((uint8_t)(az_rot >> 8)); Serial.write((uint8_t)(az_rot & 0xFF));
        Serial.write((uint8_t)(gx_rot >> 8)); Serial.write((uint8_t)(gx_rot & 0xFF));
        Serial.write((uint8_t)(gy_rot >> 8)); Serial.write((uint8_t)(gy_rot & 0xFF));
        Serial.write((uint8_t)(gz_rot >> 8)); Serial.write((uint8_t)(gz_rot & 0xFF));
      #endif

      blinkState = !blinkState;
      digitalWrite(LED_BUILTIN, blinkState);
    }
  }
}
