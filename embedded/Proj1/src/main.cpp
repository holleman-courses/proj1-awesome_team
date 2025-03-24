/*
  Active Learning Labs
  Harvard University 
  tinyMLx - OV7675 Camera Test

*/

#include <TinyMLShield.h>

bool commandRecv = false; // flag used for indicating receipt of commands from serial port
bool liveFlag = false; // flag as true to live stream raw camera bytes, set as false to take single images on command
bool captureFlag = false;

// Image buffer;
byte image[176 * 144]; // QCIF: 176x144 x 2 bytes per pixel (RGB565)
int bytesPerFrame;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  initializeShield();

  // Initialize the OV7675 camera
  if (!Camera.begin(QCIF, GRAYSCALE, 5, OV7675)) {
    Serial.println("Failed to initialize camera");
    while (1);
  }
  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();
}

void loop() {

  bool clicked = readShieldButton();
  if (clicked) {
    if (!liveFlag) {
      if (!captureFlag) {
        captureFlag = true;
      }
    }
  }

  // Read incoming commands from serial monitor

    if (captureFlag) {
      captureFlag = false;
      Camera.autoGain();
      Camera.autoExposure();
      Camera.readFrame(image);
      //Serial.println("\nImage data will be printed out in 3 seconds...");
      Serial.print("START");
      Serial.write(image, bytesPerFrame);
      Serial.print("END");
      // for (int i = 0; i < bytesPerFrame - 1; i += 2) {
      //   Serial.print("0x");
      //   Serial.print(image[i+1], HEX);
      //   Serial.print(image[i], HEX);
      //   if (i != bytesPerFrame - 2) {
      //     Serial.print(", ");
      //   }
      // }
    }
}
