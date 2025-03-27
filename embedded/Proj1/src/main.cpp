#include "object_recognition.h"

#include "mbed.h"
#include <Arduino_OV767X.h>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

static const char *label[] PROGMEM = {"Pencil", "No Pencil"};
static int32_t bytes_per_frame;
static int32_t bytes_per_pixel;
static bool debug_application = false;

static uint8_t data[160 * 120 * 2]; // QQVGA: 160x120 X 2 bytes per pixel (YUV422)

// Resolutions of the cropped image
static int32_t height_i = 0; // Initialized in the setup() function
static int32_t width_i  = 0; // Initialized in the setup() function

// Resolution of TensorFlow Lite input model
static int32_t height_o = 48;
static int32_t width_o  = 48;

// Scaling factors required by the resize operator
static float scale_x = 0.0f;  // Initialized in the setup() function
static float scale_y = 0.0f;  // Initialized in the setup() function

// Strides of the cropped image
static int32_t stride_x = 0; // Initialized in the setup() function
static int32_t stride_y = 0; // Initialized in the setup() function

template <typename T>
inline T clamp_0_255(T x) {
  return std::max(std::min(x, static_cast<T>(255)), static_cast<T>(0));
}

inline uint8_t bilinear(uint8_t v00, uint8_t v01, uint8_t v10, uint8_t v11, float xi_f, float yi_f) {
  const float xi = (int32_t)std::floor(xi_f);
  const float yi = (int32_t)std::floor(yi_f);
  const float wx1 = (xi_f - xi);
  const float wx0 = (1.0f - wx1);
  const float wy1 = (yi_f - yi);
  const float wy0 = (1.0f - wy1);

  float res = 0;
  res += (v00 * wx0 * wy0);
  res += (v01 * wx1 * wy0);
  res += (v10 * wx0 * wy1);
  res += (v11 * wx1 * wy1);

  return clamp_0_255(res);
}

inline float rescale(float x, float scale, float offset) {
  return (x * scale) - offset;
}

inline int8_t quantize(float x, float scale, float zero_point) {
  return (x / scale) + zero_point;
}

// TensorFlow Lite for Microcontroller global variables
static const tflite::Model* tflu_model            = nullptr;
static tflite::MicroInterpreter* tflu_interpreter = nullptr;
static TfLiteTensor* tflu_i_tensor                = nullptr;
static TfLiteTensor* tflu_o_tensor                = nullptr;

static constexpr int tensor_arena_size = 128000;
static uint8_t *tensor_arena = nullptr;


static float   tflu_scale     = 0.0f;
static int32_t tflu_zeropoint = 0;

void tflu_initialization() {
  Serial.println("TFLu initialization - start");

  tensor_arena = new uint8_t[tensor_arena_size];

  // Load the TFLITE model
  tflu_model = tflite::GetModel(model_tflite);
  if (tflu_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print(tflu_model->version());
    Serial.println("");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.println("");
    while(1);
  }

  tflite::AllOpsResolver tflu_ops_resolver;

  // Initialize the TFLu interpreter
  static tflite::MicroInterpreter static_interpreter(
        tflu_model,
        tflu_ops_resolver,
        tensor_arena,
        tensor_arena_size);
    tflu_interpreter = &static_interpreter;

  // Allocate TFLu internal memory
  tflu_interpreter->AllocateTensors();

  // Get the pointers for the input and output tensors
  tflu_i_tensor = tflu_interpreter->input(0);
  tflu_o_tensor = tflu_interpreter->output(0);

  const auto* i_quant = reinterpret_cast<TfLiteAffineQuantization*>(tflu_i_tensor->quantization.params);

  // Get the quantization parameters (per-tensor quantization)
  tflu_scale     = i_quant->scale->data[0];
  tflu_zeropoint = i_quant->zero_point->data[0];

  Serial.println("TFLu initialization - completed");
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!Camera.begin(QCIF, GRAYSCALE, 1)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }

  bytes_per_pixel = Camera.bytesPerPixel();
  bytes_per_frame = Camera.width() * Camera.height() * bytes_per_pixel;

  // Initialize TFLu
  tflu_initialization();

  // Initialize resolution
  // Resolutions of the cropped image
  height_i = Camera.height();
  width_i  = Camera.width();

  if(debug_application) {
    height_o = Camera.height();
    width_o  = Camera.height();
    Camera.testPattern();
  }

  // Strides of the cropped image
  stride_x = bytes_per_pixel;
  stride_y = Camera.width() * bytes_per_pixel;

  // Initialize the scaling factors required by the resize operator
  scale_x = (float)width_i / (float)width_o;
  scale_y = (float)height_i / (float)height_o;
}

void loop() {
  // Read the frame into 'data'
  Camera.readFrame(data);  // 'data' now holds 176*144 grayscale values (1 byte per pixel)

  int32_t idx = 0;  // index for storing processed pixels into the input tensor
  for (int32_t yo = 0; yo < height_o; yo++) {
    // Compute the corresponding y-coordinate in the input image
    const float yi_f = yo * scale_y;
    const int32_t yi = (int32_t)std::floor(yi_f);

    for (int32_t xo = 0; xo < width_o; xo++) {
      // Compute the corresponding x-coordinate in the input image
      const float xi_f = xo * scale_x;
      const int32_t xi = (int32_t)std::floor(xi_f);

      // Determine the positions for bilinear interpolation
      const int32_t x0 = xi;
      const int32_t y0 = yi;
      const int32_t x1 = std::min(xi + 1, width_i - 1);
      const int32_t y1 = std::min(yi + 1, height_i - 1);

      // Calculate offsets for the four neighboring pixels (1 byte per pixel)
      const int32_t off00 = x0 + y0 * width_i;
      const int32_t off01 = x1 + y0 * width_i;
      const int32_t off10 = x0 + y1 * width_i;
      const int32_t off11 = x1 + y1 * width_i;

      // Get grayscale pixel values from the input image
      uint8_t v00 = data[off00];
      uint8_t v01 = data[off01];
      uint8_t v10 = data[off10];
      uint8_t v11 = data[off11];

      // Use bilinear interpolation to calculate the output pixel value
      uint8_t pixel = bilinear(v00, v01, v10, v11, xi_f, yi_f);

      // Normalize the pixel value to the range expected by the model and quantize
      float normalized = rescale((float)pixel, 1.f/255.f, -1.f);
      int8_t quantized = quantize(normalized, tflu_scale, tflu_zeropoint);

      // Store the quantized value into the model's input tensor
      tflu_i_tensor->data.int8[idx++] = quantized;
    }
  }

  // Run inference with the TensorFlow Lite Micro interpreter
  TfLiteStatus invoke_status = tflu_interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Error invoking the TFLu interpreter");
    return;
  }

  // Find the label with the highest probability from the output tensor

  // Get the output tensor values
  float out = tflu_o_tensor->data.f[0];
  Serial.print("Output: ");
  Serial.println(out);
}