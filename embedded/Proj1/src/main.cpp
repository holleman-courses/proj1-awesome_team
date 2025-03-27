#include "object_recognition.h"

#include "mbed.h"
#include <Arduino_OV767X.h>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

static uint8_t data[176 * 144]; 

// TensorFlow Lite for Microcontroller global variables
static const tflite::Model* tflu_model            = nullptr;
static tflite::MicroInterpreter* tflu_interpreter = nullptr;
static TfLiteTensor* tflu_i_tensor                = nullptr;
static TfLiteTensor* tflu_o_tensor                = nullptr;

static constexpr int tensor_arena_size = 1024 * 180;
static uint8_t *tensor_arena = nullptr;


static float   tflu_scale     = 0.0f;
static int32_t tflu_zeropoint = 0;

void tflu_initialization() {
  Serial.println("TFLu initialization - start");

  

  // Load the TFLITE model
  tflu_model = tflite::GetModel(model_tflite);
  if (tflu_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print(tflu_model->version());
    Serial.println("");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.println("");
    while(1);
  }
  Serial.println("TFLu model loaded");
  tflite::AllOpsResolver tflu_ops_resolver;
  Serial.println("TFLu ops resolver created");
  tensor_arena = new uint8_t[tensor_arena_size];
  Serial.println("TFLu tensor arena created");
  // Initialize the TFLu interpreter
  static tflite::MicroInterpreter static_interpreter(
        tflu_model,
        tflu_ops_resolver,
        tensor_arena,
        tensor_arena_size);
    tflu_interpreter = &static_interpreter;
  Serial.println("TFLu interpreter created");
  // Allocate TFLu internal memory
  tflu_interpreter->AllocateTensors();
  Serial.println("TFLu memory allocated");
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

  if (!Camera.begin(QCIF, GRAYSCALE, 5)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }


  // Initialize TFLu
  tflu_initialization();

  int start = millis();
  tflu_interpreter->Invoke();
  int end = millis();
  Serial.print("Inference Time: ");
  Serial.print(end - start);


}

void loop() {
  Camera.readFrame(data);
  
  for (int idx = 0; idx < 176*144; idx++) {
    uint8_t pixel = data[idx];
    int8_t pixel_quant = pixel - 128;
    tflu_i_tensor->data.int8[idx] = pixel_quant;
  }

  TfLiteStatus invoke_status = tflu_interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Error invoking the TFLu interpreter");
    return;
  }

  // Get the output tensor values
  int8_t out = tflu_o_tensor->data.int8[0];
  String result = out > 0 ? "Pencil Detected" : "No Pencil Detected";

  Serial.println(result);
  Serial.print("Confidence: ");
  Serial.println(out);

}
