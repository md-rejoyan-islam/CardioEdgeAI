/*
 * CardioEdgeAI — ESP32 real-time ECG arrhythmia classification
 * (TensorFlow Lite Micro, INT8 model from Phase 5)
 *
 * Pipeline on device:
 *   ADC (or Serial input) @ 360 Hz -> 0.5-40 Hz bandpass (biquad cascade)
 *   -> moving-threshold R-peak detector -> 360-sample beat window
 *   -> z-score normalize over a 10 s rolling buffer -> INT8 quantize
 *   -> TFLite Micro inference -> class N/S/V + confidence on Serial
 *
 * Hardware (Phase 6 board bring-up):
 *   - ESP32-S3 DevKitC (any ESP32 variant works, adjust pins)
 *   - AD8232 single-lead ECG front end: OUT -> GPIO4, LO+/LO- -> GPIO5/6
 *     (only OUT is used by this sketch; lead-off pins are informational)
 *
 * Library: TensorFlowLite_ESP32 (Arduino Library Manager)
 * Model:   model.h next to this sketch (see deploy/tflite/export_model_header.py)
 */
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"

// ---------- configuration ----------
constexpr int   kSampleRate      = 360;   // Hz (MIT-BIH native rate)
constexpr int   kWindowBefore    = 144;   // 0.4 s
constexpr int   kWindowAfter     = 216;   // 0.6 s
constexpr int   kWindowLen       = kWindowBefore + kWindowAfter;  // 360
constexpr float kZscoreBufferSec = 10.0f; // rolling normalization window
constexpr int   kZscoreLen       = (int)(kZscoreBufferSec * kSampleRate);
constexpr int   kAdcPin          = 4;     // AD8232 OUT
constexpr int   kTensorArenaSize = 40 * 1024;  // INT8 13k-param model fits easily

// ---------- TFLite Micro setup ----------
tflite::AllOpsResolver       resolver;
const tflite::Model*         model = nullptr;
tflite::MicroInterpreter*    interpreter = nullptr;
TfLiteTensor*                input  = nullptr;
TfLiteTensor*                output = nullptr;
static uint8_t               tensor_arena[kTensorArenaSize];

// ---------- signal state ----------
float          g_ecg[kWindowLen];          // current beat window
float          g_hist[kZscoreLen];        // rolling history for z-score
int            g_hist_len = 0;
unsigned long  g_lastSampleMs = 0;
int            g_sampleIdx = 0;
float          g_lastVal = 0, g_deriv = 0;
float          g_threshold = 0;
unsigned long  g_lastBeatMs = 0;
constexpr unsigned long kRefractoryMs = 250;  // min distance between R peaks

// Band-pass state (2x biquad: 0.5 Hz HP + 40 Hz LP, computed for 360 Hz)
struct Biquad { float b0, b1, b2, a1, a2, x1, x2, y1, y2; };
Biquad g_hp = { 0.95871f, -1.91741f, 0.95871f, -1.95720f, 0.87737f, 0, 0, 0, 0 };
Biquad g_lp = { 0.00554f, 0.01109f, 0.00554f, 1.77812f, 0.80036f, 0, 0, 0, 0 };

float runBiquad(Biquad& f, float x) {
  float y = f.b0 * x + f.b1 * f.x1 + f.b2 * f.x2 - f.a1 * f.y1 - f.a2 * f.y2;
  f.x2 = f.x1; f.x1 = x; f.y2 = f.y1; f.y1 = y;
  return y;
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(kAdcPin, ADC_11db);

  model = tflite::GetModel(cardioedge_model_data);
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed — increase kTensorArenaSize");
    while (true) {}
  }
  input  = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("CardioEdgeAI ready. Classes: 0=N 1=S 2=V");
}

void classifyBeat(const float* window) {
  // z-score normalize with the rolling 10 s buffer statistics
  float mean = 0, var = 0;
  for (int i = 0; i < g_hist_len; i++) mean += g_hist[i];
  mean /= max(g_hist_len, 1);
  for (int i = 0; i < g_hist_len; i++) var += (g_hist[i] - mean) * (g_hist[i] - mean);
  float std_ = sqrtf(var / max(g_hist_len, 1)) + 1e-9f;

  // fill INT8 input tensor
  float scale = input->params.scale == 0 ? 1.0f : input->params.scale;
  int zero_point = input->params.zero_point;
  int8_t* in = input->data.int8;
  for (int i = 0; i < kWindowLen; i++) {
    float v = (window[i] - mean) / std_;
    in[i] = (int8_t)constrain(roundf(v / scale) + zero_point, -128, 127);
  }
  // keep a copy of the normalized window for the history after inference
  if (interpreter->Invoke() != kTfLiteOk) { Serial.println("invoke failed"); return; }

  int best = 0;
  for (int c = 1; c < 3; c++)
    if (output->data.int8[c] > output->data.int8[best]) best = c;
  // dequantize confidence for display
  float oscale = output->params.scale == 0 ? 1.0f : output->params.scale;
  float conf = (output->data.int8[best] - output->params.zero_point) * oscale;
  const char* names[3] = { "N", "S", "V" };
  Serial.printf("BEAT %s conf=%.2f hr=%.0f\n", names[best], conf,
                60000.0f / max((unsigned long)1, millis() - g_lastBeatMs));
}

void loop() {
  unsigned long now = millis();
  if (now - g_lastSampleMs < (1000UL / kSampleRate)) return;  // 360 Hz tick
  g_lastSampleMs = now;

  float raw = analogReadMilliVolts(kAdcPin) * 0.001f;
  float filtered = runBiquad(g_lp, runBiquad(g_hp, raw));

  // rolling history for normalization
  if (g_hist_len < kZscoreLen) {
    g_hist[g_hist_len++] = filtered;
  } else {
    memmove(g_hist, g_hist + 1, (kZscoreLen - 1) * sizeof(float));
    g_hist[kZscoreLen - 1] = filtered;
  }

  // simple derivative + adaptive threshold R detector
  float d = filtered - g_lastVal;
  g_lastVal = filtered;
  g_deriv = 0.8f * g_deriv + 0.2f * fabsf(d);
  g_threshold = 3.5f * g_deriv;  // adaptive noise floor multiple
  bool rPeak = (fabsf(d) > g_threshold) && (now - g_lastBeatMs > kRefractoryMs);

  if (rPeak) {
    g_lastBeatMs = now;
    g_sampleIdx = -(kWindowBefore - 1);  // start collecting the window
  }
  if (g_sampleIdx > -kWindowBefore && g_sampleIdx < kWindowAfter) {
    g_ecg[kWindowBefore - 1 + g_sampleIdx] = filtered;
    g_sampleIdx++;
    if (g_sampleIdx == kWindowAfter) classifyBeat(g_ecg);
  }
}
