/*
 * CardioEdgeAI — ESP32 real-time ECG arrhythmia classification
 * (TensorFlow Lite Micro, full-integer INT8 model from Phase 5)
 *
 * Pipeline on device (mirrors training exactly):
 *   ADC @ 360 Hz -> 0.5-40 Hz biquad bandpass -> z-score over 10 s rolling
 *   buffer -> derivative R-peak detector (adaptive threshold + refractory)
 *   -> on each new R peak: classify the PREVIOUS beat (its window is
 *   complete and post-RR is now known) -> class N/S/V on Serial.
 *
 * The one-beat latency comes from post-RR (the model's strongest S-class
 * feature); it is the same trade-off documented in the thesis (Phase 4).
 *
 * Hardware (board bring-up):
 *   ESP32-S3 DevKitC + AD8232 single-lead ECG front end
 *   AD8232 OUT -> GPIO4 (only pin this sketch uses)
 *
 * Library: TensorFlowLite_ESP32 (Arduino Library Manager)
 * Model:   model.h next to this sketch
 *          (regenerate: py deploy/tflite/export_model_header.py)
 */
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"

// ---------- configuration ----------
constexpr int   kSampleRate      = 360;    // Hz (MIT-BIH native rate)
constexpr int   kWindowBefore    = 144;    // 0.4 s
constexpr int   kWindowAfter     = 216;    // 0.6 s
constexpr int   kWindowLen       = kWindowBefore + kWindowAfter;  // 360
constexpr float kZscoreBufferSec = 10.0f;  // rolling normalization window
constexpr int   kZscoreLen       = (int)(kZscoreBufferSec * kSampleRate);
constexpr int   kAdcPin          = 4;      // AD8232 OUT
constexpr int   kTensorArenaSize = 48 * 1024;
// Fixed RR normalization from training (src/models/cnn.py)
constexpr float kRrMean = 0.8f, kRrStd = 0.3f;

// ---------- TFLite Micro ----------
tflite::AllOpsResolver    resolver;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor*             sig_input  = nullptr;
TfLiteTensor*             rr_input   = nullptr;
TfLiteTensor*             output     = nullptr;
static uint8_t            tensor_arena[kTensorArenaSize];

// ---------- signal state ----------
float g_ecg[kWindowLen];          // window being collected (current beat)
float g_prev[kWindowLen];         // completed window of previous beat
float g_hist[kZscoreLen];         // rolling history for z-score
int   g_hist_len = 0;
unsigned long g_lastSampleMs = 0;
int   g_sampleIdx = -999;         // position within current window collection
float g_lastVal = 0, g_deriv = 0;

// R-peak bookkeeping (ms timestamps; ring of last 6 for local RR)
unsigned long g_r_ms[6]; int g_r_n = 0;
constexpr unsigned long kRefractoryMs = 250;

// Band-pass state (biquads @ 360 Hz: 0.5 Hz HP + 40 Hz LP, Q ~ 0.707)
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
    Serial.println("AllocateTensors failed - increase kTensorArenaSize");
    while (true) {}
  }
  // input(0) = beat window (360,1) int8; input(1) = RR features (4,) int8
  sig_input = interpreter->input(0);
  rr_input  = interpreter->input(1);
  output    = interpreter->output(0);
  Serial.println("CardioEdgeAI ready. Classes: 0=N 1=S 2=V");
}

float normRr(float sec)  { return (sec - kRrMean) / kRrStd; }
float meanRr(int from, int to) {  // mean of g_r_ms[from..to), seconds
  float s = 0; int n = 0;
  for (int i = from; i < to; i++, n++) s += (g_r_ms[i] - g_r_ms[i - 1]) / 1000.0f;
  return n ? s / n : kRrMean;
}

void classifyPrevBeat(const float* window, unsigned long now) {
  // RR features of the PREVIOUS beat, now that its post-RR is known:
  // g_r_ms[g_r_n-2] = previous beat, g_r_ms[g_r_n-1] = current beat.
  int cur = g_r_n - 1, prev = g_r_n - 2;
  float pre  = (prev - 1 >= 0) ? (g_r_ms[prev] - g_r_ms[prev - 1]) / 1000.0f : kRrMean;
  float post = (g_r_ms[cur] - g_r_ms[prev]) / 1000.0f;
  int lo = max(0, prev - 5);
  float local = meanRr(max(lo + 1, 1), prev + 1);
  float rr[4] = { normRr(pre), normRr(post), normRr(local),
                  local > 0 ? (pre / local - 1.0f) / kRrStd : 0.0f };

  // z-score the window with the rolling 10 s buffer statistics
  float mean = 0, var = 0;
  for (int i = 0; i < g_hist_len; i++) mean += g_hist[i];
  mean /= max(g_hist_len, 1);
  for (int i = 0; i < g_hist_len; i++)
    var += (g_hist[i] - mean) * (g_hist[i] - mean);
  float sd = sqrtf(var / max(g_hist_len, 1)) + 1e-9f;

  // quantize both inputs to int8
  float sscale = sig_input->params.scale == 0 ? 1.0f : sig_input->params.scale;
  int   szp    = sig_input->params.zero_point;
  for (int i = 0; i < kWindowLen; i++) {
    float v = (window[i] - mean) / sd;
    sig_input->data.int8[i] =
        (int8_t)constrain(lroundf(v / sscale) + szp, -128, 127);
  }
  float rscale = rr_input->params.scale == 0 ? 1.0f : rr_input->params.scale;
  int   rzp    = rr_input->params.zero_point;
  for (int i = 0; i < 4; i++)
    rr_input->data.int8[i] =
        (int8_t)constrain(lroundf(rr[i] / rscale) + rzp, -128, 127);

  if (interpreter->Invoke() != kTfLiteOk) { Serial.println("invoke failed"); return; }
  int best = 0;
  for (int c = 1; c < 3; c++)
    if (output->data.int8[c] > output->data.int8[best]) best = c;
  float oscale = output->params.scale == 0 ? 1.0f : output->params.scale;
  float conf = (output->data.int8[best] - output->params.zero_point) * oscale;
  const char* names[3] = { "N", "S", "V" };
  Serial.printf("BEAT %s conf=%.2f rr_pre=%.2fs rr_post=%.2fs\n",
                names[best], conf, pre, post);
}

void loop() {
  unsigned long now = millis();
  if (now - g_lastSampleMs < (1000UL / kSampleRate)) return;  // 360 Hz tick
  g_lastSampleMs = now;

  float raw = analogReadMilliVolts(kAdcPin) * 0.001f;
  float filtered = runBiquad(g_lp, runBiquad(g_hp, raw));

  if (g_hist_len < kZscoreLen) {
    g_hist[g_hist_len++] = filtered;
  } else {
    memmove(g_hist, g_hist + 1, (kZscoreLen - 1) * sizeof(float));
    g_hist[kZscoreLen - 1] = filtered;
  }

  // derivative + adaptive-threshold R detector
  float d = filtered - g_lastVal;
  g_lastVal = filtered;
  g_deriv = 0.8f * g_deriv + 0.2f * fabsf(d);
  float threshold = 3.5f * g_deriv;
  bool rPeak = (fabsf(d) > threshold) && (now - g_r_ms[g_r_n ? g_r_n - 1 : 0]
                                          > kRefractoryMs);

  if (rPeak) {
    // classify the previous beat now that its post-RR is known
    if (g_sampleIdx >= kWindowAfter) classifyPrevBeat(g_prev, now);
    // bookkeeping: shift timestamps, start collecting the new window
    if (g_r_n < 6) g_r_ms[g_r_n++] = now;
    else { memmove(g_r_ms, g_r_ms + 1, 5 * sizeof(unsigned long));
           g_r_ms[5] = now; }
    g_sampleIdx = -(kWindowBefore - 1);
  }

  if (g_sampleIdx > -kWindowBefore && g_sampleIdx < kWindowAfter) {
    g_ecg[kWindowBefore - 1 + g_sampleIdx] = filtered;
    g_sampleIdx++;
    if (g_sampleIdx == kWindowAfter) {   // window complete -> stage it
      memcpy(g_prev, g_ecg, sizeof(g_ecg));
      g_sampleIdx = kWindowAfter + 1;    // done until next R peak
    }
  }
}
