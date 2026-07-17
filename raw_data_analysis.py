

#include <WiFi.h>
#include <Wire.h>
#include <HTTPClient.h>
#include "time.h"

#include <Adafruit_ADS1X15.h>
#include <Adafruit_BMP280.h>
#include "MAX30105.h"


const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";


const char* SERVER_IP   = "YOUR_SERVER_IP";
const int   SERVER_PORT = 5000;

String URL_DATA   = String("http://") + SERVER_IP + ":" + SERVER_PORT + "/data";
String URL_MIC    = String("http://") + SERVER_IP + ":" + SERVER_PORT + "/mic";
String URL_PPG    = String("http://") + SERVER_IP + ":" + SERVER_PORT + "/ppg";
String URL_STATUS = String("http://") + SERVER_IP + ":" + SERVER_PORT + "/status";


const char* ntpServer          = "pool.ntp.org";
const long  gmtOffset_sec      = 7200;
const int   daylightOffset_sec = 3600;


#define SDA_PIN 5
#define SCL_PIN 6
#define MIC_PIN 0
#define LED_PIN 8


#define IMU_ADDR 0x68
#define BMP_ADDR 0x76


Adafruit_ADS1015 ads;
Adafruit_BMP280  bmp;
MAX30105         particleSensor;


bool adsOK        = false;
bool bmpOK        = false;
bool imuOK        = false;
bool maxOK        = false;
bool timeOK       = false;
bool serverActive = false;


int flexBaseline = 0;
int flexRaw      = 0;
int flexDelta    = 0;


const int FLEX_BLOCK_SIZE = 16;
int16_t flexRawBuffer[FLEX_BLOCK_SIZE];
int16_t flexDeltaBuffer[FLEX_BLOCK_SIZE];
int flexBufIndex = 0;


const int MIC_BLOCK_SIZE = 200;
int  micBuffer[MIC_BLOCK_SIZE];
int  micIndex      = 0;
bool micBlockReady = false;


const int PPG_BLOCK_SIZE = 60;
uint32_t ppgIrBuffer[PPG_BLOCK_SIZE];
uint32_t ppgRedBuffer[PPG_BLOCK_SIZE];
int  ppgIndex      = 0;


int16_t accX_raw = 0, accY_raw = 0, accZ_raw = 0;
int16_t gyroX_raw = 0, gyroY_raw = 0, gyroZ_raw = 0;

float accX = 0.0f, accY = 0.0f, accZ = 0.0f;
float gyroX = 0.0f, gyroY = 0.0f, gyroZ = 0.0f;

float gyroOffsetX = 0.0f, gyroOffsetY = 0.0f, gyroOffsetZ = 0.0f;

#define GYRO_HIST 5
float gyroX_hist[GYRO_HIST] = {0};
float gyroY_hist[GYRO_HIST] = {0};
float gyroZ_hist[GYRO_HIST] = {0};
int   gyro_hist_idx = 0;

float medianOf5(float* arr) {
  float tmp[GYRO_HIST];
  for (int i = 0; i < GYRO_HIST; i++) tmp[i] = arr[i];
  for (int i = 0; i < GYRO_HIST - 1; i++)
    for (int j = 0; j < GYRO_HIST - 1 - i; j++)
      if (tmp[j] > tmp[j+1]) { float t = tmp[j]; tmp[j] = tmp[j+1]; tmp[j+1] = t; }
  return tmp[GYRO_HIST / 2];
}


float temperatureC = NAN;
float pressurehPa  = NAN;


uint32_t redValue = 0;
uint32_t irValue  = 0;


unsigned long lastPulseRead   = 0;
unsigned long lastFlexRead    = 0;
unsigned long lastImuRead     = 0;
unsigned long lastBmpRead     = 0;
unsigned long lastMicRead     = 0;
unsigned long lastDataSend    = 0;
unsigned long lastStatusCheck = 0;

const unsigned long PULSE_INTERVAL_MS  = 80;
const unsigned long FLEX_INTERVAL_MS   = 250;
const unsigned long IMU_INTERVAL_MS    = 100;
const unsigned long BMP_INTERVAL_MS    = 1000;
const unsigned long MIC_INTERVAL_MS    = 5;
const unsigned long DATA_SEND_MS       = 2000;
const unsigned long STATUS_CHECK_MS    = 10000;


bool getTimeString(char* buf, size_t len) {
  struct tm ti;
  if (!getLocalTime(&ti)) return false;
  strftime(buf, len, "%H:%M:%S", &ti);
  return true;
}


void initWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Conectare WiFi");
  unsigned long t = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t < 20000) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("WiFi OK | IP: " + WiFi.localIP().toString());
    configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
    struct tm ti;
    if (getLocalTime(&ti, 8000)) {
      timeOK = true;
      Serial.println("NTP OK");
    }
  } else {
    Serial.println("WiFi FAIL");
  }
}


bool checkServer() {
  if (WiFi.status() != WL_CONNECTED) return false;
  HTTPClient http;
  http.begin(URL_STATUS);
  http.setTimeout(3000);
  int code = http.GET();
  bool active = false;
  if (code == 200) {
    String resp = http.getString();
    active = resp.indexOf("\"active\": true") >= 0 ||
             resp.indexOf("\"active\":true") >= 0;
  }
  http.end();
  return active;
}


bool sendCombined() {
  if (WiFi.status() != WL_CONNECTED) return false;

  char timeBuf[16] = "00:00:00";
  getTimeString(timeBuf, sizeof(timeBuf));

  String json;
  json.reserve(4000);


  json += "{\"t\":\"";    json += timeBuf;
  json += "\",\"fr\":";   json += flexRaw;
  json += ",\"fd\":";     json += flexDelta;


  if (flexBufIndex > 0) {
    json += ",\"fr_s\":\"";
    for (int i = 0; i < flexBufIndex; i++) {
      if (i > 0) json += ',';
      json += flexRawBuffer[i];
    }
    json += "\",\"fd_s\":\"";
    for (int i = 0; i < flexBufIndex; i++) {
      if (i > 0) json += ',';
      json += flexDeltaBuffer[i];
    }
    json += "\"";
    flexBufIndex = 0;
  }

  json += ",\"red\":";    json += redValue;
  json += ",\"ir\":";     json += irValue;
  json += ",\"ax\":";     json += String(accX, 4);
  json += ",\"ay\":";     json += String(accY, 4);
  json += ",\"az\":";     json += String(accZ, 4);
  json += ",\"gx\":";     json += String(gyroX, 4);
  json += ",\"gy\":";     json += String(gyroY, 4);
  json += ",\"gz\":";     json += String(gyroZ, 4);
  json += ",\"tmp\":";    json += String(temperatureC, 2);
  json += ",\"prs\":";    json += String(pressurehPa, 2);


  if (ppgIndex > 0) {
    json += ",\"ppg_ir\":\"";
    for (int i = 0; i < ppgIndex; i++) {
      if (i > 0) json += ',';
      json += ppgIrBuffer[i];
    }
    json += "\",\"ppg_red\":\"";
    for (int i = 0; i < ppgIndex; i++) {
      if (i > 0) json += ',';
      json += ppgRedBuffer[i];
    }
    json += "\"";
    ppgIndex = 0;
  }


  if (micBlockReady) {
    json += ",\"mic_s\":\"";
    for (int i = 0; i < MIC_BLOCK_SIZE; i++) {
      if (i > 0) json += ',';
      json += micBuffer[i];
    }
    json += "\"";
    micBlockReady = false;
  }

  json += "}";

  HTTPClient http;
  http.begin(URL_DATA);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(3000);
  int code = http.POST(json);
  http.end();
  return code == 200;
}


bool initADS() {
  if (!ads.begin()) return false;
  ads.setGain(GAIN_SIXTEEN);
  return true;
}

void calibrateFlex() {
  ads.setGain(GAIN_SIXTEEN);
  long sum = 0;
  for (int i = 0; i < 100; i++) {
    sum += ads.readADC_SingleEnded(0);
    delay(5);
  }
  flexBaseline = sum / 100;
}

void readFlex() {
  flexRaw   = ads.readADC_SingleEnded(0);
  flexDelta = flexRaw - flexBaseline;


}


void readMicSample() {
  micBuffer[micIndex++] = analogRead(MIC_PIN);
  if (micIndex >= MIC_BLOCK_SIZE) {
    micIndex      = 0;
    micBlockReady = true;
  }
}


bool initIMU() {
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  if (Wire.endTransmission(true) != 0) return false;
  delay(50);

  Wire.beginTransmission(IMU_ADDR);
  Wire.write(0x1C);
  Wire.write(0x00);
  if (Wire.endTransmission(true) != 0) return false;

  Wire.beginTransmission(IMU_ADDR);
  Wire.write(0x1B);
  Wire.write(0x00);
  if (Wire.endTransmission(true) != 0) return false;

  return true;
}

bool readIMU() {
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  Wire.requestFrom(IMU_ADDR, 14, true);
  if (Wire.available() < 14) return false;

  accX_raw  = (Wire.read() << 8) | Wire.read();
  accY_raw  = (Wire.read() << 8) | Wire.read();
  accZ_raw  = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();
  gyroX_raw = (Wire.read() << 8) | Wire.read();
  gyroY_raw = (Wire.read() << 8) | Wire.read();
  gyroZ_raw = (Wire.read() << 8) | Wire.read();

  accX  = accX_raw / 16384.0f;
  accY  = accY_raw / 16384.0f;
  accZ  = accZ_raw / 16384.0f;

  float gx = (gyroX_raw - gyroOffsetX) / 131.0f;
  float gy = (gyroY_raw - gyroOffsetY) / 131.0f;
  float gz = (gyroZ_raw - gyroOffsetZ) / 131.0f;

  gyroX_hist[gyro_hist_idx] = gx;
  gyroY_hist[gyro_hist_idx] = gy;
  gyroZ_hist[gyro_hist_idx] = gz;
  gyro_hist_idx = (gyro_hist_idx + 1) % GYRO_HIST;

  gyroX = medianOf5(gyroX_hist);
  gyroY = medianOf5(gyroY_hist);
  gyroZ = medianOf5(gyroZ_hist);

  return true;
}

void calibrateGyro() {
  Serial.println("Calibrare giroscop - stai nemiscat 3 secunde...");
  long sumX = 0, sumY = 0, sumZ = 0;
  const int samples = 300;

  for (int i = 0; i < samples; i++) {
    Wire.beginTransmission(IMU_ADDR);
    Wire.write(0x43);
    Wire.endTransmission(false);
    Wire.requestFrom(IMU_ADDR, 6, true);
    int16_t gx = (Wire.read() << 8) | Wire.read();
    int16_t gy = (Wire.read() << 8) | Wire.read();
    int16_t gz = (Wire.read() << 8) | Wire.read();
    sumX += gx;
    sumY += gy;
    sumZ += gz;
    delay(10);
  }

  gyroOffsetX = sumX / (float)samples;
  gyroOffsetY = sumY / (float)samples;
  gyroOffsetZ = sumZ / (float)samples;

  Serial.print("Offset gyro X="); Serial.print(gyroOffsetX / 131.0f, 3);
  Serial.print(" Y=");            Serial.print(gyroOffsetY / 131.0f, 3);
  Serial.print(" Z=");            Serial.println(gyroOffsetZ / 131.0f, 3);
}


bool initBMP() {
  if (!bmp.begin(BMP_ADDR)) return false;
  bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,
                  Adafruit_BMP280::SAMPLING_X2,
                  Adafruit_BMP280::SAMPLING_X16,
                  Adafruit_BMP280::FILTER_X16,
                  Adafruit_BMP280::STANDBY_MS_500);
  return true;
}

void readBMP() {
  temperatureC = bmp.readTemperature();
  pressurehPa  = bmp.readPressure() / 100.0f;
}


bool initMAX30102() {
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) return false;






  particleSensor.setup(60, 8, 2, 200, 411, 4096);
  particleSensor.setPulseAmplitudeRed(0x45);
  particleSensor.setPulseAmplitudeIR(0x45);
  particleSensor.setPulseAmplitudeGreen(0);
  delay(100);
  return true;
}

void readPulseRaw() {

  particleSensor.check();


  while (particleSensor.available()) {
    uint32_t r  = particleSensor.getRed();
    uint32_t ir = particleSensor.getIR();
    particleSensor.nextSample();


    redValue = r;
    irValue  = ir;


    if (ppgIndex < PPG_BLOCK_SIZE) {
      ppgIrBuffer[ppgIndex]  = ir;
      ppgRedBuffer[ppgIndex] = r;
      ppgIndex++;
    }


  }
}


void setup() {
  Serial.begin(115200);
  delay(1500);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  analogReadResolution(12);

  initWiFi();

  adsOK = initADS();
  Serial.println(adsOK ? "ADS1015 OK" : "ADS1015 FAIL");
  if (adsOK) {
    Serial.println("Calibrare flex - tine in pozitia de repaus...");
    delay(1500);
    calibrateFlex();
    Serial.print("Flex baseline = ");
    Serial.println(flexBaseline);
  }

  bmpOK = initBMP();
  Serial.println(bmpOK ? "BMP280 OK" : "BMP280 FAIL");

  imuOK = initIMU();
  Serial.println(imuOK ? "IMU OK" : "IMU FAIL");
  if (imuOK) {
    delay(500);
    calibrateGyro();
  }

  maxOK = initMAX30102();
  Serial.println(maxOK ? "MAX30102 OK" : "MAX30102 FAIL");

  Serial.println("\nAstept serverul Raspberry Pi...");
  Serial.println("(Ruleaza: python3 server.py subject_01 Nume)");

  while (!serverActive) {
    serverActive = checkServer();
    if (serverActive) {
      Serial.println("Server ACTIV! Incep inregistrarea.");
    } else {
      Serial.print(".");
      digitalWrite(LED_PIN, HIGH); delay(200);
      digitalWrite(LED_PIN, LOW);  delay(800);
    }
  }

  lastPulseRead   = millis();
  lastFlexRead    = millis();
  lastImuRead     = millis();
  lastBmpRead     = millis();
  lastMicRead     = millis();
  lastDataSend    = millis();
  lastStatusCheck = millis();

  Serial.println("READY – Inregistrare activa!");
}


void loop() {
  unsigned long now = millis();


  if (maxOK && now - lastPulseRead >= PULSE_INTERVAL_MS) {
    lastPulseRead = now;
    readPulseRaw();
  }


  if (adsOK && now - lastFlexRead >= FLEX_INTERVAL_MS) {
    lastFlexRead = now;
    readFlex();
    if (flexBufIndex < FLEX_BLOCK_SIZE) {
      flexRawBuffer[flexBufIndex]   = (int16_t)flexRaw;
      flexDeltaBuffer[flexBufIndex] = (int16_t)flexDelta;
      flexBufIndex++;
    }
  }


  if (imuOK && now - lastImuRead >= IMU_INTERVAL_MS) {
    lastImuRead = now;
    readIMU();
  }


  if (bmpOK && now - lastBmpRead >= BMP_INTERVAL_MS) {
    lastBmpRead = now;
    readBMP();
  }


  if (now - lastMicRead >= MIC_INTERVAL_MS) {
    lastMicRead = now;
    readMicSample();
  }


  if (now - lastDataSend >= DATA_SEND_MS) {
    lastDataSend = now;

    if (serverActive) {
      bool ok = sendCombined();

      if (ok) {
        static int sendCount = 0;
        sendCount++;
        if (sendCount % 60 == 0) {
          Serial.print("[SEND] ");
          Serial.print(sendCount);
          Serial.println(" POST-uri trimise");
        }
        digitalWrite(LED_PIN, HIGH); delay(5);
        digitalWrite(LED_PIN, LOW);
      } else {
        for (int i = 0; i < 3; i++) {
          digitalWrite(LED_PIN, HIGH); delay(50);
          digitalWrite(LED_PIN, LOW);  delay(50);
        }
        Serial.println("WARN: trimitere date esuata");
      }
    }
  }


  if (now - lastStatusCheck >= STATUS_CHECK_MS) {
    lastStatusCheck = now;
    serverActive = checkServer();
    if (!serverActive) {
      Serial.println("Server OPRIT. Astept repornire...");
      while (!serverActive) {
        serverActive = checkServer();
        digitalWrite(LED_PIN, HIGH); delay(200);
        digitalWrite(LED_PIN, LOW);  delay(800);
      }
      Serial.println("Server ACTIV din nou!");
    }
  }
}
