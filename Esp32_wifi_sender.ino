/*
  Esp32_wifi_sender.ino
  =====================
  Trimite datele senzorilor la Raspberry Pi prin WiFi (HTTP POST).
  PPG (IR+RED) trimis la 50Hz in blocuri de 50 samples catre /ppg
  Microfon trimis la 200Hz in blocuri de 200 samples catre /mic
  Date senzori (IMU, flex, temp) trimise la 1Hz catre /data

  Inainte de a folosi:
  1. Seteaza SSID si parola WiFi
  2. Seteaza IP-ul Raspberry Pi (SERVER_IP)
  3. Porneste server.py pe Raspberry, APOI alimenteaza ESP32
*/

#include <WiFi.h>
#include <Wire.h>
#include <HTTPClient.h>
#include "time.h"

#include <Adafruit_ADS1X15.h>
#include <Adafruit_BMP280.h>
#include "MAX30105.h"

//////////////// WIFI //////////////////
const char* ssid     = "TP-Link_EA9C";
const char* password = "97205401";

//////////////// SERVER RASPBERRY //////
const char* SERVER_IP   = "192.168.1.179";
const int   SERVER_PORT = 5000;

String URL_DATA   = String("http://") + SERVER_IP + ":" + SERVER_PORT + "/data";
String URL_MIC    = String("http://") + SERVER_IP + ":" + SERVER_PORT + "/mic";
String URL_PPG    = String("http://") + SERVER_IP + ":" + SERVER_PORT + "/ppg";
String URL_STATUS = String("http://") + SERVER_IP + ":" + SERVER_PORT + "/status";

//////////////// NTP ///////////////////
const char* ntpServer          = "pool.ntp.org";
const long  gmtOffset_sec      = 7200;
const int   daylightOffset_sec = 3600;

//////////////// PINI //////////////////
#define SDA_PIN 5
#define SCL_PIN 6
#define MIC_PIN 0
#define LED_PIN 8

//////////////// ADRESE I2C ////////////
#define IMU_ADDR 0x68
#define BMP_ADDR 0x76

//////////////// SENZORI ///////////////
Adafruit_ADS1015 ads;
Adafruit_BMP280  bmp;
MAX30105         particleSensor;

//////////////// STATUS ////////////////
bool adsOK        = false;
bool bmpOK        = false;
bool imuOK        = false;
bool maxOK        = false;
bool timeOK       = false;
bool serverActive = false;

//////////////// FLEX //////////////////
int flexBaseline = 0;
int flexRaw      = 0;
int flexDelta    = 0;

//////////////// MICROFON //////////////
const int MIC_BLOCK_SIZE = 200;
int  micBuffer[MIC_BLOCK_SIZE];
int  micIndex      = 0;
bool micBlockReady = false;

//////////////// PPG HIGH FREQ /////////
const int PPG_BLOCK_SIZE = 50;        // 50 samples = 1 secunda la 50Hz
uint32_t ppgIrBuffer[PPG_BLOCK_SIZE];
uint32_t ppgRedBuffer[PPG_BLOCK_SIZE];
int  ppgIndex      = 0;
bool ppgBlockReady = false;

//////////////// IMU ///////////////////
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

//////////////// BMP280 ////////////////
float temperatureC = NAN;
float pressurehPa  = NAN;

//////////////// MAX30102 //////////////
uint32_t redValue = 0;
uint32_t irValue  = 0;

//////////////// TIMERE ////////////////
unsigned long lastPulseRead   = 0;
unsigned long lastFlexRead    = 0;
unsigned long lastImuRead     = 0;
unsigned long lastBmpRead     = 0;
unsigned long lastMicRead     = 0;
unsigned long lastDataSend    = 0;
unsigned long lastStatusCheck = 0;

const unsigned long PULSE_INTERVAL_MS  = 20;    // 50 Hz
const unsigned long FLEX_INTERVAL_MS   = 200;   // 5 Hz
const unsigned long IMU_INTERVAL_MS    = 100;   // 10 Hz
const unsigned long BMP_INTERVAL_MS    = 1000;  // 1 Hz
const unsigned long MIC_INTERVAL_MS    = 5;     // 200 Hz
const unsigned long DATA_SEND_MS       = 1000;  // 1 Hz
const unsigned long STATUS_CHECK_MS    = 10000; // 10 s

// ─────────────────────────────────────────────
//  TIME
// ─────────────────────────────────────────────
bool getTimeString(char* buf, size_t len) {
  struct tm ti;
  if (!getLocalTime(&ti)) return false;
  strftime(buf, len, "%H:%M:%S", &ti);
  return true;
}

// ─────────────────────────────────────────────
//  WIFI + NTP
// ─────────────────────────────────────────────
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

// ─────────────────────────────────────────────
//  VERIFICA SERVER
// ─────────────────────────────────────────────
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

// ─────────────────────────────────────────────
//  TRIMITE DATE SENZORI (1Hz)
// ─────────────────────────────────────────────
bool sendData() {
  if (WiFi.status() != WL_CONNECTED) return false;

  char timeBuf[16] = "00:00:00";
  getTimeString(timeBuf, sizeof(timeBuf));

  String json = "{";
  json += "\"t\":\"" + String(timeBuf)         + "\",";
  json += "\"fr\":"  + String(flexRaw)          + ",";
  json += "\"fd\":"  + String(flexDelta)         + ",";
  json += "\"red\":" + String(redValue)          + ",";
  json += "\"ir\":"  + String(irValue)           + ",";
  json += "\"ax\":"  + String(accX, 4)           + ",";
  json += "\"ay\":"  + String(accY, 4)           + ",";
  json += "\"az\":"  + String(accZ, 4)           + ",";
  json += "\"gx\":"  + String(gyroX, 4)          + ",";
  json += "\"gy\":"  + String(gyroY, 4)          + ",";
  json += "\"gz\":"  + String(gyroZ, 4)          + ",";
  json += "\"tmp\":" + String(temperatureC, 2)   + ",";
  json += "\"prs\":" + String(pressurehPa, 2);
  json += "}";

  HTTPClient http;
  http.begin(URL_DATA);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(3000);
  int code = http.POST(json);
  http.end();
  return code == 200;
}

// ─────────────────────────────────────────────
//  TRIMITE BLOC MICROFON
// ─────────────────────────────────────────────
bool sendMic() {
  if (WiFi.status() != WL_CONNECTED) return false;

  char timeBuf[16] = "00:00:00";
  getTimeString(timeBuf, sizeof(timeBuf));

  String samples = "";
  for (int i = 0; i < MIC_BLOCK_SIZE; i++) {
    samples += String(micBuffer[i]);
    if (i < MIC_BLOCK_SIZE - 1) samples += ",";
  }

  String json = "{";
  json += "\"t\":\"" + String(timeBuf) + "\",";
  json += "\"s\":\"" + samples + "\"";
  json += "}";

  HTTPClient http;
  http.begin(URL_MIC);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(5000);
  int code = http.POST(json);
  http.end();
  return code == 200;
}

// ─────────────────────────────────────────────
//  TRIMITE BLOC PPG (50Hz)
// ─────────────────────────────────────────────
// WiFiClient global reutilizat pentru toate POST-urile (eficient pe LAN)
WiFiClient ppgWifiClient;

bool sendPPG() {
  if (WiFi.status() != WL_CONNECTED) return false;

  char timeBuf[16] = "00:00:00";
  getTimeString(timeBuf, sizeof(timeBuf));

  // Construire JSON cu reserve() pentru a evita fragmentarea heap
  String json;
  json.reserve(1200);

  json = "{\"t\":\"";
  json += timeBuf;
  json += "\",\"ir\":\"";
  for (int i = 0; i < PPG_BLOCK_SIZE; i++) {
    json += ppgIrBuffer[i];
    if (i < PPG_BLOCK_SIZE - 1) json += ',';
  }
  json += "\",\"red\":\"";
  for (int i = 0; i < PPG_BLOCK_SIZE; i++) {
    json += ppgRedBuffer[i];
    if (i < PPG_BLOCK_SIZE - 1) json += ',';
  }
  json += "\"}";

  // HTTPClient creat fresh la fiecare apel — mai stabil decat reuse
  // (overhead-ul TCP handshake e ~5ms pe LAN, neglijabil)
  HTTPClient http;
  http.begin(ppgWifiClient, URL_PPG);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(1500);
  int code = http.POST(json);
  http.end();

  if (code != 200) {
    Serial.print("[PPG] HTTP fail code=");
    Serial.println(code);
    return false;
  }
  return true;
}

// ─────────────────────────────────────────────
//  ADS1015 + FLEX
// ─────────────────────────────────────────────
bool initADS() {
  if (!ads.begin()) return false;
  ads.setGain(GAIN_EIGHT);  // ±0.512V – sensibilitate ridicata pentru flex
  return true;
}

void calibrateFlex() {
  ads.setGain(GAIN_EIGHT);  // asigura gain corect la calibrare
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
  if (flexDelta < 0) flexDelta = 0;
}

// ─────────────────────────────────────────────
//  MICROFON
// ─────────────────────────────────────────────
void readMicSample() {
  micBuffer[micIndex++] = analogRead(MIC_PIN);
  if (micIndex >= MIC_BLOCK_SIZE) {
    micIndex      = 0;
    micBlockReady = true;
  }
}

// ─────────────────────────────────────────────
//  IMU (MPU6050)
// ─────────────────────────────────────────────
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

// ─────────────────────────────────────────────
//  BMP280
// ─────────────────────────────────────────────
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

// ─────────────────────────────────────────────
//  MAX30102
// ─────────────────────────────────────────────
bool initMAX30102() {
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) return false;

  // Setari care AU FUNCTIONAT (semnal vizibil primele secunde):
  // sampleAverage=1, sampleRate=100, adcRange=4096, LED=0x3F
  // adcRange=4096 are rezolutie mai fina decat 16384
  particleSensor.setup(60, 1, 2, 100, 411, 4096);
  particleSensor.setPulseAmplitudeRed(0x3F);
  particleSensor.setPulseAmplitudeIR(0x3F);
  particleSensor.setPulseAmplitudeGreen(0);
  delay(100);
  return true;
}

void readPulseRaw() {
  // safeCheck() proceseaza FIFO-ul si actualizeaza valorile interne
  // Returneaza true daca exista sample nou
  particleSensor.check();  // citeste FIFO-ul daca e plin
  redValue = particleSensor.getRed();
  irValue  = particleSensor.getIR();
}

// ─────────────────────────────────────────────
//  SETUP
// ─────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1500);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);  // I2C Fast Mode 400kHz - toti senzorii suporta (MPU6050, BMP280, MAX30102, ADS1015)
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

// ─────────────────────────────────────────────
//  LOOP
// ─────────────────────────────────────────────
void loop() {
  unsigned long now = millis();

  // MAX30102 la 50 Hz – citeste SI colecteaza buffer PPG
  if (maxOK && now - lastPulseRead >= PULSE_INTERVAL_MS) {
    lastPulseRead = now;
    readPulseRaw();

    // Salveaza in buffer PPG pentru trimitere la 50Hz
    ppgIrBuffer[ppgIndex]  = irValue;
    ppgRedBuffer[ppgIndex] = redValue;
    ppgIndex++;
    if (ppgIndex >= PPG_BLOCK_SIZE) {
      ppgIndex      = 0;
      ppgBlockReady = true;
      // DEBUG: cate blocuri am umplut
      static int blockCount = 0;
      blockCount++;
      if (blockCount % 5 == 0) {
        Serial.print("[PPG] Buffer umplut #"); Serial.print(blockCount);
        Serial.print(" @ "); Serial.println(now);
      }
    }
  }

  // Flex la 5 Hz
  if (adsOK && now - lastFlexRead >= FLEX_INTERVAL_MS) {
    lastFlexRead = now;
    readFlex();
  }

  // IMU la 10 Hz
  if (imuOK && now - lastImuRead >= IMU_INTERVAL_MS) {
    lastImuRead = now;
    readIMU();
  }

  // BMP280 la 1 Hz
  if (bmpOK && now - lastBmpRead >= BMP_INTERVAL_MS) {
    lastBmpRead = now;
    readBMP();
  }

  // Microfon la 200 Hz
  if (now - lastMicRead >= MIC_INTERVAL_MS) {
    lastMicRead = now;
    readMicSample();
  }

  // ─── TRIMITERE PPG ─ INDEPENDENT, IMEDIAT CE BUFFER E GATA (1Hz) ─────
  // Mutat aici (separat de sendData/sendMic) pentru a NU astepta
  // celelalte trimiteri => 50Hz citire continua, fara pierderi de blocuri
  if (serverActive && ppgBlockReady) {
    ppgBlockReady = false;
    unsigned long t_start = millis();
    bool ppg_ok = sendPPG();
    unsigned long t_dur = millis() - t_start;
    static int sendCount = 0, failCount = 0;
    sendCount++;
    if (!ppg_ok) failCount++;
    if (sendCount % 5 == 0) {
      Serial.print("[PPG] Trimise: "); Serial.print(sendCount);
      Serial.print(" | Esuate: "); Serial.print(failCount);
      Serial.print(" | Ultima durata: "); Serial.print(t_dur);
      Serial.println("ms");
    }
  }

  // ─── TRIMITERE DATE 1Hz (raw senzori) ────────────────────────────────
  if (now - lastDataSend >= DATA_SEND_MS) {
    lastDataSend = now;

    if (serverActive) {
      bool ok = sendData();

      if (ok) {
        digitalWrite(LED_PIN, HIGH); delay(5);
        digitalWrite(LED_PIN, LOW);
      } else {
        for (int i = 0; i < 3; i++) {
          digitalWrite(LED_PIN, HIGH); delay(50);
          digitalWrite(LED_PIN, LOW);  delay(50);
        }
        Serial.println("WARN: trimitere date esuata");
      }

      // Trimite bloc microfon daca e gata (mai rar, ~1 bloc/secunda la 200Hz)
      if (micBlockReady) {
        micBlockReady = false;
        sendMic();
      }
    }
  }

  // Verifica periodic daca serverul e inca activ
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
