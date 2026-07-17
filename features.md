**Features extrase per fereastră de 30s**



**MAX30102 — PPG / puls (7 features)**

bpm\_mean – Puls mediu în fereastră (bătăi/min)

bpm\_std – Variabilitatea BPM (cât fluctuează pulsul)

bpm\_min / bpm\_max – Limitele pulsului în fereastra de 30s

hrv\_simple – HRV (RMSSD în ms, variabilitate cardiacă)

contact\_valid\_ratio – Procent din timp când senzorul este în contact bun

spo2\_estimated – Saturația de oxigen estimată (RED/IR)



**MPU6050 — accelerometru + giroscop (5 features)**

motion\_level – Magnitudinea medie a accelerației nete (fără 1g)

motion\_per\_min – Număr de mișcări semnificative pe minut

motion\_variation – Variabilitatea mișcării (deviație standard)

dominant\_position – Poziția dominantă

(0 = lateral, 1 = față-spate, 2 = vertical)

gyro\_energy – Energia rotației (media pătratelor valorilor giroscopului)



**Flex sensor + ADS1015 — respirație (5 features)**

resp\_rate – Rata respiratorie (respirații/minut)

resp\_variability – Variabilitatea respirației (std intervale)

resp\_amplitude – Amplitudinea respirației (peak-to-peak)

apnea\_events\_simple – Pauze >10s între respirații (apnee)

breath\_valid\_ratio – Procent semnal respirator valid



**BMP280 — temperatură + presiune (2 features)**

temp\_mean – Temperatura medie a pielii în fereastră

temp\_variation – Variabilitatea temperaturii (std)



**MAX4466 — microfon (sforăit) (2 features)**

snore\_events\_count – Număr de episoade de sforăit (blocuri ≥0.5s cu energie RMS ridicată)

snore\_intensity – Intensitatea medie a sforăitului



**Total: 21 features / fereastră de 30 secunde**

