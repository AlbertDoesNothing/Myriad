#define LED_PIN 3
#define BUZZER_PIN 4

bool isActive = false;
unsigned long previousMillis = 0;
const int blinkInterval = 250;
bool state = false;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == '1') {
      isActive = true;
    } else if (cmd == '0') {
      isActive = false;
      digitalWrite(LED_PIN, LOW);
      digitalWrite(BUZZER_PIN, LOW);
    }
  }

  if (isActive) {
    unsigned long currentMillis = millis();
    if (currentMillis - previousMillis >= blinkInterval) {
      previousMillis = currentMillis;
      state = !state;
      digitalWrite(LED_PIN, state);
      digitalWrite(BUZZER_PIN, state);  // Buzzer toggles with LED
    }
  }
}
