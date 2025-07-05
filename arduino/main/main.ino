#define LED_PIN 3
#define BUZZER_PIN 4
#define LED_PIN_GREEN 2

bool isActive = false;

// === LED flashing (every 100ms) ===
const unsigned long ledInterval = 100;
unsigned long previousLedMillis = 0;
bool ledState = false;

// === Buzzer tones (beep-boo every 300ms + 200ms pause) ===
const unsigned long buzzerInterval = 300;
const unsigned long buzzerSilence = 200;
unsigned long previousBuzzerMillis = 0;

enum ToneState { BEEP, PAUSE1, BOO, PAUSE2 };
ToneState toneState = PAUSE2;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(LED_PIN_GREEN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  Serial.begin(9600);

  // Turn green LED on at startup
  digitalWrite(LED_PIN_GREEN, HIGH);
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == '1') {
      isActive = true;
    } else if (cmd == '0') {
      isActive = false;
      digitalWrite(LED_PIN, LOW);
      noTone(BUZZER_PIN);
    }
  }

  unsigned long currentMillis = millis();

  if (isActive) {
    // Turn off green LED during warning
    digitalWrite(LED_PIN_GREEN, LOW);

    // Blink main LED every 100ms
    if (currentMillis - previousLedMillis >= ledInterval) {
      previousLedMillis = currentMillis;
      ledState = !ledState;
      digitalWrite(LED_PIN, ledState);
    }

    // Handle buzzer tone switching
    if (currentMillis - previousBuzzerMillis >= buzzerInterval) {
      previousBuzzerMillis = currentMillis;

      switch (toneState) {
        case PAUSE1:
        case PAUSE2:
          noTone(BUZZER_PIN);
          toneState = (toneState == PAUSE1) ? BOO : BEEP;
          previousBuzzerMillis += buzzerSilence;
          break;

        case BEEP:
          tone(BUZZER_PIN, 1000);
          toneState = PAUSE1;
          break;

        case BOO:
          tone(BUZZER_PIN, 600);
          toneState = PAUSE2;
          break;
      }
    }
  } else {
    // When idle, keep green LED on
    digitalWrite(LED_PIN_GREEN, HIGH);
  }
}
