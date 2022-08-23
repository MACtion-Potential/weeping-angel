#define leftPin 10
#define rightPin 9
char state = '3';
char old_state = '3';
int omega = 200;

void setup() {
  pinMode(leftPin, OUTPUT);
  pinMode(rightPin, OUTPUT);
  analogWrite(leftPin, 0);
  analogWrite(rightPin, 0);
  Serial.begin(9600); // Default communication rate of the Bluetooth module
}

void loop() {
  if(Serial.available() > 0){ // Checks whether data is comming from the serial port
    state = Serial.read(); // Reads the data from the serial port
 }
 if (state != old_state) {
   if (state == '0') {
    analogWrite(leftPin, 0); // Turn LED OFF
    analogWrite(rightPin, omega); // Send back, to the phone, the String "LED: ON"
    Serial.println("RIGHT");
   }
   else if (state == '1') {
    analogWrite(leftPin, omega);
    analogWrite(rightPin, 0);
    Serial.println("LEFT");
   } 
   else if (state == '2') {
    analogWrite(leftPin, omega);
    analogWrite(rightPin, omega);
    Serial.println("BOTH");
   }
   else if (state == '3') {
    analogWrite(leftPin, 0);
    analogWrite(rightPin, 0);
    Serial.println("NOTHING");
   }
   old_state = state;
 }
}
