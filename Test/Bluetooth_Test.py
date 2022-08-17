import serial

# NOTE: If the serial port connection gives a semaphore error, or if the HC 05 cannot be found in the device list, the fix is to
# disconnect the HC 05 from the circuit (can be done by removing the wires from the 5V and GND pins), re-upload the code to the arduino, 
# then re-connect it.

#print(serial.tools.list_ports())

serial_port = serial.Serial("COM9", baudrate=9600, timeout=1)
#print(dir(serial_port))
#serial_port.flushInput()
while True:
    state = input("1 or 0: ")
    if state == "q": break
    serial_port.write(bytes(state, 'utf-8'))
serial_port.close()