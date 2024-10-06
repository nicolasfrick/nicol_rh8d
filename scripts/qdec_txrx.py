import serial

START_BYTE = 0xAA
END_BYTE = 0xBB
ESCAPE_BYTE = 0x7D

def uart_receive_data(ser):
    data_buffer = []
    escape = False

    while True:
        # Read a single byte from the serial port
        data = ser.read(1)

        if len(data) == 0:
            # Timeout or no data received
            continue

        # Convert byte to integer
        byte = ord(data)

        # Handle start byte
        if byte == START_BYTE:
            data_buffer = []  # Reset buffer for new data packet
            escape = False
            continue

        # Handle end byte
        if byte == END_BYTE:
            # End of packet, return the collected data
            if len(data_buffer) >= 6:  # We expect 3 uint16 numbers (6 bytes)
                num1 = (data_buffer[0] << 8) | data_buffer[1]
                num2 = (data_buffer[2] << 8) | data_buffer[3]
                num3 = (data_buffer[4] << 8) | data_buffer[5]
                return num1, num2, num3
            else:
                # Invalid packet length
                continue

        # Handle escape byte
        if byte == ESCAPE_BYTE:
            escape = True
            continue

        # If escape flag is set, XOR the byte with 0x20
        if escape:
            byte ^= 0x20
            escape = False

        # Add byte to data buffer
        data_buffer.append(byte)

# Usage Example
ser = serial.Serial('/dev/ttyUSB0', baudrate=9600, timeout=1)  # Replace with your UART port

try:
    num1, num2, num3 = uart_receive_data(ser)
    print(f"Received numbers: {num1}, {num2}, {num3}")
finally:
    ser.close()