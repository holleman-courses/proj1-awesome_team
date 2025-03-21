

import serial
from PIL import Image
import numpy as np
import datetime


port = 4
baud = 115200

WIDTH = 176
HEIGHT = 144

ser = serial.Serial(f'COM{port}', baud)

def wait_for_marker(marker):
    buffer = ""
    while marker not in buffer:
        buffer += ser.read(1).decode(errors="ignore")
    print(f"Found marker: {marker}")

def receive_image():
    """ Receives image data over serial and returns a numpy array """
    print("Waiting for image data...")
    
    image_data = bytearray()
    expected_size = WIDTH * HEIGHT # 2 bytes per pixel (RGB565)
    wait_for_marker("START")
    while len(image_data) < expected_size:
        chunk = ser.read(expected_size - len(image_data))  # Read remaining bytes
        if not chunk:
            print("Timeout or incomplete data received!")
            return None
        image_data.extend(chunk)
    wait_for_marker("END")
    if len(image_data) != expected_size:
        print(f"Expected {expected_size} bytes, but got {len(image_data)}")

    data = np.frombuffer(image_data, dtype=np.uint8)
    grayscale = data.reshape((HEIGHT, WIDTH))
    img = Image.fromarray(grayscale, "L")
    return img
    rgb565 = np.bitwise_or((rgb565 >> 8) & 0xFF, (rgb565 << 8) & 0xFF00)  # Swap bytes

    return rgb565#np.frombuffer(image_data, dtype=np.uint16).reshape((HEIGHT, WIDTH))

def rgb565_to_grayscale(rgb565):
    """ Converts RGB565 to grayscale using the standard luminosity formula. """
    r = ((rgb565 >> 11) & 0x1F) * 255 // 31
    g = ((rgb565 >> 5) & 0x3F) * 255 // 63
    b = (rgb565 & 0x1F) * 255 // 31
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)



counter = 413

while True:
    frame = receive_image()
    if frame is not None:
        # grayscale = rgb565_to_grayscale(frame)a

        # Save as PNG
        # img = Image.fromarray(grayscale, "L")
        name = f"no_pencil_{counter}"
        counter += 1
        frame.save(f"data_collection/no_pencil/{name}.png")
        print(f"Image saved as {name}.png")
        # break  # Stop after one image (remove to keep running)