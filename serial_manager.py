import threading
import time

import serial
import serial.tools.list_ports


class SerialReader(threading.Thread):
    def __init__(self, ser: serial.Serial) -> None:
        super().__init__()
        self.ser = ser
        self.daemon = True
        self._stop = threading.Event()
        self.data_lock = threading.Lock()
        self.data = ""

    def run(self) -> None:
        while not self._stop.is_set():
            data = self.ser.readline().decode().replace("\r", "").replace("\n", "")
            if data:
                print("Read:", data)
                with self.data_lock:
                    self.data = data
            time.sleep(0.01)
        print("Serial reader stopped.")

    def stop(self) -> None:
        self._stop.set()


class SerialManager:
    def __init__(self, baudrate: int = 115200) -> None:
        self.port: str = ""
        self.found_device = self.detect_port()
        if self.found_device:
            self.baudrate = baudrate
            self.ser = serial.Serial(self.port, self.baudrate)
            self.serial_reader = SerialReader(self.ser)
            self.serial_reader.start()

    def detect_port(self) -> bool:
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if "Arduino Mega 2560" in p.description:
                print(f"Arduino Mega 2560 found on port: {p.device}")
                self.port = p.device
                return True
        print("WARNING: Arduino Mega 2560 device not found")
        return False

    def write(self, data: str) -> int:
        if not self.found_device:
            print("Write failed: device not found")
            return -1
        print("Write:", data)
        return self.ser.write(data.encode())

    def get_last_read_data(self) -> str:
        if not self.found_device:
            print("Read failed: device not found")
            return ""
        with self.serial_reader.data_lock:
            return self.serial_reader.data

    def close(self) -> None:
        if not self.found_device:
            print("Close failed: device not found")
            return
        self.serial_reader.stop()
        self.ser.close()

    def __del__(self) -> None:
        if self.found_device:
            self.close()
            print("Serial closed.")
