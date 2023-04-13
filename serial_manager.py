import serial
import serial.tools.list_ports
import threading


class SerialReader(threading.Thread):
    def __init__(self, serial_manager: "SerialManager") -> None:
        super().__init__()
        self.serial_manager = serial_manager
        self.daemon = True
        self.stop = threading.Event()
        self.start()

    def run(self) -> None:
        while not self.stop.is_set():
            data = self.serial_manager.read()
            if data:
                print("Read:", data)
        print("Serial reader stopped.")


class SerialManager:
    def __init__(self, baudrate: int = 115200) -> None:
        self.port = self.detect_port()
        self.baudrate = baudrate
        self.ser = serial.Serial(self.port, self.baudrate)
        self.serial_reader = SerialReader(self)

    @staticmethod
    def detect_port() -> str:
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if "Arduino Mega 2560" in p.description:
                print(f"Arduino Mega 2560 found on port: {p.device}")
                return p.device
        raise IOError("Arduino Mega 2560 device not found")

    def write(self, data: str) -> int:
        print("Write:", data)
        return self.ser.write(data.encode())

    def read(self) -> str:
        return self.ser.readline().decode("utf-8").replace("\r\n", "")

    def close(self) -> None:
        self.serial_reader.stop.set()
        self.ser.close()

    def __del__(self):
        self.close()
        print("Serial closed.")
