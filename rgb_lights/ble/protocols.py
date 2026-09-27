from __future__ import annotations

# Your P031 Daybetter strip (hardcoded — no config needed for BLE identity)
DEVICE_ADDRESS = "80:AC:C8:90:CB:44"
DEVICE_NAME = "P031_CB44"
WRITE_CHARACTERISTIC = "0000a031-0000-1000-8000-00805f9b34fb"
SERVICE_UUID = "0000e031-0000-1000-8000-00805f9b34fb"
# Advertisements use c031; GATT after connect exposes e031. WinRT service filters
# only match what is in the adv packet — include both.
ALT_SERVICE_UUIDS = (
    "0000c031-0000-1000-8000-00805f9b34fb",
    "0000e031-0000-1000-8000-00805f9b34fb",
    "0000e026-0000-1000-8000-00805f9b34fb",
)


class DaybetterProtocol:
    """Daybetter / P031 BLE LED strips using A0 packets with CRC-16/MODBUS."""

    name = "daybetter"
    default_characteristic = WRITE_CHARACTERISTIC
    service_uuid = SERVICE_UUID
    alt_service_uuids = ALT_SERVICE_UUIDS
    keepalive_interval_s: float | None = None

    def power_on(self) -> bytes:
        return self._frame(0x11, bytes([0x01]))

    def power_off(self) -> bytes:
        return self._frame(0x11, bytes([0x00]))

    def set_rgb(self, r: int, g: int, b: int) -> bytes:
        return self._frame(0x15, bytes([_clamp_byte(r), _clamp_byte(g), _clamp_byte(b)]))

    def set_brightness(self, level: int) -> bytes:
        level = max(0, min(100, int(level)))
        return self._frame(0x13, bytes([level]))

    def keepalive(self) -> bytes | None:
        return None

    def _frame(self, command: int, data: bytes) -> bytes:
        length = len(data) + 3
        body = bytes([0xA0, command & 0xFF, length & 0xFF]) + data
        crc = _crc16_modbus(body)
        return body + crc.to_bytes(2, "little")


def _clamp_byte(value: int) -> int:
    return max(0, min(255, int(value)))


def _crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc & 0xFFFF
