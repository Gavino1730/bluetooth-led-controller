from __future__ import annotations

from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

from rgb_lights.ble.protocols import ALT_SERVICE_UUIDS

KNOWN_SERVICES = tuple(u.lower() for u in ALT_SERVICE_UUIDS)

NAME_HINTS = (
    "daybetter",
    "day better",
    "gupup",
    "allbest",
    "p031",
)


async def discover_devices(
    timeout: float = 5.0,
    *,
    daybetter_only: bool = True,
) -> list[tuple[BLEDevice, AdvertisementData]]:
    # Always scan unfiltered, then match in software. WinRT service_uuid filters
    # miss P031 when the adv UUID (c031) differs from the GATT UUID (e031).
    found = list((await BleakScanner.discover(timeout=timeout, return_adv=True)).values())
    if not daybetter_only:
        return found
    return [(device, adv) for device, adv in found if is_daybetter(device, adv)]


def is_daybetter(device: BLEDevice, adv: AdvertisementData) -> bool:
    for uuid in adv.service_uuids:
        if str(uuid).lower() in KNOWN_SERVICES:
            return True
    name = (device.name or adv.local_name or "").lower()
    return any(hint in name for hint in NAME_HINTS)


def format_device(device: BLEDevice, adv: AdvertisementData) -> str:
    name = device.name or adv.local_name or "(no name)"
    rssi = adv.rssi if adv.rssi is not None else "?"
    tag = " [daybetter/p031]" if is_daybetter(device, adv) else ""
    return f"{device.address}  rssi={rssi:>4}  {name}{tag}"


async def print_scan(timeout: float = 5.0, all_devices: bool = False) -> int:
    daybetter_only = not all_devices
    label = "Daybetter/P031" if daybetter_only else "all BLE"
    print(f"Scanning for {label} devices ({timeout:.0f}s)...")
    found = await discover_devices(timeout=timeout, daybetter_only=daybetter_only)
    if not found:
        print(
            "No devices found. Close the phone app, power-cycle the strip, "
            "then retry. For a wider scan: python -m rgb_lights scan --all"
        )
        return 1

    rows = sorted(
        found,
        key=lambda pair: pair[1].rssi if pair[1].rssi is not None else -999,
        reverse=True,
    )
    for device, adv in rows:
        marker = " *" if is_daybetter(device, adv) else "  "
        print(f"{marker}{format_device(device, adv)}")

    print()
    print("* = Daybetter/P031 service/name match")
    print("Copy an address into config.toml -> [device].address")
    return 0


async def explore_services(address: str) -> int:
    from bleak import BleakClient

    print(f"Connecting to {address}...")
    async with BleakClient(address) as client:
        print(f"Connected: {client.is_connected}")
        for service in client.services:
            print(f"\nService {service.uuid}  {service.description}")
            for char in service.characteristics:
                props = ", ".join(char.properties)
                print(f"  Char {char.uuid}  [{props}]  {char.description}")
    return 0
