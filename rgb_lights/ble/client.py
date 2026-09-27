from __future__ import annotations

import asyncio
import logging
import time
from typing import Sequence

from bleak import BleakClient
from bleak.exc import BleakError

from rgb_lights.ble.protocols import DEVICE_ADDRESS, DaybetterProtocol, WRITE_CHARACTERISTIC

logger = logging.getLogger(__name__)

# Cheap BLE strips + Windows WinRT often hang forever on write-with-response.
WRITE_TIMEOUT_S = 1.25
MIN_WRITE_GAP_S = 0.045  # ~22 Hz hard ceiling; lower effective rate is fine
POWER_KEEPALIVE_S = 18.0


def _is_writable(properties: Sequence[str]) -> bool:
    return "write" in properties or "write-without-response" in properties


class BleLightController:
    """Bleak controller for Daybetter / P031 BLE LED strips."""

    def __init__(
        self,
        address: str | None = None,
        characteristic: str | None = None,
        write_without_response: bool | None = None,
    ) -> None:
        self.address = address or DEVICE_ADDRESS
        self.protocol = DaybetterProtocol()
        self.characteristic = characteristic or WRITE_CHARACTERISTIC
        self.write_without_response = write_without_response
        self._client: BleakClient | None = None
        self._write_lock = asyncio.Lock()
        self._last_rgb: tuple[int, int, int] | None = None
        self._last_write_at = 0.0
        self._last_keepalive_at = 0.0
        self._disconnect_flag = False
        self._reconnect_lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        if self._disconnect_flag:
            return False
        return bool(self._client and self._client.is_connected)

    def _on_disconnect(self, _client: BleakClient) -> None:
        # Intentional disconnect already set the flag — don't warn.
        if self._disconnect_flag or self._client is None:
            self._disconnect_flag = True
            return
        self._disconnect_flag = True
        logger.warning("BLE disconnected from %s", self.address)

    async def connect(self, timeout: float = 8.0) -> None:
        if self.is_connected:
            return
        self._disconnect_flag = False

        # Windows often needs a fresh advertisement before connect-by-address works.
        # Keep the pre-scan short — known address usually appears within a couple seconds.
        device = await self._find_device(timeout=min(3.0, timeout))
        last_err: Exception | None = None
        for attempt in range(1, 3):
            try:
                client = BleakClient(
                    device if device is not None else self.address,
                    timeout=timeout,
                    disconnected_callback=self._on_disconnect,
                )
                await client.connect()
                self._client = client
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning("BLE connect attempt %s failed: %s", attempt, exc)
                if attempt < 2:
                    device = await self._find_device(timeout=2.5)
                    await asyncio.sleep(0.2)
        else:
            hint = (
                f"Could not reach LED strip {self.address}.\n"
                "  1. Close other rgb_lights windows (only one BLE client works)\n"
                "  2. Force-close the phone LED app\n"
                "  3. Power-cycle the strip (unplug / flip switch)\n"
                "  4. Run: python -m rgb_lights scan"
            )
            raise BleakError(hint) from last_err

        self.characteristic = self._resolve_characteristic(self.characteristic)
        # P031 GATT often advertises only "write", but with-response hangs on WinRT.
        # Force fire-and-forget unless the caller explicitly set a preference.
        if self.write_without_response is None:
            self.write_without_response = True
        if not self.write_without_response:
            logger.warning(
                "Using write-with-response (often times out on this hardware)"
            )
        await self._write(self.protocol.power_on(), bypass_gap=True)
        self._last_keepalive_at = time.perf_counter()
        logger.info(
            "Connected to %s (char %s, wwr=%s)",
            self.address,
            self.characteristic,
            self.write_without_response,
        )

    async def _find_device(self, timeout: float = 3.0):
        """Scan so WinRT has a live advertisement for this address."""
        from bleak import BleakScanner

        target = self.address.upper()
        try:
            device = await BleakScanner.find_device_by_address(
                self.address, timeout=timeout
            )
            if device is not None:
                return device
        except Exception as exc:  # noqa: BLE001
            logger.debug("Address scan failed: %s", exc)

        try:
            # Unfiltered scan — OS-level service UUID filters miss P031 ads (c031).
            found = await BleakScanner.discover(
                timeout=max(1.5, timeout * 0.6),
                return_adv=True,
            )
            items = found.values() if isinstance(found, dict) else found
            for item in items:
                dev = item[0] if isinstance(item, tuple) else item
                if str(getattr(dev, "address", "")).upper() == target:
                    return dev
        except Exception as exc:  # noqa: BLE001
            logger.debug("Fallback scan failed: %s", exc)
        return None

    def _resolve_characteristic(self, preferred: str) -> str:
        assert self._client is not None
        services = list(self._client.services)
        preferred_l = preferred.lower()
        known_services = {u.lower() for u in self.protocol.alt_service_uuids}

        for service in services:
            for char in service.characteristics:
                if char.uuid.lower() == preferred_l and _is_writable(char.properties):
                    return char.uuid

        for service in services:
            if service.uuid.lower() not in known_services:
                continue
            writables = [c for c in service.characteristics if _is_writable(c.properties)]
            if writables:
                # Prefer WWR-capable characteristics first
                wwr = [c for c in writables if "write-without-response" in c.properties]
                chosen = wwr[0] if wwr else writables[0]
                logger.info("Auto-selected characteristic %s", chosen.uuid)
                return chosen.uuid

        for service in services:
            for char in service.characteristics:
                if _is_writable(char.properties):
                    logger.warning("Known LED service missing; falling back to %s", char.uuid)
                    return char.uuid
        return preferred

    async def disconnect(self) -> None:
        client = self._client
        self._client = None
        # Mark intentional disconnect so the callback doesn't look like a drop.
        self._disconnect_flag = True
        if client:
            try:
                await asyncio.wait_for(client.disconnect(), timeout=1.5)
            except Exception:  # noqa: BLE001
                pass

    async def reconnect(self, timeout: float = 8.0) -> None:
        """Drop and re-open the BLE link."""
        async with self._reconnect_lock:
            logger.info("Reconnecting BLE to %s...", self.address)
            try:
                await self.disconnect()
            except Exception:  # noqa: BLE001
                self._client = None
            await asyncio.sleep(0.25)
            self._last_rgb = None
            await self.connect(timeout=timeout)

    async def ensure_connected(self) -> bool:
        if self.is_connected:
            return True
        try:
            await self.reconnect()
            return self.is_connected
        except Exception as exc:  # noqa: BLE001
            logger.warning("BLE reconnect failed: %s", exc)
            return False

    async def __aenter__(self) -> "BleLightController":
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.disconnect()

    async def power_on(self) -> None:
        await self._write(self.protocol.power_on(), bypass_gap=True)
        self._last_keepalive_at = time.perf_counter()

    async def power_off(self) -> None:
        await self._write(self.protocol.power_off(), bypass_gap=True)
        self._last_rgb = (0, 0, 0)

    async def set_rgb(self, r: int, g: int, b: int, *, force: bool = False) -> None:
        color = (int(r) & 0xFF, int(g) & 0xFF, int(b) & 0xFF)
        if not force and color == self._last_rgb:
            await self.maybe_keepalive()
            return
        await self._write(self.protocol.set_rgb(*color))
        self._last_rgb = color
        await self.maybe_keepalive()

    async def set_brightness(self, level: int) -> None:
        await self._write(self.protocol.set_brightness(level), bypass_gap=True)

    async def maybe_keepalive(self) -> None:
        """Some Daybetter firmware drops the show unless periodically poked."""
        now = time.perf_counter()
        if (now - self._last_keepalive_at) < POWER_KEEPALIVE_S:
            return
        try:
            await self._write(self.protocol.power_on(), bypass_gap=True)
            self._last_keepalive_at = now
        except Exception as exc:  # noqa: BLE001
            logger.debug("Keepalive failed: %s", exc)

    async def _write(self, data: Sequence[int] | bytes, *, bypass_gap: bool = False) -> None:
        if not self._client or not self.is_connected:
            raise BleakError("Not connected to light")

        now = time.perf_counter()
        if not bypass_gap:
            gap = now - self._last_write_at
            if gap < MIN_WRITE_GAP_S:
                await asyncio.sleep(MIN_WRITE_GAP_S - gap)

        payload = bytes(data)
        use_wwr = bool(self.write_without_response)
        async with self._write_lock:
            if not self._client or not self.is_connected:
                raise BleakError("Not connected to light")
            try:
                await asyncio.wait_for(
                    self._client.write_gatt_char(
                        self.characteristic,
                        payload,
                        response=not use_wwr,
                    ),
                    timeout=WRITE_TIMEOUT_S,
                )
            except asyncio.TimeoutError as exc:
                self._disconnect_flag = True
                raise BleakError(f"BLE write timed out after {WRITE_TIMEOUT_S:.1f}s") from exc
            except Exception:
                self._disconnect_flag = True
                raise
            finally:
                self._last_write_at = time.perf_counter()

    def _prefer_write_without_response(self) -> bool:
        assert self._client is not None
        for service in self._client.services:
            for char in service.characteristics:
                if char.uuid.lower() == self.characteristic.lower():
                    return "write-without-response" in char.properties
        return False
