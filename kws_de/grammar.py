from dataclasses import dataclass

from kws_de import config


@dataclass(frozen=True)
class Intent:
    device: str
    zone: str | None
    action: str


@dataclass(frozen=True)
class Rejection:
    reason: str


def parse(events: list[str]) -> Intent | Rejection:
    toks = [e for e in events if e not in ("_unknown_", "_silence_")]
    device = zone = action = None
    for t in toks:
        if t in config.LIGHT_COMPOUNDS:
            # A fused Licht+zone compound ("Küchenlicht") sets device+zone in one
            # token -- see config.LIGHT_COMPOUNDS' docstring for why this is not
            # yet reachable from a real recognition (pending retrain), but the
            # grammar itself is ready and falls through to the same device/action
            # checks below as "Licht <zone>" would.
            if device is not None:
                return Rejection(f"duplicate device: {t}")
            if zone is not None:
                return Rejection(f"duplicate zone: {t}")
            if action is not None:
                return Rejection("device out of order")
            device, zone = "Licht", config.LIGHT_COMPOUNDS[t]
        elif t in config.DEVICES:
            if device is not None:
                return Rejection(f"duplicate device: {t}")
            if zone is not None or action is not None:
                return Rejection("device out of order")
            device = t
        elif t in config.ZONES:
            if zone is not None:
                return Rejection(f"duplicate zone: {t}")
            if device is None or action is not None:
                return Rejection("zone out of order")
            zone = t
        elif t in config.ACTIONS:
            if action is not None:
                return Rejection(f"duplicate action: {t}")
            action = t
        else:
            return Rejection(f"unknown token: {t}")
    if device is None:
        return Rejection("missing device")
    if action is None:
        return Rejection("missing action")
    if zone is not None and device not in config.ZONED_DEVICES:
        return Rejection(f"{device} takes no zone")
    if action not in config.DEVICE_ACTIONS[device]:
        return Rejection(f"{action} invalid for {device}")
    return Intent(device, zone, action)
