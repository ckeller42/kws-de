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
        if t in config.DEVICES:
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
    return Intent(device, zone, action)
