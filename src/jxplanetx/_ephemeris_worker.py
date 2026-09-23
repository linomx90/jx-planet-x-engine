"""Private isolated CSPICE worker for :mod:`jxplanetx.ephemeris`."""

from __future__ import annotations

import json
import math
import os
import sys


_RESPONSE_SCHEMA = "jx.ephemeris-worker-response.v1"


def _emit(value: dict[str, object], return_code: int) -> int:
    sys.stdout.write(
        json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    )
    return return_code


def _failure(category: str, message: str) -> int:
    bounded = " ".join(message.split())[:2048] or "CSPICE worker failed"
    return _emit(
        {"category": category, "message": bounded, "ok": False, "schema": _RESPONSE_SCHEMA},
        2,
    )


def _body_code(spice: object, value: str) -> int:
    code = spice.bods2c(value)
    if type(code) is tuple:
        number, found = code
        if found is not True:
            raise ValueError(f"NAIF body {value!r} is unknown")
        code = number
    if type(code) is not int:
        raise ValueError(f"NAIF body {value!r} did not resolve to an integer")
    return code


def main() -> int:
    try:
        import spiceypy as spice
        from spiceypy.utils.exceptions import SpiceyError
    except Exception as exc:
        return _failure("DEPENDENCY", f"spiceypy/CSPICE import failed: {exc}")

    try:
        body = json.load(sys.stdin)
    except Exception as exc:
        return _failure("REQUEST", f"worker request JSON is invalid: {exc}")
    if type(body) is not dict or body.get("schema") != "jx.ephemeris-worker-request.v1":
        return _failure("REQUEST", "worker request schema is invalid")
    descriptors = body.get("kernel_fds")
    request = body.get("request")
    if type(descriptors) is not list or not descriptors or type(request) is not dict:
        return _failure("REQUEST", "worker kernel descriptors or request are invalid")
    if any(type(value) is not int or value < 0 for value in descriptors):
        return _failure("REQUEST", "worker kernel descriptor is invalid")

    phase = "KERNEL"
    spice.kclear()
    try:
        for descriptor in descriptors:
            path = f"/proc/self/fd/{descriptor}"
            if not os.path.isfile(path):
                return _failure("KERNEL", "passed kernel descriptor is unavailable")
            spice.furnsh(path)

        epoch = request.get("epoch")
        if type(epoch) is not dict:
            return _failure("REQUEST", "worker epoch is invalid")
        phase = "TIME"
        if epoch.get("format") == "SECONDS_PAST_J2000":
            et = epoch.get("value")
            if type(et) is not float or not math.isfinite(et) or epoch.get("time_scale") != "TDB":
                return _failure("REQUEST", "native TDB epoch is invalid")
        elif epoch.get("format") == "CALENDAR":
            scale = {"UTC": "UTC", "TT": "TDT", "TDB": "TDB"}.get(epoch.get("time_scale"))
            value = epoch.get("value")
            if scale is None or type(value) is not str:
                return _failure("REQUEST", "calendar epoch is invalid")
            et = float(spice.str2et(f"{value} {scale}"))
        else:
            return _failure("REQUEST", "epoch format is unsupported")

        phase = "QUERY"
        target = request.get("target")
        observer = request.get("observer")
        frame = request.get("frame")
        correction = request.get("aberration_correction")
        if any(type(value) is not str for value in (target, observer, frame, correction)):
            return _failure("REQUEST", "state request fields are invalid")
        state, light_time = spice.spkezr(target, et, frame, correction, observer)
        target_id = _body_code(spice, target)
        observer_id = _body_code(spice, observer)
        values = [float(component) for component in state]
        light_time = float(light_time)
        if len(values) != 6 or any(not math.isfinite(value) for value in (*values, et, light_time)):
            return _failure("EXECUTION", "CSPICE returned a non-finite state")
        return _emit(
            {
                "ephemeris_time_seconds": float(et),
                "light_time_seconds": light_time,
                "observer_naif_id": observer_id,
                "ok": True,
                "schema": _RESPONSE_SCHEMA,
                "state": values,
                "target_naif_id": target_id,
            },
            0,
        )
    except SpiceyError as exc:
        text = str(exc)
        upper = text.upper()
        if "SPKINSUFFDATA" in upper or "NOSEGMENTSFOUND" in upper:
            category = "COVERAGE"
        elif phase == "KERNEL":
            category = "KERNEL"
        elif phase == "TIME":
            category = "TIME"
        else:
            category = "EXECUTION"
        return _failure(category, text)
    except Exception as exc:
        return _failure("EXECUTION" if phase == "QUERY" else phase, str(exc))
    finally:
        spice.kclear()


if __name__ == "__main__":
    raise SystemExit(main())
