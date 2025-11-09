#!/usr/bin/env python3
import os, glob, time, math, threading, queue
from typing import Optional, Tuple, Generator, Dict
import numpy as np
import heartpy as hp
# If HeartPy warnings need to be caught explicitly:
from heartpy.exceptions import BadSignalWarning

# ---- MAX30102 driver (keep your path) ----
from Dependencies.max30102 import MAX30102

# ---------- HR Tunables ----------
SAMPLE_HZ   = 100          # match your MAX30102 config
WINDOW_SEC  = 8            # analysis window for HeartPy
STEP_SEC    = 1            # how often to push live updates
EMA_ALPHA   = 0.25         # smoothing for live BPM
USE_CHANNEL = 'ir'         # 'ir' or 'red'
MAX_RESETS  = 3            # I2C recovery attempts
# ----------------------------------

# ---------- Temperature (DS18B20) ----------
BASE_DIR = "/sys/bus/w1/devices/"

def _get_device_file() -> str:
    devices = glob.glob(os.path.join(BASE_DIR, "28-*"))
    if not devices:
        raise FileNotFoundError(
            "No DS18B20 found under /sys/bus/w1/devices/. "
            "Enable 1-Wire and check wiring (GPIO4 / pin 7, 4.7k pull-up to 3.3V)."
        )
    return os.path.join(devices[0], "w1_slave")

def _read_raw_lines(devfile: str) -> list[str]:
    with open(devfile, "r") as f:
        return f.read().strip().splitlines()

def read_temp_c_float(max_retries: int = 3, retry_delay: float = 0.2) -> float:
    """Return temperature in °C as float (e.g., 23.125)."""
    devfile = _get_device_file()
    for _ in range(max_retries):
        lines = _read_raw_lines(devfile)
        if len(lines) >= 2 and lines[0].strip().endswith("YES"):
            pos = lines[1].find("t=")
            if pos != -1:
                milli_c = int(lines[1][pos + 2:])
                return milli_c / 1000.0
        time.sleep(retry_delay)
    raise RuntimeError("Failed to get a valid reading (CRC not YES).")

def read_temp(scale: str = "C", rounding: str = "nearest") -> Dict:
    """Return {'ok': True, 'tempC' or 'tempF': value} or {'ok': False, 'error': '...'}."""
    try:
        t = read_temp_c_float()
        if rounding == "floor":
            t = math.floor(t)
        elif rounding == "ceil":
            t = math.ceil(t)
        else:
            t = round(t, 1)
        if scale.upper() == "F":
            f = round((float(t) * 9/5) + 32, 1)
            return {"ok": True, "tempF": f}
        return {"ok": True, "tempC": float(t)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
# -------------------------------------------

# ---------- Heart rate helpers ----------
def _init_sensor():
    m = MAX30102()
    for fn, args in [
        ('reset', ()),
        ('set_fifo_average', (4,)),
        ('set_adc_range', (2048,)),
        ('set_sample_rate', (SAMPLE_HZ,)),
        ('set_led_current', (12.5, 12.5)),  # (red_mA, ir_mA)
        ('set_mode', ('SpO2',))
    ]:
        try:
            getattr(m, fn)(*args)
        except Exception:
            pass
    time.sleep(0.1)
    return m

def _collect(m, need, step):
    """Fill and then maintain a rolling window. Yields chunks of size `step`."""
    red, ir = [], []
    start = time.time()
    while len(ir) < need:
        r, i = m.read_sequential()
        if r and i:
            red.extend(r); ir.extend(i)
        time.sleep(0.005)
        if time.time() - start > max(3*WINDOW_SEC, 30):
            break
    red = red[-need:]; ir = ir[-need:]
    yield np.array(red, float), np.array(ir, float)

    s_need = need
    while True:
        r, i = m.read_sequential()
        if r and i:
            red.extend(r); ir.extend(i)
            if len(ir) >= s_need + step:
                red = red[-s_need:]; ir = ir[-s_need:]
                yield np.array(red, float), np.array(ir, float)
        time.sleep(0.001)

def _compute_bpm(sig, fs):
    try:
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-9)
        wd, meas = hp.process(sig, sample_rate=fs)
        bpm = float(meas.get('bpm', float('nan')))
        if not np.isfinite(bpm) or bpm < 20 or bpm > 220:
            return None, 'out_of_range'
        return bpm, 'ok'
    # If your HeartPy throws warnings, you can catch like:
    # except BadSignalWarning:
    #     return None, 'noisy'
    except Exception:
        return None, 'error'

def get_hr(duration_sec=20):
    """
    Returns (wait_avg, dbg_iter).

    - dbg_iter: iterate to get live updates while measurement runs.
    - wait_avg(): call at the end to get the final average BPM (int or None).
    """
    q = queue.Queue(maxsize=256)
    state = {'avg': None, 'done': threading.Event()}

    def worker():
        resets = 0
        m = None
        prev_live = None
        good_bpms = []
        need = int(SAMPLE_HZ * WINDOW_SEC)
        step = int(SAMPLE_HZ * STEP_SEC)
        start_t = time.time()

        while time.time() - start_t < duration_sec:
            try:
                if m is None:
                    m = _init_sensor()

                stream = _collect(m, need, step)
                red, ir = next(stream)
                while time.time() - start_t < duration_sec:
                    sig = ir if USE_CHANNEL.lower() == 'ir' else red
                    bpm, status = _compute_bpm(sig, SAMPLE_HZ)
                    if bpm is not None:
                        live = bpm if prev_live is None else (EMA_ALPHA*bpm + (1-EMA_ALPHA)*prev_live)
                        prev_live = live
                        good_bpms.append(live)
                        msg = f"LIVE  {int(round(live))} BPM"
                    else:
                        msg = {
                            'noisy': "LIVE  --  (noisy / adjust finger)",
                            'error': "LIVE  --  (processing error)",
                            'out_of_range': "LIVE  --  (out of range)"
                        }.get(status, "LIVE  --")
                    try: q.put_nowait(msg)
                    except queue.Full: pass

                    red, ir = next(stream)

            except StopIteration:
                pass
            except OSError as e:
                resets += 1
                try: q.put_nowait(f"I2C error: {e}. Reinitializing…")
                except queue.Full: pass
                m = None
                if resets > MAX_RESETS: break
                time.sleep(0.5)
            except Exception as e:
                try: q.put_nowait(f"Unexpected error: {e}")
                except queue.Full: pass
                time.sleep(0.2)

        state['avg'] = int(round(float(np.mean(good_bpms)))) if good_bpms else None
        try: q.put_nowait(None)
        except queue.Full: pass
        state['done'].set()

    threading.Thread(target=worker, daemon=True).start()

    def dbg_iter():
        while True:
            msg = q.get()
            if msg is None: break
            yield msg

    def wait_avg():
        state['done'].wait()
        return state['avg']

    return wait_avg, dbg_iter()
# ------------------------------------------

# ---------- Combined API ----------
def get_vitals(duration_sec: int = 20, scale: str = "C"):
    """
    Returns (wait_final, dbg_iter)

    - dbg_iter yields live HR lines (and a constant temp tag).
    - wait_final() -> {'ok': True, 'avg_hr': int|None, 'tempC|tempF': val}
                      or {'ok': False, 'error': '...'}
    """
    t = read_temp(scale=scale)
    if not t.get("ok"):
        # No stream in error case (temp failed); still try HR so user sees it?
        # If you prefer to abort entirely, comment the next two lines and return error.
        hr_wait, hr_dbg = get_hr(duration_sec=duration_sec)
        def _err_dbg():
            yield f"TEMP ERROR: {t.get('error')}"
            for line in hr_dbg: yield line
        def _err_wait():
            return {'ok': False, 'error': t.get('error')}
        return _err_wait, _err_dbg()

    temp_key = "tempF" if "tempF" in t else "tempC"
    temp_val = t[temp_key]

    hr_wait, hr_dbg = get_hr(duration_sec=duration_sec)

    def dbg_iter():
        """Prefix each HR line with the (constant) temperature."""
        for line in hr_dbg:
            yield f"{temp_key.upper()} {temp_val} | {line}"

    def wait_final():
        avg = hr_wait()
        return {'ok': True, 'avg_hr': avg, temp_key: temp_val}

    return wait_final, dbg_iter()
# -----------------------------------

if __name__ == "__main__":
    # Example usage:
    wait_final, dbg = get_vitals(duration_sec=15, scale="F")
    for line in dbg:
        print(line)
    print("FINAL:", wait_final())