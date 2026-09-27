"""Live demo tool 2/3 — RS-232 serial capture logger (Path B, step 1).

Logs every byte the CardioTouch 3000 sends over its RS-232C port, with
timestamps, to a raw .bin file plus a human-readable hex preview log.
This is how the machine's serial protocol gets identified: connect the
null-modem cable, run a recording, and inspect what arrives.

Typical ECG-device serial settings to try (from CardioTouch-class
manuals): 9600 or 115200 baud, 8 data bits, no parity, 1 stop bit.

Usage:
    py tools/serial_capture.py --list
    py tools/serial_capture.py --port COM3 --baud 9600
    py tools/serial_capture.py --port COM3 --baud 9600 --out mycapture
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import serial
from serial.tools import list_ports

OUT_DIR = Path(__file__).resolve().parents[1] / "captures"


def list_serial_ports() -> None:
    ports = list(list_ports.comports())
    if not ports:
        print("no serial ports found (plug in the USB-serial adapter first)")
        return
    for p in ports:
        print(f"{p.device}  {p.description}")


def capture(port: str, baud: int, out_base: Path, timeout: float) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    bin_path = out_base.with_suffix(".bin")
    txt_path = out_base.with_suffix(".log")

    ser = serial.Serial(port=port, baudrate=baud, bytesize=8,
                        parity=serial.PARITY_NONE, stopbits=1,
                        timeout=timeout)
    print(f"capturing {port} @ {baud} 8N1 -> {bin_path}")
    print("start a recording on the machine; Ctrl+C to stop")
    total = 0
    t0 = time.time()
    try:
        with open(bin_path, "ab") as bin_fh, \
                open(txt_path, "a", encoding="utf-8") as txt_fh:
            while True:
                data = ser.read(4096)
                if not data:
                    continue
                stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                bin_fh.write(data)
                hex_preview = " ".join(f"{b:02x}" for b in data[:64])
                ascii_preview = "".join(
                    chr(b) if 32 <= b < 127 else "." for b in data[:64])
                txt_fh.write(f"{stamp} +{len(data):5d}B  "
                             f"{hex_preview:<192}  |{ascii_preview}|\n")
                txt_fh.flush()
                total += len(data)
                print(f"\r{total} bytes captured "
                      f"({total / (time.time() - t0 + 1e-9):.0f} B/s)",
                      end="", flush=True)
    except KeyboardInterrupt:
        print(f"\nstopped. {total} bytes total.")
        print(f"raw bytes : {bin_path}")
        print(f"hex log   : {txt_path}")
        print("next: look for readable structure in the .log (file magic, "
              "labels like 'SCPECG', 'II', rates) and share both files "
              "to reverse-engineer the format.")
    finally:
        ser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true",
                        help="list available serial ports and exit")
    parser.add_argument("--port", help="serial port, e.g. COM3")
    parser.add_argument("--baud", type=int, default=9600,
                        help="baud rate (default 9600; also try 115200)")
    parser.add_argument("--out", default=None,
                        help="capture basename (default captures/ct_<ts>)")
    parser.add_argument("--timeout", type=float, default=1.0,
                        help="read timeout seconds (default 1)")
    args = parser.parse_args()

    if args.list:
        list_serial_ports()
        return
    if not args.port:
        parser.error("--port is required (or use --list)")

    out_base = Path(args.out) if args.out else (
        OUT_DIR / f"ct_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    capture(args.port, args.baud, out_base, args.timeout)


if __name__ == "__main__":
    main()
