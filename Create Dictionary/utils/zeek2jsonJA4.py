"""
Bruk: py zeek2jsonJA4.py --ssl <ssl_log_path> [--conn <conn_log_path>]
Eksempel: py zeek2jsonJA4.py --ssl ssl.log --conn conn.log > output.json
"""

import sys
import argparse
import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR


separator = "\t"  # Zeek bruker tabulator som separator i loggene


def str_to_bool(value):
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError("Expected a boolean value (true/false).")

def clean_value(val):
    # Zeek bruker "-" for tomme felt.
    if val is None or val == "-" or val == "(empty)":
        return None
    return val


def format_zeek_timestamp_to_zulu(ts_value):
    cleaned = clean_value(ts_value)
    if cleaned is None:
        return None

    try:
        ts_decimal = Decimal(cleaned)
    except (InvalidOperation, TypeError, ValueError):
        return None

    seconds = int(ts_decimal.to_integral_value(rounding=ROUND_FLOOR))
    fractional = ts_decimal - Decimal(seconds)
    nanoseconds = int((fractional * Decimal("1000000000")).to_integral_value(rounding=ROUND_FLOOR))

    dt_utc = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{dt_utc.strftime('%Y-%m-%dT%H:%M:%S')}.{nanoseconds:09d}Z"

def get_fields_from_log(log_path):
    with open(log_path, "r") as f:
        for line in f:
            if line.startswith("#fields"):
                return line.strip().split(separator)[1:]
    return None

def main():
    parser = argparse.ArgumentParser(description="Extracts JA4, JA4s, JA4t, JA4ts from Zeek SSL and Conn logs to JSON stdout.")
    parser.add_argument("--ssl", required=True)
    parser.add_argument("--conn")
    parser.add_argument(
        "--complete_json",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Set true/false. True outputs a complete JSON array; false outputs one JSON object per line.",
    )
    args = parser.parse_args()

    # 1. Forbered Conn-data (hvis filen finnes)
    # Vi lager en dictionary: { "UID": {"ja4t": "...", "ja4ts": "..."} }
    conn_map = {}
    if args.conn:
        conn_fields = get_fields_from_log(args.conn)
        if conn_fields:
            with open(args.conn, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split(separator)
                    if len(parts) == len(conn_fields):
                        c_entry = dict(zip(conn_fields, parts))
                        uid = c_entry.get("uid")
                        if uid:
                            conn_map[uid] = {
                                "ja4t": c_entry.get("ja4t"),
                                "ja4ts": c_entry.get("ja4ts")
                            }

    # 2. Prosesser SSL-loggen linje for linje
    ssl_fields = get_fields_from_log(args.ssl)
    if not ssl_fields:
        print("Error: could not find #fields in SSL log", file=sys.stderr)
        return
    
    if args.complete_json:
        print("[")  # Start JSON-arrayen
    first_entry = True
    with open(args.ssl, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            
            values = line.strip().split(separator)
            if len(values) != len(ssl_fields):
                continue

            ssl_entry = dict(zip(ssl_fields, values))
            uid = ssl_entry.get("uid")

            # 3. Hent "bonus-data" fra conn_map hvis UID finnes der
            extra_data = conn_map.get(uid, {})
            
            # 4. Bygg og print resultatet med en gang
            output_data = {
                "timestamp": format_zeek_timestamp_to_zulu(ssl_entry.get("ts")),
                "dst": clean_value(ssl_entry.get("id.resp_h")),
                "srcport": clean_value(ssl_entry.get("id.orig_p")),
                "dstport": clean_value(ssl_entry.get("id.resp_p")),
                "JA4": clean_value(ssl_entry.get("ja4")),
                "JA4_r": clean_value(ssl_entry.get("ja4_r")),
                "JA4S": clean_value(ssl_entry.get("ja4s")),
                "JA4S_r": clean_value(ssl_entry.get("ja4s_r")),
                "JA4T": clean_value(extra_data.get("ja4t")),
                "JA4TS": clean_value(extra_data.get("ja4ts")),
                "domain": clean_value(ssl_entry.get("server_name")),
            }
            
            # Vi printer bare hvis vi i det minste har en JA4
            if output_data["JA4"] is not None:
                if not first_entry or not args.complete_json:
                    print(",")  # Legg til komma mellom objektene
                print(json.dumps(output_data), end="")

                first_entry = False
    if args.complete_json:
        print("\n]")  # Avslutt JSON-arrayen
if __name__ == "__main__":
    main()