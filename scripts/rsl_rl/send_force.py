import argparse
import json
import socket

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=5005)
parser.add_argument("--body_name", type=str, default="torso_link")

parser.add_argument("--fx", type=float, default=0.0)
parser.add_argument("--fy", type=float, default=0.0)
parser.add_argument("--fz", type=float, default=0.0)

parser.add_argument("--tx", type=float, default=0.0)
parser.add_argument("--ty", type=float, default=0.0)
parser.add_argument("--tz", type=float, default=0.0)

parser.add_argument("--steps", type=int, default=20)

args = parser.parse_args()

payload = {
    "body_name": args.body_name,
    "force": [args.fx, args.fy, args.fz],
    "torque": [args.tx, args.ty, args.tz],
    "steps": args.steps,
}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(json.dumps(payload).encode("utf-8"), (args.host, args.port))

print(f"[send_force] sent: {payload}")
