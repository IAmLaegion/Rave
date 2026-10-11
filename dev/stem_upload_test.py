"""End-to-end stem-pack test against a running Rave: chunked uploads, then a mashup from the packs."""
import json
import os
import sys
import time
import urllib.request
import uuid

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7865"
PACKS = sys.argv[2:]


def form(fields):
    b = uuid.uuid4().hex
    body = b"".join(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
                    for k, v in fields.items()) + f"--{b}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={b}"}


def call(method, path, data=None, headers=None):
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


ids = []
for p in PACKS:
    size = os.path.getsize(p)
    body, h = form({"name": os.path.basename(p), "size": size})
    start = call("POST", "/api/uploads", body, h)
    at, n = 0, 0
    with open(p, "rb") as f:
        while at < size:
            f.seek(at)
            chunk = f.read(start["chunk"])
            at = call("PUT", f"/api/uploads/{start['id']}?offset={at}", chunk,
                      {"Content-Type": "application/octet-stream"})["received"]
            n += 1
    print(f"uploaded {os.path.basename(p)}: {size / 1e6:.0f} MB in {n} chunks -> {start['id']}", flush=True)
    ids.append(start["id"])

opts = {"tracks": [{"stems": i} for i in ids], "style": "earworm", "format": "mp3"}
body, h = form({"options": json.dumps(opts)})
job = call("POST", "/api/jobs", body, h)
print("job", job["id"], job["status"], flush=True)
last = None
while True:
    j = call("GET", f"/api/jobs/{job['id']}")
    msg = (j["status"], j["message"])
    if msg != last:
        print(f"  {j['progress'] * 100:5.1f}%  {j['status']}: {j['message']}", flush=True)
        last = msg
    if j["status"] in ("done", "failed"):
        print(j.get("error") or j.get("report"))
        break
    time.sleep(3)
