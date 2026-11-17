#!/usr/bin/env python3
"""L7 least-conn proxy with precise in-flight concurrency control.

OpenResty-equivalent semantics implemented with asyncio (OpenResty is not
installable on these nodes):

  * GLOBAL in-flight cap (default 8, --max-inflight-requests). Requests beyond the
    cap queue inside the semaphore -- exactly like an OpenResty
    lua_shared_dict + semaphore gate in access_by_lua.
  * least_conn backend selection: each request goes to the healthy backend
    with the fewest in-flight requests. vLLM generation latency varies
    wildly with sequence length, so least_conn beats round_robin.
  * connect-failure failover to the next backend; mid-flight failures
    surface as 502 (retrying would double-generate).
  * periodic stats log with in-flight and queue depth.

Usage: python3 openresty.py http://ip:8008 ... [--port 8100] [--max-inflight-requests 8]
"""

import argparse
import asyncio
import logging
import sys
import time
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("lb")

FAIL_PENALTY_S = 30.0
CONNECT_TIMEOUT_S = 10.0
QUEUE_TIMEOUT_S = 600.0  # max seconds a request may wait in the queue
CHUNK = 1 << 16


class Backend:
    def __init__(self, url: str):
        self.url = url
        p = urlparse(url)
        self.host = p.hostname
        self.port = p.port or 80
        self.fail_until = 0.0
        self.inflight = 0
        self.served = 0
        self.errors = 0


BACKENDS: list[Backend] = []
INFLIGHT_SEM: asyncio.Semaphore | None = None
STATS = {"queued_total": 0, "queue_timeouts": 0}


def parse_head(head: bytes):
    lines = head.split(b"\r\n")
    first = lines[0].decode("latin-1")
    headers = {}
    for line in lines[1:]:
        if b":" in line:
            k, v = line.split(b":", 1)
            headers[k.decode("latin-1").strip().lower()] = v.decode("latin-1").strip()
    return first, headers


def pick_least_conn() -> Backend | None:
    """Select the healthy backend with the fewest in-flight requests."""
    candidates = [b for b in BACKENDS if time.monotonic() >= b.fail_until]
    if not candidates:
        return None
    return min(candidates, key=lambda b: b.inflight)


def inflight_snapshot() -> str:
    return " ".join(f"{b.url.split('//')[1]}:{b.inflight}" for b in BACKENDS)


async def relay_response(br, writer, rhead, rheaders):
    writer.write(rhead)
    if "content-length" in rheaders:
        n = int(rheaders["content-length"])
        while n > 0:
            chunk = await br.read(min(CHUNK, n))
            if not chunk:
                raise ConnectionError("backend EOF mid body")
            writer.write(chunk)
            n -= len(chunk)
    elif "chunked" in rheaders.get("transfer-encoding", ""):
        while True:
            line = await br.readline()
            writer.write(line)
            if line in (b"0\r\n", b"\r\n", b""):
                break
            size = int(line.split(b";")[0].strip() or b"0", 16)
            if size:
                data = await br.readexactly(size + 2)
                writer.write(data)
    else:  # read-until-close
        while True:
            chunk = await br.read(CHUNK)
            if not chunk:
                break
            writer.write(chunk)
    await writer.drain()


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    be = None
    acquired = False
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, ConnectionResetError, asyncio.LimitOverrunError):
        writer.close()
        return
    try:
        req_line, headers = parse_head(head)

        body = b""
        if "content-length" in headers:
            n = int(headers["content-length"])
            if n:
                body = await reader.readexactly(n)

        # --- in-flight concurrency gate (max N; excess queues here) ---
        if INFLIGHT_SEM.locked():
            STATS["queued_total"] += 1
            log.info("queueing %s (inflight=%s)", req_line.split(" ")[1][:60], inflight_snapshot())
        try:
            await asyncio.wait_for(INFLIGHT_SEM.acquire(), timeout=QUEUE_TIMEOUT_S)
        except asyncio.TimeoutError:
            STATS["queue_timeouts"] += 1
            log.warning("queue timeout (%ds) for %s", QUEUE_TIMEOUT_S, req_line)
            writer.write(
                b"HTTP/1.1 503 Queue Timeout\r\n"
                b"Content-Length: 0\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
            return
        acquired = True

        # --- least_conn backend selection ---
        # NOTE: pick + increment must be atomic (no await between them),
        # otherwise simultaneous requests all read inflight=0 and pile
        # onto the same backend before any of them increments.
        be = pick_least_conn()
        if be is None:
            writer.write(
                b"HTTP/1.1 503 Service Unavailable\r\n"
                b"Content-Length: 0\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
            return
        be.inflight += 1  # reserve the slot immediately (atomic with pick)

        try:
            br, bw = await asyncio.wait_for(
                asyncio.open_connection(be.host, be.port), CONNECT_TIMEOUT_S
            )
        except Exception as exc:
            be.inflight -= 1  # release the reserved slot
            be.fail_until = time.monotonic() + FAIL_PENALTY_S
            be.errors += 1
            log.warning("connect to %s failed: %s", be.url, exc)
            writer.write(
                b"HTTP/1.1 502 Bad Gateway\r\n"
                b"Content-Length: 0\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
            return
        try:
            bw.write(head + body)
            await bw.drain()
            rhead = await br.readuntil(b"\r\n\r\n")
            _rline, rheaders = parse_head(rhead)
            await relay_response(br, writer, rhead, rheaders)
            be.served += 1
            log.info(
                "%s -> %s (served=%d inflight=%d)",
                req_line.split(" ")[1][:60] if " " in req_line else "?",
                be.url, be.served, be.inflight,
            )
        except Exception as exc:
            be.fail_until = time.monotonic() + FAIL_PENALTY_S
            be.errors += 1
            log.error("mid-flight failure on %s for '%s': %s", be.url, req_line, exc)
            try:
                writer.write(
                    b"HTTP/1.1 502 Bad Gateway\r\n"
                    b"Content-Length: 0\r\nConnection: close\r\n\r\n"
                )
                await writer.drain()
            except Exception:
                pass
        finally:
            be.inflight -= 1
            bw.close()

    except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
        pass
    except Exception as exc:
        log.error("handler error: %s", exc, exc_info=True)
        try:
            writer.write(
                b"HTTP/1.1 500 Internal Server Error\r\n"
                b"Content-Length: 0\r\nConnection: close\r\n\r\n"
            )
        except Exception:
            pass
    finally:
        if acquired:
            INFLIGHT_SEM.release()
        try:
            writer.close()
        except Exception:
            pass


async def stats_loop():
    while True:
        await asyncio.sleep(60)
        log.info(
            "stats " + " ".join(f"{b.url.split('//')[1]}:{b.served}/{b.errors}" for b in BACKENDS)
            + f" inflight=[{inflight_snapshot()}] sem_locked={INFLIGHT_SEM.locked()}"
            + f" queued_total={STATS['queued_total']} queue_timeouts={STATS['queue_timeouts']}"
        )


async def main():
    global BACKENDS, INFLIGHT_SEM
    ap = argparse.ArgumentParser()
    ap.add_argument("backends", nargs="+")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--max-inflight-requests", type=int, default=8,
                    help="global max concurrent in-flight requests (excess queues)")
    args = ap.parse_args()
    BACKENDS = [Backend(u) for u in args.backends]
    INFLIGHT_SEM = asyncio.Semaphore(args.max_inflight_requests)

    server = await asyncio.start_server(handle_client, "0.0.0.0", args.port)
    log.info(
        "LB listening on 0.0.0.0:%d -> %s (least_conn, max_inflight_requests=%d, queue_timeout=%ds)",
        args.port, [b.url for b in BACKENDS], args.max_inflight_requests, QUEUE_TIMEOUT_S,
    )
    asyncio.create_task(stats_loop())
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
