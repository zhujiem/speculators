#!/usr/bin/env python3
"""L7 proxy with batch-affinity routing and precise in-flight concurrency control.

OpenResty-equivalent semantics implemented with asyncio (OpenResty is not
installable on these nodes):

  * GLOBAL in-flight BATCH cap (default 4, --max-inflight-batches). One
    X-Batch-Id consumes exactly one slot: the first request of a batch (the
    leader) acquires the slot and followers join the same entry, so the whole
    packed batch is dispatched to the backend at the same moment -- a
    per-REQUEST cap would dribble the batch's samples out one by one and
    vLLM could never fill a 32K token engine step with them. Headerless
    requests get a unique solo key and behave like the old per-request
    semaphore. Batches beyond the cap queue inside the semaphore -- exactly
    like an OpenResty lua_shared_dict + semaphore gate in access_by_lua.
  * batch-affinity routing: requests carrying the same ``X-Batch-Id`` header
    stick to the same backend. Data-loader workers pack sequences up to a
    token budget (e.g. 32K) that matches each vLLM instance's
    ``max_num_batched_tokens``; keeping one packed batch on one instance lets
    the vLLM scheduler co-batch them into a single engine step. Batches are
    spread across backends by a stable hash of the id; requests without the
    header fall back to least_conn, as do batches whose backend is in the
    failure-penalty window.
  * least_conn backend selection: each request goes to the healthy backend
    with the fewest in-flight requests. vLLM generation latency varies
    wildly with sequence length, so least_conn beats round_robin.
  * connect-failure failover to the next backend; mid-flight failures
    surface as 502 (retrying would double-generate).
  * periodic stats log with in-flight and queue depth.

Usage: python3 openresty.py http://ip:8008 ... [--port 8100] [--max-inflight-batches 4]

Clients may send an ``X-Batch-Id`` header to request batch-affinity routing.
"""

import argparse
import asyncio
import itertools
import logging
import sys
import time
import zlib
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
STATS = {"queued_total": 0, "queue_timeouts": 0, "affinity_requests": 0}
# batch_id -> pinned Backend. Entries are written on first sight; the map is
# capped and wholesale-reset (batches are short-lived, so stale pins only
# cost one re-hash at worst).
BATCH_ROUTES: dict[str, Backend] = {}
BATCH_ROUTES_MAX = 4096
BATCH_ID_HEADER = "x-batch-id"


class BatchEntry:
    """Shared gate state for one in-flight batch (or one solo request).

    ``count`` is the number of live requests in the batch. The leader (the
    request that created the entry) owns the semaphore slot; ``ready`` is
    set once that slot is acquired (or the leader gave up, ``failed``) so
    all followers wake at the same instant.
    """

    __slots__ = ("count", "ready", "slot_acquired", "failed")

    def __init__(self):
        self.count = 0
        self.ready = asyncio.Event()
        self.slot_acquired = False
        self.failed = False


ACTIVE_BATCHES: dict[str, BatchEntry] = {}
_SOLO_SEQ = itertools.count()


def batch_key(batch_id: str | None) -> str:
    """Gate key: the header value, or a unique solo key per request."""
    return batch_id if batch_id else f"__solo:{next(_SOLO_SEQ)}"


async def acquire_batch_slot(
    key: str, req_desc: str
) -> BatchEntry | None:
    """Join or create the gate entry for one request's batch.

    Returns the entry once the batch holds a semaphore slot, or None on
    queue timeout (caller answers 503). The leader queues on the global
    semaphore; followers queue on the entry's ready event so the whole
    batch is released to the backend at the same instant. A leader timeout
    fails the whole entry so followers answer 503 immediately instead of
    waiting out their own timeout.
    """
    entry = ACTIVE_BATCHES.get(key)
    leader = entry is None
    if leader:
        entry = BatchEntry()
        ACTIVE_BATCHES[key] = entry
    entry.count += 1  # atomic with the get/create above (no await between)

    if entry.ready.is_set() and not entry.failed:
        return entry  # batch already dispatched; join immediately

    if INFLIGHT_SEM.locked():
        STATS["queued_total"] += 1
        log.info(
            "queueing %s (inflight_batches=%d inflight=%s)",
            req_desc, len(ACTIVE_BATCHES), inflight_snapshot(),
        )
    try:
        if leader:
            await asyncio.wait_for(INFLIGHT_SEM.acquire(), timeout=QUEUE_TIMEOUT_S)
        else:
            await asyncio.wait_for(entry.ready.wait(), timeout=QUEUE_TIMEOUT_S)
    except asyncio.TimeoutError:
        STATS["queue_timeouts"] += 1
        log.warning("queue timeout (%ds) for %s", QUEUE_TIMEOUT_S, req_desc)
        if leader and not entry.slot_acquired:
            entry.failed = True
            entry.ready.set()  # fail followers fast
        release_batch_slot(key, entry)
        return None
    if entry.failed:
        release_batch_slot(key, entry)
        return None
    if leader:
        entry.slot_acquired = True
        entry.ready.set()
    return entry


def release_batch_slot(key: str, entry: BatchEntry) -> None:
    """Drop one request from its batch; the last one frees the slot."""
    entry.count -= 1
    if entry.count == 0:
        if entry.slot_acquired:
            INFLIGHT_SEM.release()
        ACTIVE_BATCHES.pop(key, None)


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


def pick_backend(batch_id: str | None) -> tuple[Backend | None, bool]:
    """Select the backend for one request.

    Returns ``(backend, used_affinity)``. With ``batch_id`` the choice is
    pinned per id (stable hash over the full backend list, so the healthy
    set changing size never re-shuffles unrelated batches); an unhealthy pin
    falls back to least_conn so the batch still gets served. Without
    ``batch_id`` this is plain least_conn.
    """
    if not batch_id:
        return pick_least_conn(), False
    now = time.monotonic()
    candidates = [b for b in BACKENDS if now >= b.fail_until]
    if not candidates:
        return None, False
    be = BATCH_ROUTES.get(batch_id)
    if be is not None and now >= be.fail_until:
        return be, True
    be = BACKENDS[zlib.crc32(batch_id.encode()) % len(BACKENDS)]
    if now < be.fail_until:
        be = pick_least_conn()
        if be is None:
            return None, False
    if len(BATCH_ROUTES) >= BATCH_ROUTES_MAX:
        BATCH_ROUTES.clear()
    BATCH_ROUTES[batch_id] = be
    return be, True


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
    key = None
    entry: BatchEntry | None = None
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

        batch_id = headers.get(BATCH_ID_HEADER)
        key = batch_key(batch_id)

        # --- in-flight batch gate (max N batches; excess queues here) ---
        entry = await acquire_batch_slot(key, req_line.split(" ")[1][:60] if " " in req_line else "?")
        if entry is None:
            writer.write(
                b"HTTP/1.1 503 Queue Timeout\r\n"
                b"Content-Length: 0\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
            return

        # --- backend selection: batch affinity, else least_conn ---
        # NOTE: pick + increment must be atomic (no await between them),
        # otherwise simultaneous requests all read inflight=0 and pile
        # onto the same backend before any of them increments.
        be, used_affinity = pick_backend(batch_id)
        if used_affinity:
            STATS["affinity_requests"] += 1
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
                "%s -> %s (served=%d inflight=%d%s)",
                req_line.split(" ")[1][:60] if " " in req_line else "?",
                be.url, be.served, be.inflight,
                f" batch={batch_id}" if batch_id else "",
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
        if entry is not None:
            release_batch_slot(key, entry)
        try:
            writer.close()
        except Exception:
            pass


async def stats_loop():
    while True:
        await asyncio.sleep(60)
        log.info(
            "stats " + " ".join(f"{b.url.split('//')[1]}:{b.served}/{b.errors}" for b in BACKENDS)
            + f" inflight=[{inflight_snapshot()}] inflight_requests={sum(b.inflight for b in BACKENDS)}"
            + f" inflight_batches={len(ACTIVE_BATCHES)}"
            + f" sem_locked={INFLIGHT_SEM.locked()}"
            + f" queued_total={STATS['queued_total']} queue_timeouts={STATS['queue_timeouts']}"
            + f" affinity_requests={STATS['affinity_requests']} batch_routes={len(BATCH_ROUTES)}"
        )


async def main():
    global BACKENDS, INFLIGHT_SEM
    ap = argparse.ArgumentParser()
    ap.add_argument("backends", nargs="+")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--max-inflight-batches", type=int, default=4,
                    help="global max concurrent in-flight batches (one X-Batch-Id "
                         "per slot; excess batches queue)")
    args = ap.parse_args()
    BACKENDS = [Backend(u) for u in args.backends]
    INFLIGHT_SEM = asyncio.Semaphore(args.max_inflight_batches)

    server = await asyncio.start_server(handle_client, "0.0.0.0", args.port)
    log.info(
        "LB listening on 0.0.0.0:%d -> %s (batch-affinity + least_conn fallback, max_inflight_batches=%d, queue_timeout=%ds)",
        args.port, [b.url for b in BACKENDS], args.max_inflight_batches, QUEUE_TIMEOUT_S,
    )
    asyncio.create_task(stats_loop())
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
