"""Maia-3 move service: one position in, the moves a human of a given rating would play out.

    POST /maia/move    takes {fen, moves: [uci, ...] (at most 7), selfElo, oppoElo}
                       answers {moves: [{uci, p}] (top 10, most likely first), wdl, ms}
    GET  /maia/health  {"status": "UP"} once the model is loaded, 503 STARTING before

Maia-3 reads the last 8 positions, not just the current one. So `fen` is the position 7 plies before the
one to answer, or the start position early in a game, and `moves` are the moves since: the UCI
`position fen ... moves ...` form. A shorter history is padded with its earliest position, as maia3's own
engine pads one. The whole game is never sent, because replaying it in Python on every request would
cost more than the model does.

`wdl` is [win, draw, loss] for the side to move in the answered position. These are predicted outcomes of
a game between humans of those ratings, not an engine's evaluation. `ms` is the time the request spent in
this process, waiting for the model included, so the caller can tell the service from the network.

Requests are micro-batched. One thread owns the model and runs every forward pass: it takes whatever is
queued, up to MAX_BATCH, and answers it with one pass. A request on an idle service waits for nothing,
and under load the requests that arrived during one pass share the next. The input is built in the
request handler, so a bad request never queues and the model's thread only runs the model.

Overload is refused rather than queued. Past MAX_QUEUED waiting requests, or once one has waited
MAX_WAIT_SECONDS, the answer is 503 {"error": "overloaded"}: a caller still waiting that long has usually
moved on, and a pass spent on it would only make the next request late too.

Every route but health checks X-Analysis-Key, failing closed.

Built on maia3 (https://github.com/CSSLab/maia3, commit 1e13597c42d4858b7cfd7cfdae01e297263364b2). This
service is licensed under the AGPL-3.0 like maia3 itself: see LICENSE.maia3.
"""

import asyncio
import hmac
import logging
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace

import chess
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from maia3.dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
from maia3.model_registry import resolve_model_spec
from maia3.uci import load_model as load_maia3
from maia3.utils import get_all_possible_moves, mirror_move
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("maia")

KEY_HEADER = "X-Analysis-Key"
MODELS_DIR = "/models"
TOP_MOVES = 10
# 8 positions: the one to answer and the 7 before it.
MAX_MOVES = 7
# A batch of 30 takes about 210 ms on one core; whatever is left over goes in the next pass.
MAX_BATCH = 32
MAX_QUEUED = 256
MAX_WAIT_SECONDS = 1.0

SPEC = resolve_model_spec(os.environ["MAIA_MODEL"])
CFG = SimpleNamespace(**SPEC.config, device="cpu",
                      checkpoint_path=os.path.join(MODELS_DIR, SPEC.checkpoint_filename))
ALL_MOVES = get_all_possible_moves()
MOVE_INDEX = {uci: i for i, uci in enumerate(ALL_MOVES)}


def load_model():
    """The checkpoint from /models, where the image put it.

    maia3 loads with strict=False and only prints what did not match, which would leave a layer at its
    random start behind a healthy service, so a mismatch stops the start here.
    """
    net = load_maia3(CFG)
    checkpoint = torch.load(CFG.checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    found = {key.replace("smolgen", "gab") for key in state}
    expected = set(net.state_dict())
    if found != expected:
        raise RuntimeError(f"checkpoint does not match the model: missing {sorted(expected - found)[:5]}, "
                           f"unexpected {sorted(found - expected)[:5]}")
    return net


class BadPosition(ValueError):
    """Why a request names no position to answer; the message is what the caller gets back."""


class Overloaded(Exception):
    """A request refused or dropped because the model cannot keep up."""


def build_input(fen, moves):
    """The model's input for the position `moves` reach from `fen`: tokens of the last 8 positions, the
    legal moves as a mask over maia3's move vocabulary, and whether Black is to move.

    Goes through maia3's own tokenize_board, get_historical_tokens and get_legal_moves_mask, the code
    behind its engine, which it matches. The zeros are the clock inputs of a model that ignores them.
    """
    try:
        board = chess.Board(fen)
    except ValueError:
        raise BadPosition("bad fen") from None
    if not board.is_valid():
        raise BadPosition("bad fen")
    history = deque([tokenize_board(board)], maxlen=CFG.history)
    for uci in moves:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            raise BadPosition("illegal move") from None
        if not board.is_legal(move):
            raise BadPosition("illegal move")
        board.push(move)
        history.append(tokenize_board(board))
    if not any(board.generate_legal_moves()):
        raise BadPosition("no legal moves")
    return (get_historical_tokens(history, CFG, 0, 0, 0, 0), get_legal_moves_mask(board, MOVE_INDEX),
            board.turn == chess.BLACK)


def infer_batch(net, items):
    """Answers (tokens, legal, black, selfElo, oppoElo) items in one forward pass.

    The model sees every position from the mover's side, so a move for Black comes back mirrored, and its
    value is already the side to move's. The value head is [loss, draw, win], the order maia3's own
    wdl_from_value_logits reads.
    """
    tokens = torch.stack([item[0] for item in items])
    legal = torch.stack([item[1] for item in items])
    self_elos = torch.tensor([item[3] for item in items], dtype=torch.long)
    oppo_elos = torch.tensor([item[4] for item in items], dtype=torch.long)
    with torch.inference_mode():
        logits_move, logits_value, _ = net(tokens, self_elos, oppo_elos)
        probs = torch.softmax(logits_move.float().masked_fill(~legal, float("-inf")), dim=-1)
        values = torch.softmax(logits_value.float(), dim=-1).tolist()
        top_probs, top_indices = torch.topk(probs, k=min(TOP_MOVES, probs.size(1)), dim=-1)
    top_probs, top_indices, counts = top_probs.tolist(), top_indices.tolist(), legal.sum(dim=-1).tolist()
    answers = []
    for item, row_probs, row_indices, count, (loss, draw, win) in zip(items, top_probs, top_indices,
                                                                     counts, values):
        black = item[2]
        moves = []
        for index, p in zip(row_indices[:count], row_probs[:count]):
            uci = ALL_MOVES[index]
            moves.append({"uci": mirror_move(uci) if black else uci, "p": round(p, 4)})
        answers.append({"moves": moves, "wdl": [round(win, 4), round(draw, 4), round(loss, 4)]})
    return answers


@dataclass
class _Job:
    item: tuple
    queued_at: float = field(default_factory=time.monotonic)
    future: Future = field(default_factory=Future)


class Engine:
    """The one thread that loads and runs the model.

    Loading happens on that thread too, because torch's thread count is set per thread: the count it is
    given here is the count every forward pass uses.
    """

    def __init__(self, load, threads):
        self._load = load
        self._threads = threads
        self._jobs = queue.SimpleQueue()
        self._net = None
        self.ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="maia-inference", daemon=True)

    def start(self):
        self._thread.start()

    def full(self):
        return self._jobs.qsize() >= MAX_QUEUED

    def submit(self, item):
        job = _Job(item)
        self._jobs.put(job)
        return job.future

    def _run(self):
        torch.set_num_threads(self._threads)
        started = time.perf_counter()
        try:
            self._net = self._load()
        except Exception:
            # Nothing to serve without the model, and the image is the same on every restart, so stop
            # loudly: compose shows the container restarting and a health check fails.
            log.exception("MAIA could not load its model")
            os._exit(1)
        log.info("MAIA_READY model=%s threads=%d loadMs=%d", SPEC.name, torch.get_num_threads(),
                 (time.perf_counter() - started) * 1000)
        self.ready.set()
        while True:
            self._answer(self._next_batch())

    def _next_batch(self):
        """Waits for a request, then takes whatever else is queued, up to MAX_BATCH.

        A request that has waited too long is dropped here, costing no pass, and does not count against
        the batch. A caller that merely disconnects cancels nothing, so this wait is what bounds the
        work done for one. A future is only ever cancelled when the server shuts down.
        """
        jobs = []
        while len(jobs) < MAX_BATCH:
            try:
                job = self._jobs.get(block=not jobs)
            except queue.Empty:
                break
            if not job.future.set_running_or_notify_cancel():
                continue
            if time.monotonic() - job.queued_at > MAX_WAIT_SECONDS:
                job.future.set_exception(Overloaded())
                continue
            jobs.append(job)
        return jobs

    def _answer(self, jobs):
        if not jobs:
            return
        try:
            answers = infer_batch(self._net, [job.item for job in jobs])
        except Exception as e:
            log.exception("MAIA inference failed for a batch of %d", len(jobs))
            for job in jobs:
                job.future.set_exception(e)
            return
        for job, answer in zip(jobs, answers):
            job.future.set_result(answer)


class MoveRequest(BaseModel):
    fen: str = Field(max_length=100)
    moves: list[str] = Field(max_length=MAX_MOVES)
    # maia3's own range. Checked here because a value too large for a tensor would fail its whole batch.
    selfElo: int = Field(ge=0, le=5000)
    oppoElo: int = Field(ge=0, le=5000)


def create_app(key, load=load_model, threads=None):
    if not key:
        log.warning("ANALYSIS_SERVER_KEY is unset, so /maia/move refuses every request. Set it to the "
                    "same secret the caller sends.")
    secret = key.encode()
    engine = Engine(load, threads or int(os.environ.get("MAIA_THREADS", "1")))

    @asynccontextmanager
    async def lifespan(_):
        engine.start()
        yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def require_key(request: Request, call_next):
        # Every route but health, so a route added later is guarded unless it opts out here. Constant
        # time; the key is never logged.
        if request.method == "GET" and request.url.path == "/maia/health":
            return await call_next(request)
        candidate = request.headers.get(KEY_HEADER)
        if secret and candidate is not None and hmac.compare_digest(secret, candidate.encode()):
            return await call_next(request)
        log.warning("MAIA refused a request with a missing or wrong %s", KEY_HEADER)
        return Response(status_code=401)

    @app.get("/maia/health")
    async def health():
        if engine.ready.is_set():
            return {"status": "UP"}
        return JSONResponse({"status": "STARTING"}, status_code=503)

    @app.post("/maia/move")
    async def move(body: MoveRequest):
        received = time.perf_counter()
        if not engine.ready.is_set():
            return JSONResponse({"error": "starting"}, status_code=503)
        if engine.full():
            return JSONResponse({"error": "overloaded"}, status_code=503)
        try:
            tokens, legal, black = build_input(body.fen, body.moves)
        except BadPosition as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        try:
            answer = await asyncio.wrap_future(
                engine.submit((tokens, legal, black, body.selfElo, body.oppoElo)))
        except Overloaded:
            return JSONResponse({"error": "overloaded"}, status_code=503)
        return {**answer, "ms": round((time.perf_counter() - received) * 1000)}

    return app


app = create_app(os.environ.get("ANALYSIS_SERVER_KEY", ""))
