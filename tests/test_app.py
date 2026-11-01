"""The service against the real checkpoint, run by the Dockerfile's test stage:

    docker build --target test analysis-service/maia
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import chess
import pytest
import torch
from fastapi.testclient import TestClient
from maia3.uci import Maia3UCIEngine, parse_args, wdl_from_value_logits

from app import CFG, SPEC, build_input, create_app, infer_batch, load_model

KEY = "test-key"
HEADERS = {"X-Analysis-Key": KEY}

START = chess.STARTING_FEN
BACK_RANK_MATE_IN_1 = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
SCHOLARS_MATE_IN_1 = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
ITALIAN = "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2PP1N2/PP3PPP/RNBQ1RK1 b - - 0 6"
THREE_KING_MOVES = "7k/8/8/8/8/8/8/K7 w - - 0 1"
CHECKMATED = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"
STALEMATE = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
TWO_KNIGHTS = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
POSITIONS = (START, BACK_RANK_MATE_IN_1, SCHOLARS_MATE_IN_1, ITALIAN, THREE_KING_MOVES)
# A mate in one reads 0.85 to 0.97 for the side that has it, measured on the 5M and 23M checkpoints.
WIN_IN_ONE = 0.8

# Each answers a position after this history, from a position it starts from: none at all, the short
# history of an early game, and the full 7 moves, ending with either side to move.
SEVEN = ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"]
HISTORIES = (
    (START, []),
    (ITALIAN, []),
    (START, ["e2e4", "e7e5", "g1f3"]),
    (START, SEVEN),
    (TWO_KNIGHTS, ["f1c4", "f8c5", "c2c3", "g8f6", "d2d3", "d7d6", "e1g1"]),
)
ELOS = ((800, 1500), (1150, 1150), (1900, 1300))
TRANSPOSED = (["g1f3", "g8f6", "d2d4", "d7d5"], ["d2d4", "d7d5", "g1f3", "g8f6"])


def wait_until_up(client, timeout=120):
    deadline = time.monotonic() + timeout
    while client.get("/maia/health").status_code != 200:
        assert time.monotonic() < deadline, "the model did not load"
        time.sleep(0.05)


def ask(client, fen, moves=(), elo=1500, headers=HEADERS):
    return client.post("/maia/move", headers=headers,
                       json={"fen": fen, "moves": list(moves), "selfElo": elo, "oppoElo": elo})


def mirrored(fen):
    return chess.Board(fen).mirror().fen()


def mirrored_uci(uci):
    move = chess.Move.from_uci(uci)
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square),
                      move.promotion).uci()


def item_for(fen, moves, self_elo, oppo_elo):
    return (*build_input(fen, moves), self_elo, oppo_elo)


def engine_answer(engine, fen, moves, self_elo, oppo_elo):
    """The moves maia3's own engine gives, most likely first, reading its history from `moves`."""
    engine.self_elo, engine.oppo_elo = self_elo, oppo_elo
    engine.cmd_position(f"position fen {fen}" + (" moves " + " ".join(moves) if moves else ""))
    _, top = engine.score_moves()
    return [(item["move"].uci(), item["policy"]) for item in top]


def assert_same_as_engine(answer, expected):
    assert [m["uci"] for m in answer["moves"]] == [uci for uci, _ in expected]
    for got, (_, p) in zip(answer["moves"], expected):
        assert abs(got["p"] - p) <= 2e-4


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(KEY)) as c:
        wait_until_up(c)
        yield c


@pytest.fixture(scope="module")
def net():
    return load_model()


@pytest.fixture(scope="module")
def engine():
    instance = Maia3UCIEngine(parse_args(["--model", SPEC.name, "--checkpoint-path", CFG.checkpoint_path,
                                          "--device", "cpu", "--temperature", "0", "--multipv", "10",
                                          "--use-uci-history"]))
    instance.ensure_model_loaded()
    return instance


def test_answers_the_ten_likeliest_legal_moves_most_likely_first(client):
    response = ask(client, START)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"moves", "wdl", "ms"}
    legal = {m.uci() for m in chess.Board(START).legal_moves}
    assert len(body["moves"]) == 10
    assert all(m["uci"] in legal for m in body["moves"])
    probabilities = [m["p"] for m in body["moves"]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert sum(probabilities) <= 1 + 1e-3
    assert isinstance(body["ms"], int) and body["ms"] >= 0


def test_a_position_with_fewer_than_ten_moves_answers_all_of_them(client):
    moves = ask(client, THREE_KING_MOVES).json()["moves"]
    assert sorted(m["uci"] for m in moves) == ["a1a2", "a1b1", "a1b2"]
    assert abs(sum(m["p"] for m in moves) - 1) <= 1e-3


@pytest.mark.parametrize("fen", POSITIONS)
def test_wdl_is_three_probabilities_that_sum_to_one(client, fen):
    wdl = ask(client, fen).json()["wdl"]
    assert len(wdl) == 3
    assert all(0 <= p <= 1 for p in wdl)
    assert abs(sum(wdl) - 1) <= 1e-3


@pytest.mark.parametrize("fen", [BACK_RANK_MATE_IN_1, SCHOLARS_MATE_IN_1])
def test_wdl_is_the_side_to_moves(client, fen):
    win, _, loss = ask(client, fen).json()["wdl"]
    assert win >= WIN_IN_ONE and win > loss
    win, _, loss = ask(client, mirrored(fen)).json()["wdl"]
    assert win >= WIN_IN_ONE and win > loss


@pytest.mark.parametrize("fen", [START, SCHOLARS_MATE_IN_1, ITALIAN])
def test_the_colour_mirrored_position_gets_the_mirrored_answer(client, fen):
    # Maia mirrors a position with Black to move before it looks at it, so the two are one input.
    a = ask(client, fen).json()
    b = ask(client, mirrored(fen)).json()
    assert a["wdl"] == b["wdl"]
    assert [mirrored_uci(m["uci"]) for m in a["moves"]] == [m["uci"] for m in b["moves"]]
    assert [m["p"] for m in a["moves"]] == [m["p"] for m in b["moves"]]


def test_a_batch_answers_what_maia3s_own_engine_would(net, engine):
    cases = [(fen, moves, self_elo, oppo_elo) for fen, moves in HISTORIES for self_elo, oppo_elo in ELOS]
    answers = infer_batch(net, [item_for(*case) for case in cases])
    for case, answer in zip(cases, answers):
        assert_same_as_engine(answer, engine_answer(engine, *case))


def test_the_batched_wdl_is_what_one_position_alone_reads(net):
    cases = [(fen, moves, self_elo, oppo_elo) for fen, moves in HISTORIES for self_elo, oppo_elo in ELOS]
    answers = infer_batch(net, [item_for(*case) for case in cases])
    for (fen, moves, self_elo, oppo_elo), answer in zip(cases, answers):
        tokens, _, _ = build_input(fen, moves)
        with torch.inference_mode():
            _, value, _ = net(tokens.unsqueeze(0), torch.tensor([self_elo]), torch.tensor([oppo_elo]))
        # maia3 reports it in whole permille, so this is its reading of the same value, within one.
        for got, permille in zip(answer["wdl"], wdl_from_value_logits(value[0])):
            assert abs(got - permille / 1000) <= 1.5e-3


def test_one_position_reached_two_ways_is_answered_from_each_ones_own_history(client, engine):
    answers = []
    for moves in TRANSPOSED:
        response = ask(client, START, moves).json()
        assert_same_as_engine(response, engine_answer(engine, START, moves, 1500, 1500))
        answers.append(response)
    board_a, board_b = chess.Board(), chess.Board()
    for move in TRANSPOSED[0]:
        board_a.push_uci(move)
    for move in TRANSPOSED[1]:
        board_b.push_uci(move)
    assert board_a.board_fen() == board_b.board_fen()
    assert answers[0]["moves"] != answers[1]["moves"]


def test_only_the_last_eight_positions_count(client, engine):
    # The 7-move window from a later position gives what the whole game gives maia3's engine.
    played = SEVEN + ["g8f6", "b1c3", "a7a6"]
    board = chess.Board()
    for move in played[:3]:
        board.push_uci(move)
    window = ask(client, board.fen(), played[3:]).json()
    assert_same_as_engine(window, engine_answer(engine, START, played, 1500, 1500))


def test_requests_arriving_together_each_get_their_own_answer(client):
    alone = {fen: ask(client, fen).json() for fen in POSITIONS}
    with ThreadPoolExecutor(40) as pool:
        together = list(pool.map(lambda i: (POSITIONS[i % len(POSITIONS)],
                                            ask(client, POSITIONS[i % len(POSITIONS)])), range(120)))
    for fen, response in together:
        assert response.status_code == 200
        got = {m["uci"]: m["p"] for m in response.json()["moves"]}
        want = {m["uci"]: m["p"] for m in alone[fen]["moves"]}
        assert got.keys() == want.keys()
        assert all(abs(got[uci] - want[uci]) <= 2e-4 for uci in got)


def test_every_route_but_health_needs_the_key(client):
    assert ask(client, START, headers={}).status_code == 401
    assert ask(client, START, headers={"X-Analysis-Key": "wrong"}).status_code == 401
    assert client.get("/maia/anything").status_code == 401
    assert client.get("/maia/health").status_code == 200


def test_a_blank_key_refuses_every_request():
    hold = threading.Event()

    def load():
        hold.wait()

    with TestClient(create_app("", load=load)) as c:
        assert ask(c, START, headers={"X-Analysis-Key": ""}).status_code == 401
        assert ask(c, START, headers={"X-Analysis-Key": "anything"}).status_code == 401
    hold.set()


def test_health_is_starting_until_the_model_loads():
    hold = threading.Event()

    def load():
        hold.wait()

    with TestClient(create_app(KEY, load=load)) as c:
        response = c.get("/maia/health")
        assert response.status_code == 503
        assert response.json() == {"status": "STARTING"}
        assert ask(c, START).status_code == 503
        hold.set()
        wait_until_up(c, timeout=10)
        assert c.get("/maia/health").json() == {"status": "UP"}


@pytest.mark.parametrize("fen, moves, error", [
    ("not a fen", [], "bad fen"),
    ("8/8/8/8/8/8/8/8 w - - 0 1", [], "bad fen"),
    (CHECKMATED, [], "no legal moves"),
    (STALEMATE, [], "no legal moves"),
    (START, ["f2f3", "e7e5", "g2g4", "d8h4"], "no legal moves"),
    (START, ["e2e5"], "illegal move"),
    (START, ["e2e4", "e2e4"], "illegal move"),
    (START, ["0000"], "illegal move"),
    (START, ["zz"], "illegal move"),
])
def test_a_request_with_nothing_to_answer_is_refused(client, fen, moves, error):
    response = ask(client, fen, moves)
    assert response.status_code == 400
    assert response.json() == {"error": error}


@pytest.mark.parametrize("body", [
    {"fen": START, "moves": SEVEN + ["g8f6"], "selfElo": 1500, "oppoElo": 1500},
    {"fen": START, "selfElo": 1500, "oppoElo": 1500},
    {"fen": START, "moves": [], "selfElo": 1500.5, "oppoElo": 1500},
    {"fen": START, "moves": [], "selfElo": 1500},
])
def test_a_malformed_request_is_refused(client, body):
    assert client.post("/maia/move", headers=HEADERS, json=body).status_code == 422
