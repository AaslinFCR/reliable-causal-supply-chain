from scrc.production.decision_options import evaluate_options


def test_complete_verified_plan_passes_and_supplier_stays_uncertain():
    options = evaluate_options(5, 10, 10, False, 7, 4)
    assert [o["status"] for o in options] == ["GREEN", "AMBER", "RED", "AMBER"]
    assert options[0]["score"] == 100


def test_hard_route_gate_overrides_evidence_and_score():
    options = evaluate_options(5, 100, 100, True, 7, 100)
    assert options[0]["gate"] == options[1]["gate"] == "REJECTED"
    assert all(o["status"] != "GREEN" for o in options)


def test_stock_out_and_capacity_limit_cannot_pass():
    assert evaluate_options(5, 0, 10, False, 7, 0)[0]["status"] == "GREY"
    assert evaluate_options(5, 10, 1, False, 7, 0)[0]["status"] == "AMBER"


def test_short_history_and_no_crisis_are_not_green():
    assert evaluate_options(5, 10, 10, False, 3, 0)[0]["status"] == "RED"
    assert all(o["status"] == "GREY" for o in evaluate_options(0, 10, 10, False, 7, 10))
