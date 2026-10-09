"""Transparent scenario scoring; no probability-of-success claim."""


def evaluate_options(shortage, main_budget, capacity, blocked, history_days,
                     peer_budget, fuel_spike=False):
    """Hard feasibility overrides the numerical score; alternatives share no execution."""
    evidence = min(history_days / 7, 1)
    options = []

    def add(key, title, quantity, feasible, verified, backup, reason):
        coverage = min(quantity / shortage, 1) if shortage > 0 else 0
        parts = {"feasibility": 35 if feasible else 0,
                 "shortfall_coverage": round(30 * coverage, 1),
                 "evidence": round(20 * evidence if verified else 0, 1),
                 "operational_checks": 10 if verified else 0,
                 "fuel_stability": 0 if fuel_spike else 5}
        score = round(sum(parts.values()), 1)
        if shortage <= 1e-8 or not feasible:
            status, gate = "GREY", "REJECTED"
        elif not verified or history_days < 7:
            status, gate = "RED", "REVIEW_REQUIRED"
        elif backup or score < 85 or coverage < 1 - 1e-8:
            status, gate = "AMBER", "BACKUP_ONLY"
        else:
            status, gate = "GREEN", "PASS"
        options.append({"id": key, "title": title, "score": score,
                        "score_components": parts, "status": status, "gate": gate,
                        "quantity_tonnes": round(quantity, 3),
                        "shortfall_coverage_pct": round(100 * coverage, 1),
                        "reason": reason, "automatic_real_execution": False,
                        "score_basis": "Heuristic scenario planning score /100; not success probability"})

    qty = 0 if blocked else max(0, min(shortage, main_budget, capacity))
    add("main_transfer", "Replenish from main warehouse", qty, qty > 1e-8,
        True, False, "Route closed" if blocked else
        "Checks main available stock after reservations and protected reserve, plus receiving capacity")
    qty = 0 if blocked else max(0, min(shortage, peer_budget, capacity))
    add("regional_rebalance", "Rebalance from another regional warehouse", qty,
        qty > 1e-8, True, True,
        "Route closed; no regional transfer permitted" if blocked else
        "Protects donor three-day stress buffer and reservations; proposal only, donor budget not reserved")
    add("emergency_supplier", "Seek an emergency supplier", shortage, shortage > 0,
        False, False, "Supplier stock, price, delivery time and route are unverified; no purchase executed")
    add("ration_stock", "Prioritise essential demand and defer other demand", 0,
        shortage > 0, True, True,
        "Backup mitigation only: does not create stock or fulfill unmet demand; customer priorities require review")
    return options
