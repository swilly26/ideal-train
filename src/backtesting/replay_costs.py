"""The replay cost model, extracted so more than one engine can pay the same bill.

Why this module exists
----------------------
The turbo replay (``src/backtesting/turbo_engine.py``, PR #41/#cfbc715) carries
the team's *honest* cost lines: 2 bps or 1 cent of adverse slippage per fill,
whichever is larger, inside the fill price, plus optional half-spread and
commissions.  Those lines are why the turbo replay reports −80.4 % where the
live account showed +18.3 %, and they are the reason round 1 of the edge search
found that the only positive-gross lever (trading less) still lost money.

Stage 1 of the strategy search runs four *new* strategy families through the
same harness.  Re-deriving costs for them would make their numbers
incomparable with everything already measured, so the model is **moved** here
verbatim and the old engine keeps using it through the same functions;
``tests/test_replay_costs_parity.py`` pins the two paths together.

Semantics (unchanged from ``TurboReplay``)
------------------------------------------
* ``slip_per_share(price) = max(|price| * slippage_pct, slippage_abs)
  + |price| * half_spread_pct``
* a buy fills at ``price + slip``, a sell at ``price - slip`` (always adverse),
* ``fees(qty, price) = qty * commission_per_share + |qty * price| * commission_pct``
  charged on **both** the entry and the exit fill,
* gross P&L is ``(exit_fill - entry_fill) * qty * side``, net is gross minus
  both fees, and the *same-fill* cost drag reported next to it is
  ``qty * (entry_slip + exit_slip) + entry_fee + exit_fee``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

#: The five TurboConfig/ScalpSetConfig fields that make up the cost model.
COST_FIELDS = (
    "slippage_pct",
    "slippage_abs",
    "half_spread_pct",
    "commission_per_share",
    "commission_pct",
)


@dataclass(frozen=True)
class CostModel:
    """One cost model. Same five knobs, same arithmetic as the turbo replay."""

    slippage_pct: float = 0.0002
    slippage_abs: float = 0.01
    half_spread_pct: float = 0.0
    commission_per_share: float = 0.0
    commission_pct: float = 0.0
    label: str = "baseline"

    # ── constructors ───────────────────────────────────────────────────
    @classmethod
    def baseline(cls) -> "CostModel":
        """The realistic model every number in the plan is quoted at."""
        return cls(label="baseline")

    @classmethod
    def zero_cost(cls) -> "CostModel":
        """Gross-edge diagnostic: every cost switched off."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, label="zero_cost")

    @classmethod
    def pessimistic(cls) -> "CostModel":
        """Headroom model: 2 bps/1c slippage + 1 bp half-spread + $0.005/share."""
        return cls(0.0002, 0.01, 0.0001, 0.005, 0.0, label="pessimistic")

    @classmethod
    def from_config(cls, cfg: Any) -> "CostModel":
        """Read the five cost fields off a TurboConfig/ScalpSetConfig-like object."""
        return cls(**{f: float(getattr(cfg, f)) for f in COST_FIELDS})

    def as_dict(self) -> dict:
        d = {f: getattr(self, f) for f in COST_FIELDS}
        d["label"] = self.label
        return d

    # ── the model ──────────────────────────────────────────────────────
    def slip_per_share(self, price: float) -> float:
        """Adverse slippage (per share) charged on one fill at *price*."""
        return (max(abs(price) * self.slippage_pct, self.slippage_abs)
                + abs(price) * self.half_spread_pct)

    def fill_price(self, price: float, is_buy: bool) -> float:
        """Fill price for an order at *price* — slippage is always against you."""
        slip = self.slip_per_share(price)
        return price + slip if is_buy else price - slip

    def fees(self, qty: float, price: float) -> float:
        """Commission charged on one fill of *qty* shares at *price*."""
        return float(qty * self.commission_per_share
                     + abs(qty * price) * self.commission_pct)

    # ── a round trip ───────────────────────────────────────────────────
    def round_trip(
        self,
        qty: float,
        entry_signal_price: float,
        exit_signal_price: float,
        side: int = 1,
    ) -> dict:
        """One round trip priced end to end.

        ``side`` is +1 for long, −1 for short.  Returns the fills, fees, gross
        and net P&L and the same-fill cost drag exactly as ``TurboReplay._close``
        composes them.
        """
        entry_fill = self.fill_price(entry_signal_price, is_buy=side > 0)
        exit_fill = self.fill_price(exit_signal_price, is_buy=side < 0)
        entry_fee = self.fees(qty, entry_fill)
        exit_fee = self.fees(qty, exit_fill)
        gross = side * (exit_fill - entry_fill) * qty
        fees = entry_fee + exit_fee
        drag = qty * (self.slip_per_share(entry_signal_price)
                      + self.slip_per_share(exit_signal_price)) + fees
        return {
            "entry_fill": entry_fill,
            "exit_fill": exit_fill,
            "gross": gross,
            "fees": fees,
            "net": gross - fees,
            "drag": drag,
        }


def cost_fields_match_config(model: CostModel, cfg: Any) -> bool:
    """True when *model* carries exactly the cost fields of *cfg* (parity helper)."""
    return all(float(getattr(cfg, f)) == float(getattr(model, f)) for f in COST_FIELDS)


def config_cost_names() -> tuple[str, ...]:
    """The field names the parity test must find on a TurboConfig."""
    return COST_FIELDS


__all__ = ["COST_FIELDS", "CostModel", "cost_fields_match_config", "config_cost_names"]

# ``fields`` is imported for the dataclass introspection used by tests/tools.
_ = fields
