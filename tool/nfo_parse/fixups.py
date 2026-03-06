"""
fixups.py — Post-processing fixups applied to parsed house data before NML emission.

Each fixup function takes the cargo acceptance data (or other relevant fields)
and returns a (possibly modified) version plus metadata describing what was
changed.  The caller is responsible for writing ``/* CUSTOM: … */`` comments
into the NML output when a fixup fires.

Fixups
------
    fixup_food_only_add_pass_mail
        If a house accepts *only* FOOD (no PASS, no MAIL, no other cargos),
        add PASS and MAIL with the same acceptance amount as FOOD so that
        the house also contributes to passenger and mail coverage.
"""

from __future__ import annotations

from typing import Sequence


def fixup_food_only_add_pass_mail(
    cargos: list[tuple[str, int]],
) -> tuple[list[tuple[str, int]], bool]:
    """If the house accepts only FOOD, add equal PASS and MAIL acceptance.

    Parameters
    ----------
    cargos:
        List of ``(cargo_label, amount)`` pairs where *amount* > 0.

    Returns
    -------
    (new_cargos, applied):
        *new_cargos* is the (possibly extended) cargo list.
        *applied* is ``True`` when the fixup actually fired.
    """
    if not cargos:
        return cargos, False

    # Check: every accepted cargo must be FOOD (there may be multiple FOOD
    # entries in theory, but typically there is exactly one).
    labels = {label for label, _amt in cargos}
    if labels != {"FOOD"}:
        return cargos, False

    # Sum up all FOOD amounts (normally just one entry).
    food_amount = sum(amt for label, amt in cargos if label == "FOOD")
    if food_amount <= 0:
        return cargos, False

    # Prepend PASS and MAIL with the same amount, keep FOOD last.
    new_cargos: list[tuple[str, int]] = [
        ("PASS", food_amount),
        ("MAIL", food_amount),
    ] + list(cargos)

    return new_cargos, True
