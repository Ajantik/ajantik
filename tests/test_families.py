"""Published names stay separate from internal ids."""

import pytest

from ajantik.families import FAMILIES, SIGNALS, family_of, module_of, signal_rank
from ajantik.faults import MODULES


def test_every_fault_module_has_a_published_name():
    """A family with no label would appear in the archive as an internal id."""
    assert set(MODULES) <= set(FAMILIES), set(MODULES) - set(FAMILIES)


def test_ids_map_to_families():
    assert module_of("phantom-success:set_field") == "phantom_success"
    assert family_of("phantom-success:set_field").name == "Phantom success"
    assert family_of("clean") is None


def test_signals_are_declared_values():
    assert {f.signal for f in FAMILIES.values()} <= set(SIGNALS)


def test_detectability_order_matches_the_measured_one():
    """The first round found the silent-failure rate falling along this order:
    phantom success 100%, truncated 50%, premature 30%, transient 0% (Haiku, n=10)."""
    order = ["phantom-success:set_field", "premature-read:read_intake",
             "truncated-reply:read_intake", "transient-error:set_field"]
    assert [signal_rank(f) for f in order] == sorted(signal_rank(f) for f in order)
    assert signal_rank("phantom-success:x") < signal_rank("transient-error:x")


@pytest.mark.parametrize("module", sorted(FAMILIES))
def test_each_family_states_what_the_tool_replies(module):
    f = FAMILIES[module]
    assert f.reply and f.description and f.name
