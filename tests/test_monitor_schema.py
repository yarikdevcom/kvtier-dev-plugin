"""The eviction section's field contract, pinned from SOURCE, not from a value.

The brief asked for "an AST test in test/test_monitor_schema.py pinning that every
parsed key is declared and every declared key is either parsed or allow-listed."
This repo has no `monitor.py` / `EXPECTED_FIELDS` reader (that vocabulary belongs
to the sibling proxy, `kv_tier_diagnostics.py`) - the plugin-side expression of the
same guarantee is a declared field tuple (`evictions.EXPECTED_FIELDS`) plus a single
builder dict (`evictions._fields`) whose key set a degraded view shares by
construction. This module pins THAT contract with the AST, so a rename or a field
added to the wire but not to the declaration fails HERE, at the source, even in a
path no runtime test happens to exercise.

Three independent directions are pinned:
  * every key the `_fields` builder LITERALLY assigns is declared;
  * every declared key is assigned by the builder (or in ALLOW_UNEMITTED);
  * every override the callers pass (section / empty_section) is declared too,
    and the schema_version literal in the source == the declared SCHEMA_VERSION.
"""

from __future__ import annotations

import ast
from pathlib import Path

from kvtier_dev_plugin import evictions

SRC = Path(evictions.__file__).resolve()
TREE = ast.parse(SRC.read_text(encoding="utf-8"))


def _module_constant(name: str):
    """A module-level `name = <literal>` value, evaluated from the AST only."""
    for node in TREE.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign):  # `NAME: type = <literal>`
            target = node.target
            if isinstance(target, ast.Name) and target.id == name and node.value:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a module-level literal in {SRC.name}")


def _declared_fields():
    return set(_module_constant("EXPECTED_FIELDS"))


def _allow_unemitted():
    return set(_module_constant("ALLOW_UNEMITTED"))


def _builder_keys(func_name: str) -> set:
    """Every literal key the `_fields` dict assigns, read off the AST."""
    func = next((n for n in TREE.body
                 if isinstance(n, ast.FunctionDef) and n.name == func_name), None)
    assert func is not None, f"{func_name} is missing from {SRC.name}"
    keys: set = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):  # fields = { ... }
            for value in _dict_literals(node.value):
                for key in value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"):
            for arg in node.args:  # the overrides=... call shape
                for value in _dict_literals(arg):
                    for key in value.keys:
                        if isinstance(key, ast.Constant):
                            keys.add(key.value)
    return keys


def _dict_literals(node):
    if isinstance(node, ast.Dict):
        yield node
    for child in ast.walk(node):
        if isinstance(child, ast.Dict):
            yield child


def _call_kwargs(func_name: str) -> set:
    """The keyword names a caller passes to `_fields(...)` (its overrides)."""
    func = next((n for n in TREE.body
                 if isinstance(n, ast.FunctionDef) and n.name == func_name), None)
    assert func is not None, f"{func_name} is missing from {SRC.name}"
    names: set = set()
    for node in ast.walk(func):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_fields"):
            for kw in node.keywords:
                if kw.arg is not None:
                    names.add(kw.arg)
    return names


# --- direction 1: every key the builder assigns is DECLARED --------------------

def test_every_field_the_builder_assigns_is_declared():
    emitted = _builder_keys("_fields")
    undeclared = emitted - _declared_fields()
    assert not undeclared, (
        f"_fields emits keys not in EXPECTED_FIELDS: {sorted(undeclared)} - "
        f"a new wire field must be declared before it ships")


# --- direction 2: every DECLARED key is emitted (or allow-listed) --------------

def test_every_declared_field_is_emitted_or_allow_listed():
    emitted = _builder_keys("_fields")
    missing = _declared_fields() - emitted - _allow_unemitted()
    assert not missing, (
        f"EXPECTED_FIELDS declares keys _fields never emits: {sorted(missing)} - "
        f"they are declared but will be absent on the wire")


# --- direction 3: the overrides the callers use are declared too ---------------

def test_the_sections_overrides_are_declared():
    for caller in ("section", "empty_section"):
        used = _call_kwargs(caller)
        undeclared = used - _declared_fields()
        assert not undeclared, (
            f"{caller}() overrides a field not in EXPECTED_FIELDS: "
            f"{sorted(undeclared)}")


# --- the schema version is one number, read from the source --------------------

def test_the_declared_schema_version_is_two():
    assert _module_constant("SCHEMA_VERSION") == evictions.SCHEMA_VERSION == 2


# --- the runtime agrees with the source, on every path -------------------------

def test_runtime_sections_carry_exactly_the_declared_keys():
    declared = _declared_fields()
    assert set(evictions.section(None)) == declared
    assert set(evictions.empty_section("boom")) == declared


def test_allow_unemitted_is_empty_today():
    # The escape hatch exists for the future; today nothing legitimately stops
    # being emitted, so the two "every declared key is emitted" checks are total.
    assert _allow_unemitted() == set()
