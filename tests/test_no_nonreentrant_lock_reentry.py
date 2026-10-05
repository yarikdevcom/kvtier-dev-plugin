r"""No function may hold a non-reentrant lock and call another function that takes it.

`threading.Lock` is NOT reentrant. A thread that acquires it twice blocks on a lock it
already owns: no error, no traceback, no timeout, and on an engine thread that is a
permanently wedged scheduler. `disk_read._state_lock` had exactly this - `_settle`,
`finish_batch` and `cancel_batch` each held it and called `_give_buffer`, which took it
again - and it was found because a unit test hung for six minutes inside
`start_load_kv`. The fix split the locked form (`_give_buffer_locked`) out from the
lock-taking one (`_give_buffer`).

This test turns that lesson into an invariant, and it is written to be believed:

1. it reports ZERO findings on the package today (the absence half);
2. it reports the THREE known historical hits on a transcription of the pre-fix shape,
   and a synthetic two-function bug, so the checker is shown to be able to see the
   thing it claims to check. A checker test that can only pass is worthless, and this
   session paid for that lesson twice;
3. it stays FLOW-SENSITIVE: `with L: work()` followed by an unlocked `give()` later in
   the same function is legal, and the first version of this checker ignored that and
   produced five false positives on already-correct code - which would have sent the
   fix chasing a bug that was not there.

Pure AST, stdlib only, no engine import: this reads the package's source, which is the
same posture as `test_guards.py`'s source scan and `test_dead_code_ast.py`.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PLUGIN = pathlib.Path(__file__).resolve().parents[1] / "kvtier_dev_plugin"

# `RLock` is reentrant and `Semaphore`/`Condition` do not deadlock on re-acquire in
# the way a mutex does; a checker that flagged them would be noise, and noise here
# means the next agent deletes the test.
REENTRANT = {"RLock", "Condition", "Semaphore"}

# The pre-fix `disk_read.py`, transcribed to the shape that mattered (module-level
# lock, a helper that acquires it, three callers that hold it while calling it).
# Validated against the real thing: `git show 4475009:kvtier_dev_plugin/disk_read.py`
# reports these same three functions and nothing else.
PRE_FIX_SOURCE = '''
import threading

_lock = threading.Lock()


def _give_buffer(buffer):
    with _lock:
        POOL.append(buffer)


def _settle(job, outcome):
    with _lock:
        _give_buffer(job["buffer"])


def finish_batch(batch_id):
    with _lock:
        for job in batch_id:
            _give_buffer(job)


def cancel_batch(batch_id):
    with _lock:
        _give_buffer(batch_id)


def legal_now(batch_id):
    with _lock:
        taken = list(batch_id)
    _give_buffer(taken)          # AFTER the block closes: no lock held, legal
'''


def module_locks(tree):
    """Module-level `NAME = threading.Lock()` bindings, by name."""
    locks = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value.func
            kind = getattr(call, "attr", None) or getattr(call, "id", None)
            if kind in ("Lock", "RLock", "Semaphore", "Condition"):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        locks[target.id] = kind
    return locks


def _lock_name(context):
    if isinstance(context, ast.Name):
        return context.id
    if isinstance(context, ast.Call):
        func = context.func
        if isinstance(func, ast.Name):
            return func.id
        if (isinstance(func, ast.Attribute) and func.attr == "acquire"
                and isinstance(func.value, ast.Name)):
            return func.value.id
    return None


def _scan(nodes, inside, held, under_lock):
    """Walk `nodes`, recording which locks are held and which calls sit under them.

    `inside` is the lock stack: it grows at a `with L:` and does not follow the call
    after the block closes, which is what makes the sweep flow-sensitive. Module-level
    rather than a closure because the accumulator dictionaries belong to one function
    being analyzed, not to the loop that walks functions (ruff B023).
    """
    for item in nodes:
        if isinstance(item, ast.With):
            nested = list(inside)
            for entry in item.items:
                name = _lock_name(entry.context_expr)
                if name:
                    held.add(name)
                    nested.append(name)
            _scan(list(item.body), nested, held, under_lock)
            continue
        if isinstance(item, ast.Call):
            func = item.func
            if isinstance(func, ast.Name):
                for lock in inside:
                    under_lock.setdefault(lock, set()).add(func.id)
            elif (isinstance(func, ast.Attribute)
                  and func.attr == "acquire"
                  and isinstance(func.value, ast.Name)):
                held.add(func.value.id)
        _scan(list(ast.iter_child_nodes(item)), inside, held, under_lock)


def analyze(tree):
    """Per function: the locks it ever holds, and the calls made INSIDE each block."""
    out = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        held, under_lock = set(), {}
        _scan(list(node.body), [], held, under_lock)
        out[node.name] = (held, under_lock)
    return out


def findings(source: str, name: str = "<source>") -> list:
    """Every `holder -> callee` pair that would self-deadlock on a plain Lock."""
    tree = ast.parse(source, name)
    locks = module_locks(tree)
    functions = analyze(tree)
    hits = []
    for caller, (held, under) in functions.items():
        for lock in sorted(held):
            if lock not in locks or locks[lock] in REENTRANT:
                continue
            for callee in sorted(under.get(lock, ())):
                if callee not in functions:
                    continue
                callee_held, _ = functions[callee]
                if lock in callee_held:
                    hits.append((caller, callee, lock, locks[lock]))
    return hits


def test_no_function_reacquires_a_lock_it_holds():
    """The invariant over the whole package: zero self-deadlocks, by construction.

    `disk_read._give_buffer_locked` / `_give_buffer` is the pattern this exists to
    keep: a locked internal form for callers that already hold the lock, and the
    lock-taking form only where the source proves the lock is not held.
    """
    reports = {}
    for path in sorted(PLUGIN.glob("*.py")):
        found = findings(path.read_text(), str(path))
        if found:
            reports[path.name] = found
    assert not reports, (
        "a function holds a non-reentrant lock and calls a function that acquires "
        f"it (self-deadlock, no error, no traceback): {reports}")


def test_the_checker_sees_the_bug_it_was_written_for():
    """The detection half, on the historical shape: three hits, named, not silence."""
    found = findings(PRE_FIX_SOURCE, "disk_read_prefix.py")
    pairs = {(caller, callee) for caller, callee, _lock, _kind in found}
    assert pairs == {("_settle", "_give_buffer"),
                     ("finish_batch", "_give_buffer"),
                     ("cancel_batch", "_give_buffer")}, found


def test_the_checker_is_flow_sensitive_not_just_scope_sensitive():
    """A call AFTER the `with` block is legal, and must not be reported.

    `legal_now` is in the same source as the three real hits and is NOT one of them.
    Without this the checker would report five false positives on the fixed tree, and
    a noisy invariant test is a test that gets deleted.
    """
    found = findings(PRE_FIX_SOURCE, "disk_read_prefix.py")
    assert all(caller != "legal_now" for caller, _c, _l, _k in found), found


def test_a_synthetic_two_function_deadlock_is_detected(tmp_path):
    """A minimal fresh bug, so the detection cannot be an artifact of one file."""
    source = "import threading\n" \
             "_gate = threading.Lock()\n" \
             "def inner():\n" \
             "    with _gate:\n" \
             "        return 1\n" \
             "def outer():\n" \
             "    with _gate:\n" \
             "        return inner()\n"
    path = tmp_path / "bug_module.py"
    path.write_text(source)
    found = findings(path.read_text(), str(path))
    assert found == [("outer", "inner", "_gate", "Lock")], found


def test_a_reentrant_lock_is_not_reported():
    """`RLock` may be re-acquired by its owner: reporting it would be noise."""
    source = "import threading\n" \
             "_gate = threading.RLock()\n" \
             "def inner():\n" \
             "    with _gate:\n" \
             "        return 1\n" \
             "def outer():\n" \
             "    with _gate:\n" \
             "        return inner()\n"
    assert findings(source) == []


def test_the_locks_that_matter_are_still_plain_locks():
    """If someone silences the invariant by turning every `Lock` into an `RLock`,
    this test says so: the fix for the deadlock was the locked/unlocked split, not a
    reentrant lock hiding the same re-entry."""
    disk_read = (PLUGIN / "disk_read.py").read_text()
    assert "_state_lock = threading.Lock()" in disk_read
    tree = ast.parse(disk_read)
    assert module_locks(tree)["_state_lock"] == "Lock"
    assert "_give_buffer_locked" in disk_read and "_give_buffer(" in disk_read


def test_every_module_level_lock_in_the_package_is_covered():
    """The sweep is not vacuous: the package really does contain module-level locks."""
    names = set()
    for path in sorted(PLUGIN.glob("*.py")):
        names.update(module_locks(ast.parse(path.read_text(), str(path))))
    assert names, "no module-level locks found: the invariant test would prove nothing"
    assert "_state_lock" in names or "_lock" in {n for n in names}, names


@pytest.mark.parametrize("path", sorted(PLUGIN.glob("*.py")),
                         ids=lambda p: p.name)
def test_each_module_parses_for_the_checker(path):
    """A module the checker cannot parse would be silently skipped by the sweep."""
    ast.parse(path.read_text(), str(path))
