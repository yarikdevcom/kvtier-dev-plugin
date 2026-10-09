"""`plugin_version`: which CODE a process loaded, published in all three surfaces.

THE DEFECT THIS PINS (2026-10-09). An instrument printed `plugin=0.2.49` while the
engine had been running since before 0.2.49 was deployed, because it read
`importlib.metadata` - the version of the distribution INSTALLED ON DISK, which is
what the NEXT process to import this package will load. A long-running engine
keeps the code it imported at boot, so that reading can never show what is
running. Meanwhile every read-only route this plugin exposes was probed and NONE
of them reported the loaded version at all: `/snapshot` said `"version": 1` (a
schema integer), `/disk?op=status` said `"protocol_version": 2` (a protocol
number), `/version` was a 404, `/status` said `"plugin": "kvtier_dev"` (a name),
the rank files' `version` was `1` (schema again), and the boot banner carried no
version string. The running version was only INFERABLE from version-gated keys -
a rank file's `disk_gpu_shadow.pairs_cross_execution` (0.2.48+, a WORKER section: the
snapshot carries no `disk_gpu_shadow` at all) and `writer.heartbeat` (0.2.47+) -
which is an inference, and an inference you have to remember to make is one you
will eventually skip.

WHAT IS NOW PUBLISHED, and the rule that keeps it from becoming the new confusion:

    `version`           SCHEMA version. Unchanged in value AND meaning: a parser
                        pins it to ask "can I read this document?"
    `plugin_version`    the version of the CODE the publishing process imported.
                        A reader pins it to ask "which build is running?"

THREE PROCESSES, THREE PUBLISHERS, NEVER ONE'S VALUE READ AS ANOTHER'S: `/status`
is answered by the API-server process, `/snapshot` is BUILT on the EngineCore loop
thread and only forwarded by the API server, and the rank file is written by a
WORKER that answers no route. Each publishes its own process's loaded version,
labelled by `plugin_version_scope`; `test_the_status_route_answers_over_http_...`
proves a route substitutes another process's value in neither direction.

Every assertion here compares against `kvtier_dev_plugin.__version__` or a
runtime-patched stand-in for it - never against a literal. The
`test_*_follows_a_runtime_change_*` tests are the ones that make a copied literal
impossible.
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import fakes_debug as fake
import pytest
from test_routes import (
    SNAPSHOT,
    STATUS,
    FakeAsyncLLM,
    FakeEngineCoreClient,
    build_app,
    client_for,
)
from test_snapshot import RaisingEngine, _NullHolder, make_engine
from test_worker_diag import handler, installed_worker, metadata, rank_document

import kvtier_dev_plugin
from kvtier_dev_plugin import (
    config,
    disk_store,
    endpoint,
    guard_mode,
    loaded_version,
    snapshot,
    worker_diag,
)

CANARY = "0.0.0-installed-metadata-canary"
PROBE = "9.9.9-runtime-probe"

CANARY = "0.0.0-installed-metadata-canary"
PROBE = "9.9.9-runtime-probe"

# --- fixtures: the same two `tests/test_worker_diag.py` drives its writer with
# (the fork's worker class, and a diagnostics directory that turns the writer on).
# Defined here rather than imported: a fixture imported under its own name is a
# name every test signature then shadows.

@pytest.fixture
def worker_class():
    """The fork's worker class, exec'd under the fork's module name."""
    cls, _module = fake.fork_class(worker_diag.TARGET_MODULE,
                                   worker_diag.TARGET_CLASS,
                                   fake.WORKER_SOURCE, register=True)
    yield cls
    sys.modules.pop(worker_diag.TARGET_MODULE, None)


@pytest.fixture
def writes_to(tmp_path, monkeypatch):
    """A diagnostics directory plus the env that turns the writer on."""
    directory = tmp_path / "kv-tier-diagnostics"
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    monkeypatch.setenv(config.ENV_DIAG_MIN_INTERVAL, "0")
    monkeypatch.setenv(config.ENV_DIAG_INTERVAL, "1")
    return directory


# --- helpers ----------------------------------------------------------------

def current_version() -> str:
    return kvtier_dev_plugin.__version__


def _strings(node):
    """Every string in a JSON-shaped payload, keys included."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from _strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item)
    elif isinstance(node, str):
        yield node


def assert_no_canary(payload) -> None:
    assert not [text for text in _strings(payload) if CANARY in text], (
        "a published payload carries the installed-distribution metadata's answer "
        f"({CANARY}); the version fields must come from __version__")


def snapshot_states():
    """EVERY state a snapshot document can be in, as (name, engine, arm) triples.

    Enumerated from the states `tests/test_snapshot.py` and
    `tests/test_containment.py` already exercise, not invented here: a tier that is
    present, a tier that is absent (no scheduler at all), guard mode armed and
    disarmed, and the containment path where the build itself raised and the
    document is hand-built from formatted strings.
    """
    return [
        pytest.param("tier_present", make_engine(), False, id="tier_present"),
        pytest.param("tier_absent", fake.engine(None), False, id="tier_absent"),
        pytest.param("guard_armed", make_engine(), True, id="guard_armed"),
        pytest.param("guard_disarmed", make_engine(), False,
                     id="guard_disarmed"),
        pytest.param("degraded_containment", RaisingEngine(), False,
                     id="degraded_containment"),
    ]


def build_snapshot(engine) -> dict:
    return snapshot.snapshot_for_engine(engine, _NullHolder())


def build_status_payload() -> dict:
    return endpoint.status_payload()


def write_rank_document(worker_class, writes_to) -> dict:
    """One real rank file, written by the real step path."""
    worker = installed_worker(worker_class, writes_to)
    worker.get_finished(set())
    return rank_document(writes_to)


def test_the_published_value_follows_the_loaded_code_when_the_install_differs():
    """The 2026-10-09 shape, live in this checkout: a wheel installed AFTER the
    process booted makes `importlib.metadata` say a version nothing is running.
    Whenever installed and loaded disagree, the payload names the loaded one."""
    installed = importlib.metadata.version("kvtier-dev-plugin")
    payload = build_status_payload()
    assert payload["plugin_version"] == kvtier_dev_plugin.__version__
    if installed != kvtier_dev_plugin.__version__:
        assert payload["plugin_version"] != installed, (
            "the payload echoed the installed distribution while the install and "
            "the loaded code disagree - the exact misreport this release exists "
            "to make impossible")


def test_the_version_is_one_fact_written_in_three_places():
    """`__version__` is the source of truth for the PUBLISHED field; pyproject and
    uv.lock must still agree with it, because a release that bumps one of the
    three and not the others re-creates the blindness this whole change is about.
    Extends `tests/test_in_tree_standdown.py::test_the_package_version_matches_the_
    build_metadata` (which pins pyproject) with the third place a bump must land.
    (Only the project's own `[[package]]` entry in uv.lock is read: a dependency's
    version is a different package's fact.)"""
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text().splitlines()
    declared = [line.split('"')[1] for line in pyproject
                if line.startswith("version = ")]
    assert declared, "pyproject.toml declares no version"
    lock = (root / "uv.lock").read_text().splitlines()
    locked = [lock[index + 1].split('"')[1]
              for index, line in enumerate(lock)
              if line == 'name = "kvtier-dev-plugin"']
    assert locked, "uv.lock has no kvtier-dev-plugin entry"
    assert kvtier_dev_plugin.__version__ == declared[0] == locked[0], (
        f"__version__={kvtier_dev_plugin.__version__} "
        f"pyproject={declared[0]} uv.lock={locked[0]}")


# --- the accessor itself ----------------------------------------------------

def test_the_accessor_reads_the_package_object():
    assert loaded_version.loaded_version() == current_version()
    assert loaded_version.loaded_version_reason() == ""


def test_the_accessor_follows_a_runtime_change_to_the_package_version():
    """The reason this is a function and not a constant: `__version__` is read when
    the payload is built, so no payload can be holding a copied literal."""
    original = kvtier_dev_plugin.__version__
    kvtier_dev_plugin.__version__ = PROBE
    try:
        assert loaded_version.loaded_version() == PROBE
        assert loaded_version.fields()["plugin_version"] == PROBE
    finally:
        kvtier_dev_plugin.__version__ = original
    assert loaded_version.loaded_version() == current_version()


@pytest.mark.parametrize("broken", [
    pytest.param(None, id="absent"),
    pytest.param(42, id="not_a_string"),
    pytest.param("", id="blank"),
    pytest.param("   ", id="whitespace_only"),
    pytest.param("9" * 65, id="over_the_cap"),
])
def test_an_unusable_version_is_refused_with_a_reason_never_a_guess(broken,
                                                                    monkeypatch):
    monkeypatch.setattr(kvtier_dev_plugin, "__version__", broken, raising=False)
    fields = loaded_version.fields()
    assert fields[loaded_version.PLUGIN_VERSION_FIELD] is None, broken
    reason = fields[loaded_version.PLUGIN_VERSION_REASON_FIELD]
    assert isinstance(reason, str) and reason, (
        "a process that cannot know its version must say why, not publish a "
        "plausible default")


def test_the_accessor_reads_the_package_object_out_of_sys_modules():
    """The lookup IS `sys.modules["kvtier_dev_plugin"]`, which is why a runtime
    change to the package object is visible: this package keeps no second copy of
    the value anywhere."""
    assert loaded_version._package_module() is kvtier_dev_plugin


def test_a_process_without_the_package_object_says_so(monkeypatch):
    """The one state where the answer cannot exist: no package object in this
    process. Refused with a reason, never with a default."""
    monkeypatch.setattr(loaded_version, "_package_module", lambda: None)
    fields = loaded_version.fields()
    assert fields[loaded_version.PLUGIN_VERSION_FIELD] is None
    assert "sys.modules" in fields[loaded_version.PLUGIN_VERSION_REASON_FIELD]


def test_the_key_set_is_fixed_whatever_the_answer_is():
    assert set(loaded_version.fields()) == set(loaded_version.VERSION_FIELDS)
    assert loaded_version.VERSION_FIELDS == (
        "plugin_version", "plugin_version_reason", "plugin_version_scope")


# --- the installed metadata must never be the answer ------------------------

def test_no_surface_reports_the_installed_distribution_metadata(monkeypatch):
    """The exact defect, pinned twice over: no payload may CONTAIN the installed
    metadata's answer, and building a payload may not even CALL for it."""
    calls = []

    def poisoned(name):
        calls.append(name)
        return CANARY

    monkeypatch.setattr(importlib.metadata, "version", poisoned)
    assert importlib.metadata.version("kvtier-dev-plugin") == CANARY
    calls.clear()                             # the probe above was the poison
    payloads = [build_snapshot(make_engine()), build_status_payload(),
                worker_diag.build_document(handler(), "PP0_TP0", "env")]
    for payload in payloads:
        assert_no_canary(payload)
    assert calls == [], (
        "a payload builder consulted importlib.metadata; that answers for the NEXT "
        "boot, which is the whole defect this release closed")
    for payload in payloads:
        assert payload["plugin_version"] == current_version()


# --- surface 1: the snapshot document ---------------------------------------

@pytest.mark.parametrize("name,engine,armed", snapshot_states())
def test_the_snapshot_publishes_the_loaded_version_in_every_state(name, engine,
                                                                  armed):
    guard_mode.set_guard_mode(armed, engine)
    payload = build_snapshot(engine)
    assert payload["guard_mode"] is armed, f"{name} is not the state it names"
    assert payload["guard_mode"] is armed, f"{name} is not the state it names"
    for field in loaded_version.VERSION_FIELDS:
        assert field in payload, f"{field} missing from the {name} snapshot"
    assert payload["plugin_version"] == current_version()
    assert payload["plugin_version_reason"] is None
    assert "EngineCore" in payload["plugin_version_scope"]
    # AND the schema meaning is untouched: `version` is still PROTOCOL_VERSION, and
    # a reader that pins it keeps pinning 1.
    assert payload["version"] == snapshot.PROTOCOL_VERSION == 1


@pytest.mark.parametrize("name,engine,_armed", snapshot_states())
def test_the_snapshot_version_follows_a_runtime_change_and_the_schema_does_not(
        name, engine, _armed, monkeypatch):
    monkeypatch.setattr(kvtier_dev_plugin, "__version__", PROBE)
    payload = build_snapshot(engine)
    assert payload["plugin_version"] == PROBE, name
    assert payload["version"] == snapshot.PROTOCOL_VERSION == 1


def test_the_snapshot_fallback_publishes_the_same_keys_as_the_helper(monkeypatch):
    """The containment path spells the three keys literally; this is what stops
    that literal from drifting away from the helper's."""
    def explode(_scope=None):
        raise RuntimeError("the version module is gone")

    monkeypatch.setattr(loaded_version, "fields", explode)
    fields = snapshot.loaded_version_fields()
    assert set(fields) == set(loaded_version.VERSION_FIELDS)
    assert fields["plugin_version"] is None
    assert "did not build" in fields["plugin_version_reason"]
    payload = build_snapshot(make_engine())
    assert payload["plugin_version"] is None
    assert payload["version"] == snapshot.PROTOCOL_VERSION


def test_the_snapshot_version_is_the_engines_not_the_api_servers():
    """`plugin_version_scope` is the payload's own statement of WHICH process it
    describes, because the three surfaces live in three different processes."""
    scope = build_snapshot(make_engine())["plugin_version_scope"]
    assert "EngineCore" in scope and "not the API server" in scope


# --- surface 2: the no-token /status route ----------------------------------

def status_states():
    """EVERY state the `/status` payload can be in, as (name, builder) pairs.

    Enumerated from the states `tests/test_routes.py`, `tests/test_blockdump.py`
    and `tests/test_disk_fill_block.py` already put it in: nothing has been
    mirrored from the engine yet, an engine answer HAS been mirrored, the raw-byte
    dump's arm has been observed, and the disk-fill block failed to build
    (degraded). The inert posture is a separate test below, because it is the
    route that goes 503, not the payload.
    """
    def mirrored():
        return endpoint.status_payload({
            "guard_mode": True, "trace_armed": False,
            "lookup_ledger_armed": False, "blockdump_armed": False,
            "source": "snapshot read", "epoch": "aabbccdd",
            "observed_at": 1.0, "age_seconds": 2.0,
            "note": "mirrored in this test"})

    def blockdump_armed():
        return endpoint.status_payload({"blockdump_armed": True})

    def fill_degraded(monkeypatch):
        def explode():
            raise RuntimeError("the rollup is broken")

        monkeypatch.setattr(disk_store, "disk_fill", explode)
        return endpoint.status_payload()

    return [
        pytest.param("never_observed", lambda monkeypatch: build_status_payload(),
                     id="never_observed"),
        pytest.param("mirrored", lambda monkeypatch: mirrored(), id="mirrored"),
        pytest.param("blockdump_armed", lambda monkeypatch: blockdump_armed(),
                     id="blockdump_armed"),
        pytest.param("fill_degraded", fill_degraded, id="fill_degraded"),
    ]


@pytest.mark.parametrize("name,build", status_states())
def test_the_status_route_publishes_the_loaded_version_in_every_state(
        name, build, monkeypatch, enabled):
    payload = build(monkeypatch)
    for field in loaded_version.VERSION_FIELDS:
        assert field in payload, f"{field} missing from the {name} status payload"
    assert payload["plugin_version"] == current_version()
    assert payload["plugin_version_reason"] is None
    assert "API-server" in payload["plugin_version_scope"]
    assert payload["plugin"] == config.PLUGIN_NAME          # the NAME, unchanged


@pytest.mark.parametrize("name,build", status_states())
def test_the_status_version_follows_a_runtime_change(name, build, monkeypatch,
                                                     enabled):
    monkeypatch.setattr(kvtier_dev_plugin, "__version__", PROBE)
    payload = build(monkeypatch)
    assert payload["plugin_version"] == PROBE, name


def test_an_inert_plugin_still_publishes_its_version():
    """The route answers 503 when the plugin is inert, but the payload builder -
    the thing a test, a log line or a future route reaches - still says which code
    is loaded, because that question is not gated on being enabled."""
    payload = build_status_payload()
    assert payload["enabled"] is False
    assert payload["plugin_version"] == current_version()


def test_the_status_route_answers_over_http_with_its_own_process_version(
        enabled):
    """A route must not forward, and must not substitute. The fake engine answers a
    UTILITY call with a DIFFERENT version string: `/status` never asks it, and
    `/snapshot` forwards it untouched - so the two answers CAN disagree, which is
    how a reader catches "the package was upgraded, one process restarted, the
    other did not" instead of being told one confident number."""
    engine = FakeEngineCoreClient(result={"version": 1,
                                          "plugin_version": "0.2.49-in-the-engine"})
    client = client_for(build_app(FakeAsyncLLM(engine)))
    status = client.get(STATUS).json()
    assert status["plugin_version"] == current_version()
    body = client.get(SNAPSHOT).json()
    assert body["plugin_version"] == "0.2.49-in-the-engine"
    assert status["plugin_version"] != body["plugin_version"]


def test_the_status_payload_carries_a_version_where_none_existed_before(enabled):
    """Before 0.2.50 no field in `/status` carried a version at all; `plugin` is a
    NAME. Both are present now and they mean different things."""
    payload = build_status_payload()
    assert payload["plugin"] == "kvtier_dev"
    assert payload["plugin_version"] == current_version()
    assert payload["plugin_version"] != payload["plugin"]


# --- surface 3: the per-rank diagnostics file -------------------------------

def test_the_rank_file_publishes_the_loaded_version_of_the_worker_that_wrote_it(
        worker_class, writes_to):
    document = write_rank_document(worker_class, writes_to)
    for field in loaded_version.VERSION_FIELDS:
        assert field in document, f"{field} missing from the rank file"
    assert document["plugin_version"] == current_version()
    assert document["plugin_version_reason"] is None
    assert "worker process" in document["plugin_version_scope"]
    assert "PP0_TP0" in document["plugin_version_scope"]
    # ... and the file's own schema key is untouched, which is what its readers
    # (the proxy host's reader AND `disk_store`'s rollup) gate on.
    assert document["version"] == worker_diag.SCHEMA_VERSION == 1
    # `writer.plugin` is a TAG whose trailing "/1" reads like a version. Unchanged,
    # and it is NOT the build number.
    assert document["writer"]["plugin"] == worker_diag.WRITER_TAG == (
        "kvtier-dev-plugin/1")


def test_the_rank_file_version_follows_a_runtime_change(worker_class, writes_to,
                                                        monkeypatch):
    monkeypatch.setattr(kvtier_dev_plugin, "__version__", PROBE)
    assert write_rank_document(worker_class, writes_to)["plugin_version"] == PROBE


def test_the_rank_document_carries_the_version_keys_in_every_state():
    """The states a rank document is built in, from `tests/test_worker_diag.py`:
    plain, with a store in flight, every allowed section contributed, a section
    REFUSED for size, a section that RAISED, and a handler that answers none of the
    arena questions (every reading null, the degraded shape)."""
    nothing = handler(cpu_kv_caches=None, num_cpu_blocks=None,
                      cpu_capacity_bytes=None)
    cases = [
        ("plain", handler(), None),
        ("store_in_flight",
         handler(_connector_metadata=metadata(stores=3, loads=1)), None),
        ("all_sections", handler(), "sections"),
        ("section_refused", handler(), "refused"),
        ("section_raised", handler(), "raised"),
        ("handler_answers_nothing", nothing, None),
    ]
    for name, subject, mode in cases:
        worker_diag.reset_state_for_tests()   # one clean seam per state
        if mode == "sections":
            for section in worker_diag.CONTRIBUTION_SECTIONS:
                assert worker_diag.register_contribution(
                    section, {"present": True}, owner=section)["registered"]
        elif mode == "refused":
            assert worker_diag.register_contribution("disk", {"blob": "x" * 9000},
                                                     owner="disk_store")[
                "registered"]
        elif mode == "raised":
            def explode():
                raise RuntimeError("the contributor blew up")

            assert worker_diag.register_contribution("disk", explode,
                                                     owner="disk_store")[
                "registered"]
        document = worker_diag.build_document(subject, "PP0_TP0", "env")
        for field in loaded_version.VERSION_FIELDS:
            assert field in document, f"{field} missing from the {name} document"
        assert document["plugin_version"] == current_version(), name
        assert document["plugin_version_reason"] is None, name
        assert document["version"] == worker_diag.SCHEMA_VERSION, name


def test_the_version_keys_survive_the_byte_ceiling_triage(worker_class, writes_to,
                                                          monkeypatch):
    """A trimmed file is a smaller file, never one that forgot who wrote it: the
    three version keys are in KEEP_KEYS precisely so the triage sheds them last."""
    worker_diag.install(worker_class)
    worker = handler()
    full = len(worker_diag._dump(worker_diag.build_document(worker, "PP0_TP0",
                                                            "env")))
    monkeypatch.setattr(worker_diag, "MAX_FILE_BYTES", full - 5)
    report = worker_diag.write_snapshot(worker, "PP0_TP0", "env")
    assert report["written"] is True
    document = rank_document(writes_to)
    assert document["writer"]["trimmed"] is True, "the triage never ran"
    for field in loaded_version.VERSION_FIELDS:
        assert field in document, f"{field} was shed by the byte ceiling"
    assert document["plugin_version"] == current_version()


def test_the_rank_file_stays_inside_its_caps_with_the_new_field_present():
    """Byte cap and section budget, with the new field in place and every allowed
    contribution filled to its own cap: the file must still be a file the reader
    parses, and the version keys must still be in it."""
    for name in worker_diag.CONTRIBUTION_SECTIONS:
        payload = {"blob": "x" * (worker_diag.MAX_CONTRIBUTION_BYTES - 40)}
        assert len(json.dumps(payload).encode()) \
            <= worker_diag.MAX_CONTRIBUTION_BYTES, name
        assert worker_diag.register_contribution(name, payload,
                                                 owner=name)["registered"]
    document = worker_diag.build_document(handler(), "PP0_TP0", "env")
    clipped = worker_diag._clip_document(document)
    raw = worker_diag._dump(clipped)
    assert len(raw) <= worker_diag.MAX_FILE_BYTES, (
        f"{len(raw)} bytes over the {worker_diag.MAX_FILE_BYTES} cap")
    # The exact top-level shape of a FULL house: every writer-owned key (the nine
    # the file always had, plus the three version keys) and every contributed
    # section - nothing shed, so the three new keys did not cost a section slot or
    # push a guest over the byte ceiling.
    writer_owned = {"version", "epoch", "ts", "role", "rank", "arena",
                    "counters", "writer", "error"}
    assert set(clipped) == writer_owned | set(loaded_version.VERSION_FIELDS) | set(
        worker_diag.CONTRIBUTION_SECTIONS)
    for field in loaded_version.VERSION_FIELDS:
        assert field in clipped, field
    assert clipped["plugin_version"] == current_version()
    # The section budget itself is untouched by this release.
    assert len(worker_diag.CONTRIBUTION_SECTIONS) == worker_diag.MAX_CONTRIBUTIONS


def test_the_version_keys_are_writer_owned_and_a_contributor_cannot_take_them():
    """The allow-list is the whole trust: a guest may add `disk`, never a
    writer-owned key. `plugin_version` is writer-owned, so it is not in the
    allow-list and a contribution by that name is refused."""
    for field in loaded_version.VERSION_FIELDS:
        assert field not in worker_diag.CONTRIBUTION_SECTIONS, field
        report = worker_diag.register_contribution(field, {})
        assert report["registered"] is False, field
        assert "allow-list" in report["reason"]
    # The keys the reader REQUIRES still lead KEEP_KEYS, and the version keys are
    # in it too: the byte ceiling sheds a guest's section before it sheds the
    # answer to "which build wrote this file".
    assert set(worker_diag.KEEP_KEYS) >= {"version", "epoch", "ts", "role",
                                          "rank", "arena", "counters"}
    for field in loaded_version.VERSION_FIELDS:
        assert field in worker_diag.KEEP_KEYS, field


def test_the_new_key_does_not_make_a_rank_file_unreadable_to_our_own_reader(
        worker_class, writes_to):
    """`disk_store` classifies a rank file it cannot trust, and `bad_version` is
    the verdict that would fire if `version` had been repurposed for the code
    version. A real file with a real `disk` section must still be `counted`."""
    worker_diag.register_contribution("disk", {"installed": True, "binds": 1},
                                      owner="disk_store")
    write_rank_document(worker_class, writes_to)
    path = writes_to / "rank-PP0_TP0.json"
    row, status, detail = disk_store._rank_file_rollup_row(path,
                                                           "rank-PP0_TP0.json")
    assert status == "counted", (status, detail)
    assert row["rank"] == "PP0_TP0"


# --- three processes, one field name, no substitution -----------------------

def test_the_three_surfaces_name_their_own_process_and_never_each_others():
    scopes = {
        "snapshot": build_snapshot(make_engine())["plugin_version_scope"],
        "status": build_status_payload()["plugin_version_scope"],
        "rank": worker_diag.build_document(handler(), "PP1_TP0",
                                           "env")["plugin_version_scope"],
    }
    assert "EngineCore" in scopes["snapshot"]
    assert "API-server" in scopes["status"]
    assert "worker process" in scopes["rank"] and "PP1_TP0" in scopes["rank"]
    assert len(set(scopes.values())) == 3, (
        "two surfaces describe the same process to a reader: the whole point of the "
        "scope field is that these are three different processes")


def test_the_source_string_names_where_the_value_came_from():
    """`importlib.metadata` is named in the payload, so the next reader does not
    reach for it again."""
    scope = build_status_payload()["plugin_version_scope"]
    assert "__version__" in scope
    assert "importlib.metadata" in scope


def test_the_version_path_pulls_in_nothing_heavy():
    """The read side and a worker's step path both call it, and `sys` is all it
    needs. (The package-wide AST test covers the import list; this covers the call
    path actually taken.)"""
    build_snapshot(make_engine())
    worker_diag.build_document(handler(), "PP0_TP0", "env")
    assert "torch" not in sys.modules
    assert "vllm" not in sys.modules
