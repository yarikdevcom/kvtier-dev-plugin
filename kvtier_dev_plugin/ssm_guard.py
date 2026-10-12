"""THE SSM BOOT GUARD: the tier refuses to install on a boot with an SSM-recovery flag.

WHAT THIS EXISTS BECAUSE OF
---------------------------
The tier persists KV block contents and later serves them as a *correctness claim*.
On a hybrid model some groups are not per-token KV at all: they hold recurrent (Mamba)
boundary state. ReplaySSM / RecoverSSM change when that state is authoritative - the
fork's own words for the mechanism (patch 0072, quoted in
`patches/drafts/DRAFT-0006-replayssm-connector-validation/patch.manifest`) are "a
READ-ONLY CHECKPOINT and a PER-TOKEN REPLAY RECORD; after sampling ONE COMMIT replays
the accepted tokens and WRITES THE RECURRENT STATE ONCE". So per step a live row has a
deferred commit that has not landed yet.

Nothing in this stack orders a boundary STORE against that commit. A boundary block
spilled during the window may therefore be the state as it stood BEFORE the commit that
advanced it - and the tier would hand those bytes back as an authoritative restore. The
engine's own reviewer put it exactly: "nothing orders Mamba boundary stores after the
RecoverSSM commit, so lifting the replayssm refusal would persist the wrong state".

Why that is a refuse-to-run bug and not a degrade bug: a wrong KV claim does not raise.
It returns a plausible continuation. This project has live evidence of that failure
class (`Register Register Register...` at `temperature=0`, 2026-10-06 00:08), which is
why "it booted and answered" is not a test and why the answer here is to not boot.

THE FIRST LINE OF DEFENCE, AND WHY THIS IS THE SECOND
-----------------------------------------------------
The engine already vetoes this combination:
`vllm/config/vllm.py::validate_mamba_cached_kernel` raises `ValueError("--use-replayssm
is incompatible with KV connectors (P/D disaggregation, KV cache offload)")`. That
clause sits OUTSIDE the `if self.cache_config.use_recoverssm:` branch, so it vetoes
plain ReplaySSM too, and `is_kv_transfer_instance` is true for our `kv_role: "kv_both"`
launcher - so it fires for us today. This guard exists because that clause is precisely
what `DRAFT-0006-replayssm-connector-validation` (a DO-NOT-APPLY draft) would relax. If
this module's ValueError is what an operator sees on a boot, the draft is in the tree.

THE THREE SPELLINGS
-------------------
The flag is spelled three ways across the forks this project holds, so all three names
are read defensively by attribute name and none is assumed to exist:

  * `use_replayssm`      - the fork that serves (`SlimServe/vllm/config/cache.py`) and
                           the same name in the v2.5 fork; the v2.5 engine-side veto
                           reads this one.
  * `use_recoverssm`     - named in the DRAFT-0006 manifest as the branch the KV clause
                           sits outside of; present in the v2.5 tree's `CacheConfig`.
  * `use_kda_recoverssm` - the v2.5 KDA field, and the one the WORKER-side sink reads:
                           `model_state.recoverssm = RecoverSSMState() if
                           cache_config.use_kda_recoverssm else None`
                           (`vllm/v1/worker/gpu/model_states/mamba_hybrid.py`).

THE SEMANTICS, ONE PER STATE (each is deliberate)
-------------------------------------------------
  * present AND truthy   -> raise `ValueError`. The tier does not install.
  * ABSENT               -> not SSM. No raise. This is the load-bearing direction: an
                           older fork where the feature does not exist must still boot,
                           so "I could not find it" must NEVER be read as "it is on".
                           Reading absence as true would brick every engine that has no
                           SSM at all, which is the opposite of a guard.
  * present and falsy    -> no raise. This is every normal boot.
  * READING IT RAISED, or no `cache_config` in reach -> do NOT raise our own error; log
                           ONE loud WARNING naming what could not be determined. This is
                           a DELIBERATE FAIL-OPEN, and its rationale is that the
                           engine's own `validate_mamba_cached_kernel` is the primary
                           defence here: a plugin that refuses to boot because ITS
                           OWN PROBE was confused invents a new failure mode (a boot
                           that cannot start for a reason that does not exist) while
                           the thing it is trying to detect is already vetoed one
                           layer up. Positive proof of the hazard still outranks an
                           unrelated probe failure: if one name reads truthy and
                           another reads raised, this REFUSES.

NO ENV OPT-OUT, ON PURPOSE
--------------------------
The remedy is "drop the flag, or do not install the tier". There is deliberately no
`VLLM_E2_TIER_*` name that lets such a boot through: an opt-out would reintroduce the
exact hazard this exists to close, and a plugin-side override would put the last line of
defence in the position of the first. If it is ever needed it arrives with a witness
from a live engine - not before. `VLLM_E2_TIER_DEV_BOOT_INERT` is NOT an opt-out and is
treated accordingly: an inert boot installs and stops there (no guard, no arming, see
`backport.boot_mode()`), so it cannot spill and the hazard is absent - the guard warns
in that case rather than refusing a boot whose tier does nothing.

WHERE THIS MUST BE CALLED, AND WHY IT IS NOT SILENCE
----------------------------------------------------
It raises, so its value depends entirely on the exception reaching the operator. The
tier's own install path is deliberately contained (`backport._after_connector_created`
runs inside `try/except Exception` because "nothing here may raise into
`create_connector`"), and `engine_side.register()` is wrapped the same way. Calling this
from either of those would make the refusal a no-op - a gate that cannot fail. It is
therefore called from the connector-creation hook WRAPPER
(`backport._make_create_connector_hook.create_connector`), OUTSIDE that contained region
and before `_after_connector_created` installs or arms anything, so the ValueError
propagates through `Scheduler.__init__` (no try) and `EngineCore.__init__` (no try) and
reaches the operator: in-process as this very exception, and in the background
EngineCore process through `run_engine_core`'s `except Exception`, which logs
`EngineCore failed to start.` and RE-RAISES.
`tests/test_ssm_boot_guard.py` proves the escape, and pins structurally that the call
site is not inside a `try`.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("kvtier_dev_plugin.ssm_guard")

# The flag names, read off `vllm_config.cache_config`, in the order reported. Every one
# is looked up by name with a sentinel default: see the docstring for why ABSENT must
# never read as true.
SSM_FLAG_ATTRS = ("use_replayssm", "use_recoverssm", "use_kda_recoverssm")

FLAG_NOTES = {
    "use_replayssm": "--use-replayssm (ReplaySSM Mamba2 decode kernel)",
    "use_recoverssm": "--use-recoverssm (RecoverSSM recurrent-state checkpointing)",
    "use_kda_recoverssm": "--use-kda-recoverssm (Kimi-K3 KDA RecoverSSM; the "
                          "worker-side `model_state.recoverssm` sink)",
}

# THE CLOSED VOCABULARY. `refused` is the only one that raises; the two
# cannot-determine verdicts warn instead (the deliberate fail-open above), and
# `refused-inert` is a refusal the inert posture made harmless.
VERDICT_CLEAR = "clear"
VERDICT_REFUSED = "refused"
VERDICT_REFUSED_INERT = "refused-inert"
VERDICT_INDETERMINATE = "indeterminate"
VERDICT_NO_CONFIG = "no-config"
VERDICTS = (VERDICT_CLEAR, VERDICT_REFUSED, VERDICT_REFUSED_INERT,
            VERDICT_INDETERMINATE, VERDICT_NO_CONFIG)

# NOT a probe verdict - `probe()` can never return it - and so deliberately absent from
# `VERDICTS`. It is the BOOT LINE's rendering of "no guard record exists for this boot"
# (the connector hook was never fitted, or the connector was not ours, so no probe ever
# ran), the same fact `load_failure_policy.VERDICT_NOT_RUN` and
# `lookup_ledger.VERDICT_NOT_RUN` carry for their tokens. Absence has to be SAYable, or
# a missing token is read as the satisfied case.
VERDICT_NOT_RUN = "not_run"

# The engine's own validator, named in the refusal so a reader can find the guard that
# was relaxed. Never a claim that it RAN - only that it normally would have.
ENGINE_VALIDATOR = "VllmConfig.validate_mamba_cached_kernel"
DRAFT_NAME = "DRAFT-0006-replayssm-connector-validation"

_MAX_PROBLEM_CHARS = 200


class _Missing:
    """Sentinel for "this fork has no such attribute" - never `None`, which a
    legitimately-`None` field could be confused with."""

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "<absent>"


_MISSING = _Missing()


def _clip(value) -> str:
    return str(value)[:_MAX_PROBLEM_CHARS]


def _empty() -> dict:
    """THE RECORD'S KEY SET, built once so every path returns the same shape. A
    missing key and a `None` are different facts to a reader, so there are no missing
    keys: `flag: None` means "no flag was found", not "we forgot to look"."""
    return {"verdict": VERDICT_CLEAR, "flag": None, "flag_note": None,
            "value": None, "cache_config": None, "config_source": None,
            "attributes_seen": {}, "attributes_absent": [], "problems": [],
            "inert": False, "reason": None}


def _cache_config_of(vllm_config) -> tuple:
    """The `cache_config` off the config the factory was handed. Never raises.

    Returns `(cache, kind, problem)`. `kind` keeps the two no-answer cases apart,
    because they carry different remedies: `absent` (no such attribute / it is None -
    this hook was handed nothing useful) and `raised` (the attribute exists but reading
    it exploded - a config class we cannot trust to read).
    """
    if vllm_config is None:
        return None, "absent", ("no vllm_config was handed to the connector-creation "
                                "hook, so the SSM flag cannot be read")
    try:
        cache = getattr(vllm_config, "cache_config", _MISSING)
    except Exception as exc:  # noqa: BLE001 - a property that raises is a no-answer
        from . import snapshot  # noqa: PLC0415 - sibling layer, lazy by idiom
        snapshot._error(exc, "ssm_guard read cache_config")
        return None, "raised", _clip(f"reading vllm_config.cache_config raised "
                                     f"{type(exc).__name__}: {exc}")
    if cache is _MISSING:
        return None, "absent", "vllm_config has no cache_config attribute"
    if cache is None:
        return None, "absent", "vllm_config.cache_config is None"
    return cache, "ok", None


def _read_flag(cache, name: str) -> tuple:
    """`(present, value, problem)` for one flag name. `present` is None = unknown."""
    try:
        value = getattr(cache, name, _MISSING)
    except Exception as exc:  # noqa: BLE001 - an unreadable flag is no-answer, not ON
        from . import snapshot  # noqa: PLC0415 - sibling layer, lazy by idiom
        snapshot._error(exc, f"ssm_guard read {name}")
        return None, None, _clip(f"reading cache_config.{name} raised "
                                 f"{type(exc).__name__}: {exc}")
    if value is _MISSING:
        # ABSENT. The load-bearing branch: an older fork without the feature, which is
        # NOT SSM and must boot. Never collapse this into "assume on".
        return False, None, None
    return True, value, None


def probe(vllm_config) -> dict:
    """Read the three names and say what this boot is. Never raises."""
    record = _empty()
    try:
        from . import config  # noqa: PLC0415 - sibling layer, lazy by idiom
        record["inert"] = bool(config.boot_inert())
    except Exception as exc:  # noqa: BLE001 - the inert posture is context, not a gate
        record["inert"] = False
        record["problems"] = [_clip(f"reading the inert posture raised "
                                    f"{type(exc).__name__}", 80)]

    cache, kind, problem = _cache_config_of(vllm_config)
    if kind != "ok":
        # No attribute reads are attempted: guessing about objects we could not resolve
        # is how a probe becomes a liar. Both kinds warn; neither raises (fail-open,
        # documented above).
        record["verdict"] = (VERDICT_INDETERMINATE if kind == "raised"
                             else VERDICT_NO_CONFIG)
        record["problems"] = ([problem] if problem else []) + record["problems"]
        record["reason"] = ("the SSM posture could not be determined, so this boot is "
                            "left to the engine's own "
                            f"{ENGINE_VALIDATOR}")
        return record

    record["cache_config"] = f"{type(cache).__module__}.{type(cache).__name__}"
    record["config_source"] = "vllm_config.cache_config"

    for name in SSM_FLAG_ATTRS:
        present, value, flag_problem = _read_flag(cache, name)
        if flag_problem:
            record["problems"] = (record["problems"] + [flag_problem])[:4]
            continue
        if not present:
            record["attributes_absent"].append(name)
            continue
        record["attributes_seen"][name] = bool(value)
        if value:
            # POSITIVE PROOF OF THE HAZARD, and it outranks any unreadable sibling.
            record["verdict"] = VERDICT_REFUSED
            record["flag"] = name
            record["flag_note"] = FLAG_NOTES.get(name, name)
            record["value"] = bool(value)
            record["reason"] = (f"cache_config.{name} reads {bool(value)!r}")
            return record

    if record["problems"]:
        record["verdict"] = VERDICT_INDETERMINATE
        record["reason"] = ("at least one SSM flag name could not be read, so the "
                            "posture is not established; this boot is left to the "
                            f"engine's own {ENGINE_VALIDATOR}")
        return record

    record["verdict"] = VERDICT_CLEAR
    record["reason"] = (f"none of {list(SSM_FLAG_ATTRS)} is set "
                        f"({len(record['attributes_absent'])} absent, "
                        f"{len(record['attributes_seen'])} present-and-false): this "
                        "boot has no SSM-recovery state for the tier to persist "
                        "incorrectly")
    return record


def refusal_message(record: dict) -> str:
    """The refusal text: WHAT we found, WHY it is fatal, WHO normally stops it, and the
    REMEDY. The engine-validator sentence is the one that tells a future reader that
    DRAFT-0006 is in the tree, so it is never dropped or shortened."""
    flag = record.get("flag") or "?"
    note = record.get("flag_note") or flag
    return (
        f"kvtier-dev-plugin: refusing to install the KV tier on this boot. "
        f"The engine was configured with an SSM-recovery flag: the attribute "
        f"cache_config.{flag} ({note}) reads True on "
        f"{record.get('cache_config') or 'the engine config'}.\n\n"
        f"WHY THIS IS FATAL. On a hybrid model some KV groups hold recurrent (Mamba) "
        f"boundary state, not per-token KV. Nothing in this stack orders a boundary "
        f"STORE against the RecoverSSM commit, so a boundary block this tier "
        f"spills may be the state as it stood BEFORE the commit that advanced it. "
        f"The tier serves "
        f"what it stored as a correctness claim, and a wrong KV claim does not raise - "
        f"it returns a plausible-looking continuation. Plausible garbage in front of "
        f"users is the one outcome worse than refusing to run, so this plugin refuses "
        f"to run instead.\n\n"
        f"WHO NORMALLY STOPS THIS. The engine's own {ENGINE_VALIDATOR} vetoes this "
        f"combination first (\"--use-replayssm is incompatible with KV connectors "
        f"(P/D disaggregation, KV cache offload)\"). Seeing THIS message therefore "
        f"means that guard was relaxed: the KV-connector clause has been edited out of "
        f"the tree, i.e. {DRAFT_NAME} (a DO-NOT-APPLY draft whose own manifest says "
        f"the question underneath it is unanswered) is in the tree. Read that manifest "
        f"before going further.\n\n"
        f"REMEDY. Drop the flag (--use-replayssm / --use-recoverssm / "
        f"--use-kda-recoverssm, whichever this boot was given), or do not install the "
        f"tier. There is deliberately no environment variable that lets this boot "
        f"through: an opt-out would reintroduce the hazard this guard exists to close.")


def boot_token(record) -> str:
    """The BOOT line's `SSM[what]` token, from THIS boot's own record.

    Bracketed and never `key=value`, for the reason `mamba_anchor.boot_token()` gives in
    full: the restart verifier in the llm-api repo pulls its PASS criterion with
    `grep -oE "tier=[A-Za-z_]+" | tail -1`, and a token with no `=` can never become a
    candidate match. Read from the BOOT DICT (the `JOURNAL_DRAIN` / `LOOKUP_LEDGER` /
    `LOAD_POLICY` discipline), so the line describes the boot that produced it.

    WHY THE TOKEN EXISTS. The guard's SATISFIED case used to be a single `logger.info`,
    and a plugin INFO line on the connector-hook path does not reach
    `/tmp/vllm-serve.log` at all (measured on the live engine 2026-10-22: the hook's own
    INFO line had 0 occurrences in a log carrying 2,775 plugin lines, and `SSM BOOT
    GUARD` had 0 too). So an operator reading a boot could not tell "the guard ran and
    found no SSM" from "the guard is not installed" - a safety control invisible when
    satisfied. The BOOT line is a WARNING and DOES land, so the verdict rides there. It
    is REPORTING, never a refusal: this token never joins `refusals`.

    `not_run` is the honest rendering of "this boot produced no record", exactly as
    `LOAD_POLICY[not_run]` means it. A record whose verdict this module cannot name
    renders `indeterminate` and NEVER `clear` - a verdict we cannot read is literally
    "the posture is not established", while `refused` would claim an action this record
    does not show happened.
    """
    try:
        if not isinstance(record, dict):
            return f"SSM[{VERDICT_NOT_RUN}]"
        verdict = record.get("verdict")
        if verdict == VERDICT_NOT_RUN:
            return f"SSM[{VERDICT_NOT_RUN}]"
        if verdict not in VERDICTS:
            return f"SSM[{VERDICT_INDETERMINATE}]"
        return f"SSM[{verdict}]"
    except Exception:  # noqa: BLE001 - a token is never worth a lost boot line
        return f"SSM[{VERDICT_INDETERMINATE}]"


def boot_check(vllm_config) -> dict:
    """THE BOOT STEP. Raises `ValueError` on a refusal; never raises otherwise.

    Called by `backport._make_create_connector_hook.create_connector` for a tier
    connector, OUTSIDE the contained `_after_connector_created` region, before anything
    is installed or armed - see the module docstring for why anywhere else would make
    this a gate that cannot fail.
    """
    record = probe(vllm_config)
    verdict = record["verdict"]
    if verdict == VERDICT_REFUSED:
        if record["inert"]:
            # An inert boot installs and stops there: no guard, no arming, so it spills
            # nothing and the hazard is absent. Say so loudly rather than refusing a
            # boot whose tier provably does nothing.
            record["verdict"] = VERDICT_REFUSED_INERT
            logger.warning("kvtier-dev-plugin: SSM BOOT GUARD: %s is set on this boot "
                           "and the tier WOULD refuse it - not refusing because "
                           "VLLM_E2_TIER_DEV_BOOT_INERT makes this an install-only "
                           "boot that arms nothing (it spills no KV, so there is no "
                           "wrong claim to make). Turn the inert switch off and the "
                           "boot refuses.", record["flag"])
            return record
        # LOUD IN THE LOG FIRST: the traceback may be wrapped (a worker-role refusal
        # comes back through the executor as a worker error), this line will not be.
        logger.warning("kvtier-dev-plugin: SSM BOOT GUARD: REFUSED - %s",
                       refusal_message(record))
        raise ValueError(refusal_message(record))
    if verdict in (VERDICT_INDETERMINATE, VERDICT_NO_CONFIG):
        # The deliberate fail-open, made as loud as a refusal so nobody reads it as a
        # pass: we could not tell, and we are NOT the primary defence.
        logger.warning("kvtier-dev-plugin: SSM BOOT GUARD: NOT ESTABLISHED - %s; the "
                       "tier installs anyway and relies on the engine's own %s. "
                       "Problems: %s", record["reason"], ENGINE_VALIDATOR,
                       "; ".join(record["problems"]) or "none")
        return record
    logger.info("kvtier-dev-plugin: SSM BOOT GUARD: %s", record["reason"])
    return record
