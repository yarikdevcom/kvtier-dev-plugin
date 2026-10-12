"""THE ENGINE'S KV LOAD-FAILURE POLICY: who decides it, and what this boot did.

THE FACT THIS EXISTS BECAUSE OF
-------------------------------
`KVTransferConfig.kv_load_failure_policy` is the fork's switch for what a KV load
that cannot be served costs the user. It is declared
`Literal["recompute", "fail"] = "fail"` (`vllm/config/kv_transfer.py:69`) and read
exactly once, in `Scheduler.__init__`:

    self.connector = KVConnectorFactory.create_connector(...)  # :140
    kv_load_failure_policy = kv_transfer_config.kv_load_failure_policy   # :147
    self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"  # :148

WITH THE SHIPPED DEFAULT, `fail`, a restore that cannot land FAILS the request with
an error finish reason (`scheduler.py:1928`, and `should_fail` in
`_handle_invalid_blocks` at `:2835`, which also skips the `evict_blocks` call the
recompute path deliberately makes). For the host-RAM tier above this one that is
nearly invisible - a RAM miss is a re-prefill either way. For the SSD tier beneath it
a read can fail for reasons that have nothing to do with correctness: the file was
capacity-evicted between the scheduler's lookup and the worker's read, it is
unreadable, its geometry digest no longer matches. Under `fail`, each of those
becomes a user-visible error instead of a slower, correct recompute. The upstream
owner's own words (2026-10-22): "the default kv_load_failure_policy is \"fail\", not
\"recompute\", so a failed restore kills the request unless that's set."

Until this module the field was the operator's job by hand, in the connector JSON -
on this deployment `KV_LOAD_FAILURE_POLICY=recompute` in the launcher, which emits it
into `--kv-transfer-config`. One unset environment variable in a launcher therefore
made every disk-read miss a 5xx, and nothing in the boot said so.

THE SEAM, AND WHY IT IS THE ONLY ONE
------------------------------------
`Scheduler.__init__` calls `KVConnectorFactory.create_connector` at
`scheduler.py:140`, EIGHT LINES BEFORE it reads the field at `:147`, so the
connector-creation hook (`backport._after_connector_created`) can set it while the
write still matters. There is no later seam: the scheduler latches the bool once at
construction, so a write after the connector returns from the factory call and
before `create_connector` returns is the last moment the field is still a choice. The
hook runs in the EngineCore process, which is the process that reads the field.

WHAT THIS MODULE MAY NEVER DO
-----------------------------
It may never override a choice the operator made. Recompute-versus-fail is a product
decision about whether a user gets a slow correct answer or a fast error, and a
plugin that silently reversed it is a plugin that has to be reverted. So the write
happens ONLY on a positive answer to "did nobody name this field?" - see WITNESS for
what counts as an answer, and `decide()` for what happens when there is none. It
touches no other field, imports no vllm module at module scope, and never raises.

WITNESS: HOW "EXPLICITLY SET" IS DECIDED, AND WHAT IT CANNOT SEE
----------------------------------------------------------------
`KVTransferConfig` is a PYDANTIC dataclass (`vllm/config/utils.py`'s `@config`), and a
pydantic dataclass does NOT record which fields its constructor was given: unlike a
`BaseModel` it has no `model_fields_set` / `__pydantic_fields_set__` (verified on
this host against pydantic 2.13: the attribute does not exist on an instance built
either way, and `TypeAdapter(...).validate_json()` does not add it). So the object in
hand can NEVER tell "absent, so the default was used" from "written out as `\"fail\"`
by hand", and the EngineCore process receives a PICKLED copy, which adds nothing.
Anyone who tells you they can read that off the config object is reading a guess.

The witness that CAN see the difference is the operator's own input: the JSON string
behind `--kv-transfer-config`. `EngineArgs` parses that flag with
`TypeAdapter(KVTransferConfig).validate_json(value)` (`arg_utils.py`, the dataclass
branch of `_compute_kwargs`), so a key present in that JSON is a key the operator
wrote and a key absent from it is a key the engine defaulted. `operator_witness()`
reads the raw string out of `sys.argv` and parses it itself - never the validated
object, which has already lost the distinction. Two properties of that witness,
stated because they bound what the verdict is allowed to claim:

  * IT SURVIVES THE PROCESS BOUNDARY. The scheduler's `create_connector` runs in the
    EngineCore process, and `multiprocessing.spawn` restores `sys.argv` from the
    parent's (`spawn.get_preparation_data` records `sys_argv`, the child assigns it
    back); a fork inherits it. Verified on this host (python 3.14) with a spawn child
    printing the parent's argv.
  * IT ONLY EXISTS FOR A CLI BOOT. A config built in Python (`KVTransferConfig(...)`,
    or the `MultiConnector` path that constructs one per child) leaves no
    `--kv-transfer-config` token to read. There the verdict is
    `unresolved`-not-set-and-untouched: the honest reading of a `fail` this process
    cannot prove was not chosen is that it WAS chosen. `VLLM_E2_TIER_DISK_LOAD_
    FAILURE_POLICY_DEFAULT=1` is the operator's answer to that case - it asserts "I did
    not set it, write it anyway", and it is the only way to reach it.

The SECONDARY witness, used only when there is no argv witness at all, is the value
itself compared against the ENGINE'S OWN default read from the field declaration at
runtime (`engine_default()`), never a hardcoded literal: a value that is not the
default cannot be an accident, so it is reported as operator-chosen and left alone.
That witness can say "someone chose this"; it can never say "nobody chose this",
which is why it never authorises a write.
"""

from __future__ import annotations

import json
import logging
import sys

from . import config

logger = logging.getLogger("kvtier_dev_plugin.load_failure_policy")

# The engine's field, and its two literals. The literals are NOT the default: the
# default is READ (`engine_default()`), because a plugin that hardcodes it would
# report a future engine that flipped the default as an operator choice.
POLICY_FIELD = "kv_load_failure_policy"
POLICY_RECOMPUTE = "recompute"
POLICY_FAIL = "fail"

KV_TRANSFER_MODULE = "vllm.config.kv_transfer"
KV_TRANSFER_CLASS = "KVTransferConfig"

# The CLI flag the field can only ever arrive on. argparse accepts the dash form; the
# underscore form is what `EngineArgs`' field name would produce, so both are read -
# reading only one would answer "nobody named it" about a boot that named it.
CONNECTOR_FLAGS = ("--kv-transfer-config", "--kv_transfer_config")

MAX_REASON_CHARS = 400
MAX_PROBLEM_ITEMS = 4
MAX_PROBLEM_CHARS = 160
MAX_RAW_CHARS = 80

# THE CLOSED VOCABULARY OF DECISIONS. Every one of these is reachable, every one is
# printed on the boot line, and `decide()` returns exactly one per call.
VERDICT_PLUGIN_SET = "plugin-set"          # we wrote `recompute` and READ IT BACK
VERDICT_OPERATOR = "operator"              # the operator's own JSON named the field
VERDICT_OPERATOR_VALUE = "operator-value"  # the value is not the engine's default
VERDICT_ALREADY = "engine-already"         # already `recompute`; nothing to write
VERDICT_UNRESOLVED = "unresolved"          # cannot prove nobody chose it: untouched
VERDICT_OPTED_OUT = "opted-out"            # the env name said do not touch it
VERDICT_INERT = "inert"                    # an inert boot changes nothing at all
VERDICT_NO_CONFIG = "no-config"            # no KVTransferConfig in reach
VERDICT_WRITE_REFUSED = "write-refused"    # the write, or the read back, failed
VERDICT_NOT_RUN = "not_run"                # this boot never reached the step
VERDICTS = (VERDICT_PLUGIN_SET, VERDICT_OPERATOR, VERDICT_OPERATOR_VALUE,
            VERDICT_ALREADY, VERDICT_UNRESOLVED, VERDICT_OPTED_OUT, VERDICT_INERT,
            VERDICT_NO_CONFIG, VERDICT_WRITE_REFUSED, VERDICT_NOT_RUN)

# WHAT THE ENGINE WILL DO, in the words the boot line prints beside the decision. Two
# values only, because that is the engine's `Literal`; anything else is `unknown`,
# which is a fact about what we could read and not a third policy.
VALUE_TOKENS = {POLICY_RECOMPUTE: "recompute", POLICY_FAIL: "fail"}


def _clip(value, limit: int) -> str:
    return str(value)[:limit]


def _empty() -> dict:
    """THE VERDICT'S KEY SET, built once so every path - including a path that raises
    in the first line - returns the same shape. `None` is a distinct answer from a
    value: `before: None` means there was no object to read, not that the field said
    nothing. The token is rendered too, so a verdict that was never decided still says
    `not_run` rather than an empty string a reader would have to interpret.
    """
    out = {"decision": VERDICT_NOT_RUN, "field": POLICY_FIELD,
           "before": None, "after": None, "applied": False,
           "engine_default": None, "engine_default_source": None,
           "witness": None, "operator_names_field": None, "operator_value": None,
           "forced": False, "env_var": config.ENV_DISK_LOAD_FAILURE_POLICY_DEFAULT,
           "env_raw": None, "reason": None, "problems": [], "token": None}
    out["token"] = boot_token(out)
    return out


def engine_default() -> dict:
    """THE ENGINE'S OWN DEFAULT for the field, read from its declaration.

    Never a hardcoded literal, for the reason in the module docstring: this number is
    the reference for "is the value we found an accident or a choice?", and an engine
    that flipped its default would otherwise be reported as an operator choice. Reads
    the class out of `sys.modules` first (the engine process has it imported by
    definition) and only then tries an import, so this never starts vllm in a process
    that has not. Fails CLOSED: `default: None` means "no reference", and `decide()`
    treats that as "cannot compare", which never authorises a write.
    """
    out = {"default": None, "source": "unreadable", "reason": None}
    try:
        import dataclasses  # noqa: PLC0415 - stdlib, only needed to read the field
        module = sys.modules.get(KV_TRANSFER_MODULE)
        if module is None:
            from importlib import import_module  # noqa: PLC0415 - see above
            module = import_module(KV_TRANSFER_MODULE)
        cls = getattr(module, KV_TRANSFER_CLASS, None)
        if cls is None or not dataclasses.is_dataclass(cls):
            out["reason"] = (f"{KV_TRANSFER_MODULE}.{KV_TRANSFER_CLASS} is not a "
                             "dataclass in this engine, so its field defaults are "
                             "not readable")
            return out
        for item in dataclasses.fields(cls):
            if item.name != POLICY_FIELD:
                continue
            default = item.default
            if default is dataclasses.MISSING:
                out["reason"] = (f"{KV_TRANSFER_CLASS}.{POLICY_FIELD} has no plain "
                                 "default (a factory or a required field), so there "
                                 "is no literal to compare against")
                return out
            out["default"] = str(default)
            out["source"] = "kv_transfer_config"
            return out
        out["reason"] = (f"{KV_TRANSFER_CLASS} has no field {POLICY_FIELD}: the fork "
                         "renamed or removed it")
        return out
    except Exception as exc:  # noqa: BLE001 - a missing default is a verdict
        from . import snapshot  # noqa: PLC0415 - sibling layer, lazy by idiom
        snapshot._error(exc, "load_failure_policy engine default")
        out["reason"] = _clip(f"reading the engine default raised "
                              f"{type(exc).__name__}: {exc}", MAX_REASON_CHARS)
        return out


def operator_witness(argv=None) -> dict:
    """DID THE OPERATOR'S OWN TEXT NAME THE FIELD? Read the CLI, not the config object.

    `argv` is a parameter so a test can hand this a command line; nothing here
    interprets the rest of it. Every `--kv-transfer-config` value found is parsed:
    a key in it is a key the operator wrote. `reliable` is the field the decision
    reads, and it is deliberately strict about the negative direction - claiming
    "nobody named it" on a command line whose JSON this process could not parse would
    be a guess dressed as a witness, which is the exact failure mode the boot line
    exists to prevent.
    """
    out = {"flag_seen": False, "values_scanned": 0, "names_field": None,
           "operator_value": None, "parse_failures": 0, "reliable": False,
           "source": None, "problems": []}
    try:
        args = list(sys.argv if argv is None else argv)
    except Exception as exc:  # noqa: BLE001 - an unreadable argv is no witness
        from . import snapshot  # noqa: PLC0415 - sibling layer, lazy by idiom
        snapshot._error(exc, "load_failure_policy argv read")
        out["problems"] = [_clip(f"argv unreadable: {type(exc).__name__}",
                                MAX_PROBLEM_CHARS)]
        out["source"] = "argv-unreadable"
        return out

    index = 0
    while index < len(args):
        token = str(args[index])
        index += 1
        value = None
        for flag in CONNECTOR_FLAGS:
            if token == flag:
                # The value is the NEXT token; a flag at the end of the line is a
                # command line we cannot price, so it counts as a parse failure.
                value = str(args[index]) if index < len(args) else ""
                index += 1
                break
            if token.startswith(flag + "="):
                value = token[len(flag) + 1:]
                break
        if value is None:
            continue
        out["flag_seen"] = True
        out["values_scanned"] += 1
        try:
            parsed = json.loads(value)
        except Exception:  # noqa: BLE001 - any parse error means the same: no witness
            out["parse_failures"] += 1
            continue
        if not isinstance(parsed, dict):
            out["parse_failures"] += 1
            continue
        if POLICY_FIELD in parsed:
            out["names_field"] = True
            out["operator_value"] = _clip(parsed.get(POLICY_FIELD), MAX_RAW_CHARS)
        elif out["names_field"] is None:
            # Only a NOTICING so far: a later value that names the field outranks it.
            out["names_field"] = False

    if not out["flag_seen"]:
        out["source"] = "no-cli-witness"
    elif out["parse_failures"]:
        out["source"] = "cli-unparseable"
        out["problems"] = [_clip(f"{out['parse_failures']} --kv-transfer-config "
                                 "value(s) did not parse, so the absence of "
                                 f"{POLICY_FIELD} cannot be established",
                                 MAX_PROBLEM_CHARS)]
    else:
        out["reliable"] = True
        out["source"] = "cli"
    return out


def kv_transfer_config_of(vllm_config):
    """The `kv_transfer_config` off whatever the factory was handed. Never raises."""
    try:
        return getattr(vllm_config, "kv_transfer_config", None)
    except Exception as exc:  # noqa: BLE001 - a config that lies is a missing config
        from . import snapshot  # noqa: PLC0415 - sibling layer, lazy by idiom
        snapshot._error(exc, "load_failure_policy read config")
        return None


def _value_of(kv_config):
    """The field's current value, or `None` when there is no field to read."""
    if kv_config is None:
        return None
    try:
        return getattr(kv_config, POLICY_FIELD, None)
    except Exception as exc:  # noqa: BLE001 - a property that raises reads as absent
        from . import snapshot  # noqa: PLC0415 - sibling layer, lazy by idiom
        snapshot._error(exc, "load_failure_policy read field")
        return None


def decide(kv_config, *, env=None, witness=None, default=None) -> dict:
    """THE DECISION, without performing it. Never raises; returns one verdict dict.

    `kv_config` is the engine's `KVTransferConfig` (or None). The order is the whole
    policy, so it is spelled out:

      1. OPT-OUT FIRST. `VLLM_E2_TIER_DISK_LOAD_FAILURE_POLICY_DEFAULT=off` means the
         plugin does not touch the field, whatever it says. Checked before anything
         else, because it is the operator's veto over this module.
      2. THE OPERATOR NAMED IT. A reliable CLI witness with the key present says so,
         and the value - `fail` included - is left exactly where it is. This is the
         case the whole module is built to get right, and note what it does NOT
         require: the value does not have to differ from the default, because the
         witness is about the operator's TEXT, not about the number.
      3. NOBODY NAMED IT (a reliable CLI witness without the key). Then the value we
         are holding IS the engine default, and writing `recompute` is filling a gap
         rather than reversing a choice. If the engine's own default is already
         `recompute`, there is nothing to write and the verdict says `engine-already`
         rather than taking credit.
      4. NO WITNESS AT ALL (a Python-built config, or a CLI value we could not parse).
         The value is compared against the engine's own default: a non-default value
         cannot be an accident, so it is reported as an operator choice. A
         default-valued field is `unresolved` and NOTHING IS WRITTEN - except under
         `force`, where the operator has asserted he did not set it.
      5. A WRITE IS ONLY A WRITE WHEN IT READS BACK. `after` is always re-read from
         the object, never the value this module intended, and a mismatch is
         `write-refused` - a provenance string that reports intent is how the
         2026-10-06 `cap_ms` investigation lost an evening.
    """
    verdict = _empty()
    try:
        env = config.disk_load_failure_policy_default() if env is None else env
        verdict["env_raw"] = env.get("raw")
        verdict["forced"] = bool(env.get("force"))

        # THE OPT-OUT FIRST, AND NOTHING ELSE READ: not the command line, not the
        # engine's config class. `off` is the operator's veto over THIS MODULE, and a
        # step that went on to parse his command line anyway would be a rollback with
        # side effects - which is not what a rollback is for.
        if not env.get("enabled", True):
            verdict["decision"] = VERDICT_OPTED_OUT
            verdict["reason"] = (
                f"{config.ENV_DISK_LOAD_FAILURE_POLICY_DEFAULT}="
                f"{verdict['env_raw']!r} is a disable word: the plugin leaves "
                f"{POLICY_FIELD}={_value_of(kv_config)!r} exactly as the engine set "
                "it")
            verdict["after"] = _value_of(kv_config)
            return verdict

        witness = operator_witness() if witness is None else witness
        verdict["witness"] = witness.get("source")
        verdict["operator_names_field"] = witness.get("names_field")
        verdict["operator_value"] = witness.get("operator_value")
        verdict["problems"] = list(witness.get("problems") or [])[:MAX_PROBLEM_ITEMS]

        before = _value_of(kv_config)
        verdict["before"] = before
        verdict["after"] = before

        if kv_config is None or before is None:
            verdict["decision"] = VERDICT_NO_CONFIG
            verdict["reason"] = ("no KVTransferConfig (or no "
                                 f"{POLICY_FIELD}) was reachable from the config the "
                                 "connector factory was handed, so there is nothing "
                                 "to decide")
            return verdict

        if witness.get("reliable") and witness.get("names_field"):
            verdict["decision"] = VERDICT_OPERATOR
            verdict["reason"] = (
                f"the operator's own --kv-transfer-config names "
                f"{POLICY_FIELD}={witness.get('operator_value')!r}: an explicit choice "
                "is never overridden, whichever way it points")
            return verdict

        if witness.get("reliable"):
            # Nobody named it: what we hold IS the engine's default, whatever it is.
            if before == POLICY_RECOMPUTE:
                verdict["decision"] = VERDICT_ALREADY
                verdict["reason"] = (f"{POLICY_FIELD} already reads "
                                     f"{POLICY_RECOMPUTE!r} (the engine's own default, "
                                     "since the operator's JSON does not name it): "
                                     "nothing was written")
                return verdict
            return _write_recompute(verdict, kv_config,
                                    f"the operator's --kv-transfer-config does not "
                                    f"name {POLICY_FIELD}, so "
                                    f"{POLICY_FIELD}={before!r} is the engine's "
                                    "default and not a choice")

        # NO WITNESS. Only the "someone chose this" direction is readable, and it
        # outranks even `force`: an operator asserting he did not set a field does not
        # make a value that is NOT the engine's default into a default.
        default = engine_default() if default is None else default
        verdict["engine_default"] = default.get("default")
        verdict["engine_default_source"] = default.get("source")
        if (default.get("default") is not None
                and before != default["default"]):
            verdict["decision"] = VERDICT_OPERATOR_VALUE
            verdict["reason"] = (
                f"{POLICY_FIELD}={before!r} is not the engine's own default "
                f"({default['default']!r}), so somebody chose it: left alone "
                f"(witness: {witness.get('source')})")
            return verdict

        if before == POLICY_RECOMPUTE:
            verdict["decision"] = VERDICT_ALREADY
            verdict["reason"] = (f"{POLICY_FIELD} already reads "
                                 f"{POLICY_RECOMPUTE!r}: nothing was written")
            return verdict

        if not verdict["forced"]:
            verdict["decision"] = VERDICT_UNRESOLVED
            verdict["problems"] = (verdict["problems"]
                                   + ([] if default.get("default") is not None
                                      else [_clip(default.get("reason")
                                                  or "no engine default readable",
                                                  MAX_PROBLEM_CHARS)])
                                   )[:MAX_PROBLEM_ITEMS]
            verdict["reason"] = (
                f"{POLICY_FIELD} reads {before!r} and this boot cannot show that "
                f"nobody chose it (no --kv-transfer-config names it: "
                f"{witness.get('source')}; engine default "
                f"{default.get('default')!r} via "
                f"{default.get('source')}), so it is treated as chosen and left "
                "alone. Name it in the connector JSON, or assert you did not set it "
                f"with {config.ENV_DISK_LOAD_FAILURE_POLICY_DEFAULT}=1")
            return verdict

        return _write_recompute(
            verdict, kv_config,
            f"no CLI witness exists ({witness.get('source')}), "
            f"{POLICY_FIELD}={before!r} is not a value this process can attribute to "
            f"anybody, and "
            f"{config.ENV_DISK_LOAD_FAILURE_POLICY_DEFAULT}={verdict['env_raw']!r} "
            "asserts the operator did not set it")
    except Exception as exc:  # noqa: BLE001 - a policy step never costs a boot
        from . import snapshot  # noqa: PLC0415 - sibling layer, lazy by idiom
        snapshot._error(exc, "load_failure_policy decide")
        verdict["decision"] = VERDICT_WRITE_REFUSED
        verdict["reason"] = _clip(f"the policy step raised {type(exc).__name__}: "
                                  f"{exc}", MAX_REASON_CHARS)
        return verdict
    finally:
        verdict["token"] = boot_token(verdict)


def _write_recompute(verdict: dict, kv_config, why: str) -> dict:
    """THE WRITE, and the read-back that is the only proof of it. Never raises."""
    try:
        setattr(kv_config, POLICY_FIELD, POLICY_RECOMPUTE)
    except Exception as exc:  # noqa: BLE001 - a frozen config is the engine's answer
        from . import snapshot  # noqa: PLC0415 - sibling layer, lazy by idiom
        snapshot._error(exc, "load_failure_policy write")
        verdict["decision"] = VERDICT_WRITE_REFUSED
        verdict["after"] = _value_of(kv_config)
        verdict["reason"] = _clip(f"{why}; the write was refused by "
                                  f"{type(exc).__name__}: {exc}", MAX_REASON_CHARS)
        return verdict

    # READ IT BACK. Everything the boot line and the payload claim about the engine's
    # behaviour comes from this value, never from `POLICY_RECOMPUTE` above: what the
    # engine will do is whatever the object says AFTER us, and a config class with a
    # property, a validator or a frozen slot would otherwise be reported as changed.
    after = _value_of(kv_config)
    verdict["after"] = after
    if after != POLICY_RECOMPUTE:
        verdict["decision"] = VERDICT_WRITE_REFUSED
        verdict["reason"] = _clip(
            f"{why}; setattr was accepted but the field reads {after!r} afterwards, "
            f"so the engine will NOT see {POLICY_RECOMPUTE!r}", MAX_REASON_CHARS)
        return verdict
    verdict["decision"] = VERDICT_PLUGIN_SET
    verdict["applied"] = True
    verdict["reason"] = _clip(f"{why}: {POLICY_FIELD} set to "
                              f"{POLICY_RECOMPUTE!r} and read back as such",
                              MAX_REASON_CHARS)
    return verdict


def boot_step(vllm_config) -> dict:
    """THE BOOT STEP, run by the connector-creation hook. Never raises, logs ONE line.

    An inert boot (`VLLM_E2_TIER_DEV_BOOT_INERT`) is refused here and not inside
    `decide()`, for the same reason `lookup_ledger.arm_at_boot()` refuses it there: the
    inert promise is "this plugin changed nothing in this process", and a verdict that
    says `inert` is that promise kept in the payload rather than inferred from the
    absence of a field.
    """
    verdict = _empty()
    try:
        if config.boot_inert():
            verdict["decision"] = VERDICT_INERT
            verdict["reason"] = (f"this boot is inert "
                                 f"({config.ENV_BOOT_INERT}): the plugin writes no "
                                 f"engine state, {POLICY_FIELD} included")
        else:
            verdict = decide(kv_transfer_config_of(vllm_config))
    except Exception as exc:  # noqa: BLE001 - a boot step never costs a boot
        from . import snapshot  # noqa: PLC0415 - sibling layer, lazy by idiom
        snapshot._error(exc, "load_failure_policy boot step")
        verdict["decision"] = VERDICT_WRITE_REFUSED
        verdict["reason"] = _clip(f"the boot step raised {type(exc).__name__}: "
                                  f"{exc}", MAX_REASON_CHARS)
    verdict["token"] = boot_token(verdict)

    # ONE LINE PER BOOT, because the question "what will a failed restore cost my
    # user?" is asked at least once per incident and the answer must survive in the
    # log without anybody curling a verb. WARNING for the two states where the engine
    # will FAIL requests and the plugin did not fix it; INFO for the states that are
    # what the operator asked for.
    line = (f"kvtier-dev-plugin: {POLICY_FIELD}={verdict.get('after')!r} "
            f"(who decided: {verdict.get('decision')}) - "
            f"{verdict.get('reason')}")
    if verdict.get("decision") in (VERDICT_UNRESOLVED, VERDICT_WRITE_REFUSED,
                                   VERDICT_NO_CONFIG):
        logger.warning("%s", line)
    else:
        logger.info("%s", line)
    return verdict


def _value_token(value) -> str:
    """One of `recompute` / `fail` / `unknown`, for a token that must stay bounded."""
    return VALUE_TOKENS.get(str(value), "unknown")


def boot_token(record) -> str:
    """The BOOT line's `LOAD_POLICY[what:who]` token, from THIS boot's own record.

    Bracketed and never `key=value`, for the reason `mamba_anchor.boot_token()` gives
    in full: the restart verifier in the llm-api repo pulls its PASS criterion with
    `grep -oE "tier=[A-Za-z_]+" | tail -1`, and a token with no `=` can never become
    a candidate match. Read from the BOOT DICT (the `JOURNAL_DRAIN` discipline), so a
    line describes the boot that produced it. The two halves of the token are the two
    facts the upstream finding asked for: WHAT the engine will do, and WHO decided -
    `LOAD_POLICY[fail:operator]` and `LOAD_POLICY[fail:unresolved]` are the same
    behaviour with two different remedies, and a line that could not tell them apart
    would send an operator to the wrong file.
    """
    try:
        if not isinstance(record, dict):
            return f"LOAD_POLICY[{VERDICT_NOT_RUN}]"
        decision = record.get("decision")
        if decision == VERDICT_NOT_RUN:
            return f"LOAD_POLICY[{VERDICT_NOT_RUN}]"
        if decision == VERDICT_PLUGIN_SET:
            return f"LOAD_POLICY[{_value_token(record.get('after'))}:plugin]"
        if decision in (VERDICT_OPERATOR, VERDICT_OPERATOR_VALUE):
            return f"LOAD_POLICY[{_value_token(record.get('after'))}:operator]"
        if decision == VERDICT_ALREADY:
            return f"LOAD_POLICY[{_value_token(record.get('after'))}:engine]"
        if decision == VERDICT_UNRESOLVED:
            return f"LOAD_POLICY[{_value_token(record.get('after'))}:unresolved]"
        if decision == VERDICT_WRITE_REFUSED:
            return f"LOAD_POLICY[{_value_token(record.get('after'))}:refused]"
        if decision in (VERDICT_OPTED_OUT, VERDICT_INERT):
            return f"LOAD_POLICY[untouched:{decision}]"
        if decision == VERDICT_NO_CONFIG:
            return "LOAD_POLICY[none]"
        return f"LOAD_POLICY[{VERDICT_WRITE_REFUSED}]"
    except Exception:  # noqa: BLE001 - a token is never worth a lost boot line
        return f"LOAD_POLICY[{VERDICT_NOT_RUN}]"
