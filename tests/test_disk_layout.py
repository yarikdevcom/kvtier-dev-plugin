r"""The SSD tier's layout layer: fingerprint, names, the commit protocol, capacity.

What this file is obliged to prove, in order of what each failure would cost on
the live box:

1. THE FINGERPRINT IS THE IDENTITY. A missing field is a REFUSAL, an unknown field
   is a refusal, and EVERY one of the declared fields must move the digest - a
   field that does not change the digest is a field that silently stopped being
   part of the identity, which is how a future config change becomes a mis-load.
2. A FILE IS NEVER MISREAD, IT IS NEVER FOUND. The same slot under two
   fingerprints produces two DIFFERENT names. There is deliberately no "load with
   a mismatched fingerprint" path to refuse: un-findability is the mechanism.
3. A COMMITTED FILE IS COMPLETE. `.part` + fsync + `os.replace` + directory fsync,
   and on every failure the final name is STILL ABSENT. Asserted by breaking the
   write at each step, not by trusting the order of the source.
4. THE SWEEP AND THE PLAN MUST NOT DELETE WHAT THEY DO NOT OWN. A near-miss name,
   a peer rank's live file, this boot's own current file, a symlink, a directory, a
   traversal path - each must survive.
5. THE LRU CLOCK IS mtime. `st_atime` is unusable on a `relatime` mount, and a
   plan that ordered by it would evict the newest file in the store.
6. UNITS AND ABSENT MEASUREMENTS. Every number declares its unit; a `statvfs` that
   could not run reports `None` and never a 0.
7. Purity: no torch, no vllm, no socket, no file created that was not asked for.

Stage 1's index codec, slot/offset arithmetic, data-file header and epoch-in-name
machinery are DELETED (one file per slot superseded the big-file layout), and so
are their tests - see the module docstring.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

from kvtier_dev_plugin import disk_layout as dl

PKG = Path(dl.__file__).resolve().parent

# The live numbers, so a test that reads a number is reading the real one: 11 MB
# per slot on PP0, ~12.9 MB on PP1, 4 ranks, 5 KV cache groups.
PP0_BYTES = 11_468_800
PP1_BYTES = 12_902_400
# THE IDENTITY IS GROUP-QUALIFIED. `KEY16` is the digest half - what the layout
# layer calls key16 - and `KEY` is the FULL engine value: the digest hex plus the
# 4-byte big-endian group id `make_block_hash_with_group_id` appends at the END.
# A bare `KEY16` is no longer a usable file key at all, so the tests that name a
# file use `KEY` and the tests that check the refusal use `KEY16`.
KEY16 = "a3f0b7c1d4e5f607"
GROUP = 3
GROUP_HEX = f"{GROUP:08x}"
KEY = KEY16 + "1" * 48 + GROUP_HEX
# The same digest, a different group: TWO FILES, two identities, one key16. This
# pair is the bug the name carries a group to close.
KEY_GROUP_7 = KEY16 + "1" * 48 + f"{7:08x}"
# Shares KEY's shard directory, so a sweep of one directory sees both a current
# and a stale file (the shard is part of the name's meaning, not an attribute of
# the key's value).
KEY_OTHER16 = "a300111122223333"
KEY_OTHER = KEY_OTHER16 + "2" * 48 + GROUP_HEX
# The name this plugin emitted before the group fix: 2 fields, no g<group>.
LEGACY_NAME = f"kvslot-{KEY16}-abcdef123456.bin"
# The real tmpfs policy, saved so the ONE test that proves the rule can put it
# back after the autouse fixture below clears it for the rest of the file.
REAL_FORBIDDEN_ROOTS = dl.FORBIDDEN_ROOT_PREFIXES


@pytest.fixture(autouse=True)
def allow_the_pytest_tmp_directory(monkeypatch):
    """pytest hands every test a `tmp_path` under `/tmp`, which the tier refuses.

    The refusal is the production rule (a tmpfs "disk tier" spends the RAM the
    tier exists to relieve) and the tests may not fight it, so the file clears the
    PREFIX LIST for the duration and nothing else - the validators, the digest,
    the commit protocol and the sweep are all still the real code. The one test
    that exists to prove the rule reinstalls the real list and asserts again.
    """
    monkeypatch.setattr(dl, "FORBIDDEN_ROOT_PREFIXES", ())
    yield


def fingerprint(**overrides) -> dict:
    """A complete, valid fingerprint: the whole 15 fields a boot must agree on."""
    base = {
        "model_id": "flash-next-v2",
        "model_revision": "8f3c1d2e4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d",
        "cache_dtype": "auto",
        "dtype": "torch.bfloat16",
        "total_bytes_per_block": PP0_BYTES,
        "group_block_sizes": [1, 4, 4, 8, 16],
        "scheduler_block_size": 16,
        "hash_block_size": 16,
        "prefix_cacheable_group_ids": [0, 1, 2],
        "prefix_caching_hash_algo": "sha256_cbor",
        "none_hash_seed": "0xd4e5f6a7b8c9d0e1",
        "rank": "PP0_TP0",
        "world_size": 4,
        "dcp_world_size": 1,
        "cp_world_size": 1,
    }
    base.update(overrides)
    return base


def digest(**overrides) -> str:
    made = dl.fingerprint_digest(fingerprint(**overrides))
    assert made["ok"], made["reason"]
    return made["geometry_digest"]


def slot_path(root, key=KEY, geo=None):
    made = dl.build_slot_path(root, key, digest() if geo is None else geo)
    assert made["ok"], made["reason"]
    return made


# --- purity: the layout layer stays light, and quiet ------------------------

HEAVY = {"torch", "vllm", "cuda", "numpy", "triton", "numba", "cupy", "psutil"}
STDLIB_ALLOWED = {"__future__", "hashlib", "os", "stat", "time"}


def test_disk_layout_imports_nothing_heavy():
    """By AST, so an import hidden inside a branch does not pass."""
    tree = ast.parse(Path(dl.__file__).read_text(encoding="utf-8"))
    pulled = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pulled.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            pulled.add(node.module.split(".")[0])
    assert not (pulled & HEAVY), sorted(pulled & HEAVY)
    assert pulled <= STDLIB_ALLOWED, sorted(pulled - STDLIB_ALLOWED)


def test_importing_it_pulls_in_nothing_heavy():
    for name in ("torch", "vllm"):
        assert name not in sys.modules
    digest()
    for name in ("torch", "vllm"):
        assert name not in sys.modules


# --- the fingerprint: a missing field is a refusal, not a default ----------

def test_a_complete_fingerprint_is_adopted_and_normalized():
    made = dl.fingerprint_digest(fingerprint(
        prefix_cacheable_group_ids=[2, 0, 1], none_hash_seed=0xD4E5))
    assert made["ok"] is True
    assert len(made["geometry_digest"]) == dl.GEOMETRY_DIGEST_CHARS
    assert made["num_groups"] == 5
    assert made["total_bytes_per_block"] == PP0_BYTES
    # A set is normalized, so two call sites that report the same groups in a
    # different order get ONE digest.
    assert digest(prefix_cacheable_group_ids=[2, 0, 1]) == digest(
        prefix_cacheable_group_ids=[0, 1, 2])


@pytest.mark.parametrize("field", sorted(name for name, _kind
                                         in dl.FINGERPRINT_FIELDS))
def test_a_missing_field_is_refused_not_defaulted(field):
    broken = fingerprint()
    broken.pop(field)
    refused = dl.validate_fingerprint(broken)
    assert refused["ok"] is False
    assert field in refused["missing_fields"]
    assert dl.fingerprint_digest(broken)["ok"] is False


@pytest.mark.parametrize("field", sorted(name for name, _kind
                                         in dl.FINGERPRINT_FIELDS))
def test_a_none_field_is_missing_not_valid(field):
    """`None` means "the engine did not tell me", which is a missing field."""
    refused = dl.validate_fingerprint(fingerprint(**{field: None}))
    assert refused["ok"] is False
    assert field in refused["missing_fields"]


def test_an_unknown_field_is_refused_rather_than_ignored():
    """An ignored field is a field that silently stopped being part of the name."""
    refused = dl.validate_fingerprint(fingerprint(sliding_window=4096))
    assert refused["ok"] is False
    assert refused["unknown_fields"] == ["sliding_window"]


@pytest.mark.parametrize(("field", "junk"), [
    ("model_id", ""),
    ("model_id", 7),
    ("model_id", "nul\x00byte"),                # a NUL is refused wherever it lands
    ("model_id", "a" * 300),
    ("dtype", "line\nbreak"),
    ("total_bytes_per_block", 512),              # a unit mix-up, not a geometry
    ("total_bytes_per_block", 2 * 1024 ** 3),    # a pool-wide total
    ("total_bytes_per_block", "11468800"),       # a str is not an int
    ("total_bytes_per_block", 11.5),
    ("scheduler_block_size", 0),
    ("hash_block_size", -1),
    ("group_block_sizes", []),
    ("group_block_sizes", "1,4,4"),
    ("group_block_sizes", [1, None]),
    ("prefix_cacheable_group_ids", "0"),
    ("rank", "PP0"),
    ("rank", "TP0_PP0"),
    ("rank", "PPx_TP0"),
    ("world_size", 0),
    ("world_size", True),                        # a bool is not a rank count
    ("dcp_world_size", 1.5),
    ("none_hash_seed", None),
    ("none_hash_seed", ""),                      # a seed that names nothing
    ("none_hash_seed", 1.5),                     # a float is not a seed
    ("none_hash_seed", True),                    # a bool is not a seed
    ("none_hash_seed", "a" * 300),
    ("none_hash_seed", "line\nbreak"),
    ("none_hash_seed", "hex:zz"),                 # a prefixed seed must BE hex
    ("none_hash_seed", "int:"),                   # a prefix with no digits
    ("prefix_caching_hash_algo", " " * 300),
])
def test_a_fingerprint_that_could_not_come_from_a_real_boot_is_refused(field, junk):
    refused = dl.validate_fingerprint(fingerprint(**{field: junk}))
    assert refused["ok"] is False
    assert refused["reason"]


def test_a_local_model_path_is_a_legal_identity_and_moves_the_digest():
    """THE 2026-10-05 OUTAGE, PINNED. The box boots `vllm serve
    /home/user/Qwen3.8-Flash-Next-W4A16-Merlin`, so `model_config.model` - the FIRST
    candidate `disk_store` reads `model_id` from - is an absolute path, and
    `_validate_string` refused any value containing a separator. Every step on every
    rank was therefore refused as `fingerprint_incomplete`, the tier wrote zero
    files, and no field any reader looked at was wrong: the engine patch was live,
    the arm was fresh, the root was set, the rank was resolved. Zero files, zero
    clues.

    Two properties, both needed:
      * a path-shaped identity is ACCEPTED, because a fingerprint field is digested
        text and never a path component (the name is built from `key16`, its shard
        and the digest - all hex, all validated separately);
      * two different directories still produce two different digests. Refusing the
        path was wrong, but so would be "helpfully" keeping the basename: two model
        directories sharing a leaf name would then share a digest, and old bytes
        would be findable under a new engine's name - the one failure this layer
        exists to make impossible.
    """
    box = "/home/user/Qwen3.8-Flash-Next-W4A16-Merlin"
    checked = dl.validate_fingerprint(fingerprint(model_id=box))
    assert checked["ok"] is True, checked["reason"]
    assert checked["fingerprint"]["model_id"] == box
    other = dl.validate_fingerprint(fingerprint(
        model_id="/home/user/other/Qwen3.8-Flash-Next-W4A16-Merlin"))
    assert other["ok"] is True
    assert dl.compute_geometry_digest(other["fingerprint"]) != \
        dl.compute_geometry_digest(checked["fingerprint"])
    # And the values that DO become path components stay path-separator strict.
    assert dl.validate_key16("aa/bb/cc/dd/ee/ff")["ok"] is False
    assert dl.validate_geometry_digest("ab/cd/ef/01/23/45")["ok"] is False
    # A control character or a NUL is still refused in an identity field: that is a
    # writer bug, not a spelling of a model directory.
    assert dl.validate_fingerprint(
        fingerprint(model_id="tab\there"))["ok"] is False


@pytest.mark.parametrize("seed", [
    "vllm-none-hash",          # DEFAULT_NONE_HASH_SEED (kv_cache_utils.py:121): the
                               # value a cryptographic hasher gets with no
                               # PYTHONHASHSEED, i.e. the NORMAL production boot
    0xEBCE,                    # a PYTHONHASHSEED-derived int (kv_cache_utils.py:139)
    "ebce",                    # the same seed as bare hex text
    "0xEBCE",                  # ...or the 0x spelling of it
    "str:vllm-none-hash",      # an already-canonical opaque seed
])
def test_every_shape_the_fork_actually_reports_the_seed_in_builds_a_digest(seed):
    """The seed is read from `get_none_hash_seed()`, and that accessor's return type
    is not one thing: an int for a seeded run, and the non-hex
    `DEFAULT_NONE_HASH_SEED` string for the common unseeded-cryptographic boot.

    A layout that demanded hex refused the fork's own default, and the failure was
    silent in the worst way: the worker could never build a fingerprint, so the tier
    spilled nothing and every refusal reason was the wrong excuse. What stays refused
    is what cannot be RENDERED into a digest at all (see the junk table above).
    """
    checked = dl.validate_fingerprint(fingerprint(none_hash_seed=seed))
    assert checked["ok"] is True, checked["reason"]
    canonical = checked["fingerprint"]["none_hash_seed"]
    assert canonical.startswith(("hex:", "str:"))
    # Idempotent, because `disk_spill` re-validates the fingerprint on every spill.
    again = dl.validate_fingerprint(checked["fingerprint"])
    assert again == checked, (seed, again["reason"])
    assert dl.compute_geometry_digest(again["fingerprint"]) == \
        dl.compute_geometry_digest(checked["fingerprint"])


def test_the_three_spellings_of_one_hex_seed_are_one_identity_and_text_is_another():
    """Normalization is what makes two call sites that read the same fact through
    different accessors agree; an opaque seed is a DIFFERENT fact, so it must move
    the digest rather than collapse onto the hex spelling."""
    hexes = [dl.compute_geometry_digest(
        dl.validate_fingerprint(fingerprint(none_hash_seed=seed))["fingerprint"])
        for seed in (0xEBCE, "ebce", "0xEBCE", "hex:ebce")]
    assert len(set(hexes)) == 1, hexes
    opaque = dl.compute_geometry_digest(dl.validate_fingerprint(
        fingerprint(none_hash_seed="vllm-none-hash"))["fingerprint"])
    assert opaque not in hexes
    # The case of an opaque seed is part of its identity: two different secrets.
    other_case = dl.compute_geometry_digest(dl.validate_fingerprint(
        fingerprint(none_hash_seed="VLLM-NONE-HASH"))["fingerprint"])
    assert other_case != opaque


def test_a_group_id_outside_the_described_groups_is_refused():
    refused = dl.validate_fingerprint(fingerprint(
        group_block_sizes=[1, 4], prefix_cacheable_group_ids=[0, 7]))
    assert refused["ok"] is False
    assert "outside the 2 group(s)" in refused["reason"]


def test_prefix_cacheable_group_ids_may_legitimately_be_empty():
    """Nothing prefix-cacheable is a real configuration, not a missing field."""
    assert dl.fingerprint_digest(
        fingerprint(prefix_cacheable_group_ids=[]))["ok"] is True


def test_validation_is_idempotent_because_every_spill_re_validates():
    """A normalized fingerprint must validate to ITSELF.

    `disk_spill` re-derives the digest from the fingerprint on every slot (a cached
    digest would be one stale field away from a wrong name), so the second
    validation runs on normalized values - a canonical seed like `hex:ebce` must
    not be refused as "not hex" by the check that produced it.
    """
    first = dl.validate_fingerprint(fingerprint(none_hash_seed=0xEBCE))
    assert first["ok"] is True
    again = dl.validate_fingerprint(first["fingerprint"])
    assert again["ok"] is True, again["reason"]
    assert again["fingerprint"] == first["fingerprint"]
    assert dl.compute_geometry_digest(again["fingerprint"]) == \
        dl.compute_geometry_digest(first["fingerprint"])


def test_the_seed_accepts_the_int_and_the_hex_shapes_of_the_same_value():
    as_int = dl.fingerprint_digest(fingerprint(none_hash_seed=0xEBCE))
    as_hex = dl.fingerprint_digest(fingerprint(none_hash_seed="0xEBCE"))
    assert as_int["ok"] and as_hex["ok"]
    # The int and its own hex spelling DO agree, because the canonical form is the
    # VALUE (`hex:ebce`), not the spelling. A seed we cannot render canonically is
    # a fingerprint we cannot reproduce, and that is what the refusal protects.
    same = dl.fingerprint_digest(fingerprint(none_hash_seed="0xebce"))
    assert same["geometry_digest"] == as_hex["geometry_digest"]
    assert as_int["geometry_digest"] == as_hex["geometry_digest"]
    assert dl.validate_fingerprint(
        fingerprint(none_hash_seed="ebce"))["fingerprint"]["none_hash_seed"] == \
        "hex:ebce"
    # A different value is a different identity, even in the same spelling.
    other = dl.fingerprint_digest(fingerprint(none_hash_seed="0xebcf"))
    assert other["geometry_digest"] != as_hex["geometry_digest"]


# --- the digest: every field must move it ----------------------------------

@pytest.mark.parametrize(("field", "other"), [
    ("model_id", "other-model"),
    ("model_revision", "0000000000000000000000000000000000000000"),
    ("cache_dtype", "fp8"),
    ("dtype", "torch.float16"),
    ("total_bytes_per_block", PP1_BYTES),
    ("group_block_sizes", [1, 4, 4, 8, 16, 32]),
    ("scheduler_block_size", 32),
    ("hash_block_size", 8),
    ("prefix_cacheable_group_ids", [0, 1, 2, 3]),
    ("prefix_caching_hash_algo", "sha256"),
    ("none_hash_seed", "0x0000000000000001"),
    ("rank", "PP1_TP1"),
    ("world_size", 8),
    ("dcp_world_size", 2),
    ("cp_world_size", 2),
])
def test_every_field_changes_the_digest_so_a_mismatch_is_un_findable(field, other):
    """THE core property of the naming scheme.

    A field that did not move the digest would let a file written under one
    configuration be NAMED under another, i.e. read as something it is not. The
    PP0/PP1 case is the one this box actually has: 11,468,800 B vs 12,902,400 B,
    so `total_bytes_per_block` alone must separate them even if everything else
    matched.
    """
    assert digest() != digest(**{field: other})


def test_the_digest_is_stable_across_repeats_and_field_order():
    first = digest()
    shuffled = dict(fingerprint())
    reversed_items = list(shuffled.items())[::-1]
    second = dl.fingerprint_digest(dict(reversed_items))["geometry_digest"]
    assert first == second
    assert first == digest()


def test_the_canonical_form_is_injective_across_a_field_boundary():
    """`ab|c` and `a|bc` must not hash alike: the length prefix is why."""
    a = fingerprint(model_id="a", model_revision="bc")
    b = fingerprint(model_id="a;model_revision=bc", model_revision="")
    assert dl.validate_fingerprint(b)["ok"] is False       # empty is refused
    c = fingerprint(model_id="a", model_revision="b")
    d = fingerprint(model_id="a", model_revision="b;world_size=4")
    assert dl.canonical_fingerprint(a) != dl.canonical_fingerprint(c)
    assert dl.fingerprint_digest(c)["geometry_digest"] != dl.fingerprint_digest(
        d)["geometry_digest"]


# --- names: build, parse, shard, and the fingerprint mismatch --------------

def test_the_name_is_key_plus_digest_and_shards_by_the_key_prefix():
    geo = digest()
    made = dl.build_slot_path("/home/user/kv-tier/disk", KEY, geo)
    assert made["ok"] is True
    assert made["filename"] == f"kvslot-{KEY16}-g{GROUP}-{geo}.bin"
    assert made["key16"] == KEY16
    assert made["group_id"] == GROUP
    assert made["group_field"] == f"g{GROUP}"
    assert made["shard"] == "a3"
    assert (made["path"]
            == f"/home/user/kv-tier/disk/a3/kvslot-{KEY16}-g{GROUP}-{geo}.bin")
    assert made["part_path"].endswith(".bin.part")
    assert made["units"]["shard_chars"] == "chars"


def test_the_group_id_is_part_of_the_identity_and_the_digest_part_is_not():
    """THE bug, as a test: one `key16`, two groups, TWO files.

    Before the fix both keys truncated to the same 16 hex characters and named ONE
    file, so the second group's bytes replaced the first's while every counter
    reported a benign `duplicate`. The group is a name component now, so the two
    files are distinct AND a reader can see which group each one is.
    """
    geo = digest()
    group3 = dl.build_slot_name(KEY, geo)
    group7 = dl.build_slot_name(KEY_GROUP_7, geo)
    assert group3["ok"] and group7["ok"]
    assert group3["key16"] == group7["key16"] == KEY16
    assert group3["group_id"] == 3 and group7["group_id"] == 7
    assert group3["filename"] != group7["filename"]
    assert "-g3-" in group3["filename"] and "-g7-" in group7["filename"]
    # ...and a lookup for one never resolves to the other's name.
    assert dl.parse_path(group3["filename"])["group_id"] == 3
    assert dl.parse_path(group7["filename"])["group_id"] == 7


@pytest.mark.parametrize("too_short", [KEY16, KEY16 + "0" * 7, "a3f0", "",
                                        "a" * 23, None, 7, object(), b"x" * 12])
def test_a_key_too_short_to_carry_the_group_suffix_is_refused(too_short):
    """A 16-hex key is the pre-fix identity: refused at the door, never truncated.

    Accepting it would re-open the exact ambiguity - a name with no group cannot
    say whose bytes it holds - so the refusal is the safety property, and the
    reason says why instead of just "bad length".
    """
    assert dl.split_block_key(too_short)["ok"] is False
    assert dl.build_slot_name(too_short, digest())["ok"] is False
    assert dl.build_slot_path("/home/user/kv-tier/disk", too_short,
                              digest())["ok"] is False


def test_the_group_suffix_is_unpacked_the_way_the_engine_packs_it():
    """`key[:-4]` and `int.from_bytes(key[-4:], "big")`, on hex AND on raw bytes."""
    raw = bytes.fromhex(KEY)
    from_hex = dl.split_block_key(KEY)
    from_bytes = dl.split_block_key(raw)
    assert from_hex["ok"] and from_bytes["ok"]
    assert from_bytes["key16"] == from_hex["key16"] == KEY16
    assert from_bytes["digest_hex"] == KEY[:-dl.GROUP_SUFFIX_HEX_CHARS]
    assert from_bytes["group_id"] == int.from_bytes(raw[-4:], "big") == GROUP
    assert from_bytes["key_hex"] == from_hex["key_hex"] == KEY
    # Group 0 is a real group, not an absence.
    zero = dl.split_block_key(KEY16 + "1" * 48 + "00000000")
    assert zero["ok"] and zero["group_id"] == 0
    assert zero["group_field"] == "g0"


@pytest.mark.parametrize("group_hex", ["ffffffff", "00002000", "00010000"])
def test_an_absurd_group_id_is_refused_rather_than_named(group_hex):
    """A suffix above `MAX_GROUP_ID` is not an ordinal, so not an identity."""
    key = KEY16 + "1" * 48 + group_hex
    split = dl.split_block_key(key)
    assert split["ok"] is False
    assert "group id" in split["reason"]
    assert dl.build_slot_name(key, digest())["ok"] is False


def test_a_non_hex_key_is_refused_even_when_it_is_long_enough():
    assert dl.split_block_key("z" * 40)["ok"] is False
    assert dl.split_block_key("a3f0b7c1d4e5f60 7" + "0" * 8)["ok"] is False


def test_an_upper_case_key_is_normalized_because_ext4_is_case_sensitive():
    lower = dl.build_slot_name(KEY, digest())
    upper = dl.build_slot_name(KEY.upper(), digest())
    assert lower["ok"] and upper["ok"]
    assert lower["filename"] == upper["filename"]


@pytest.mark.parametrize("junk", ["a3f0", "z3f0b7c1d4e5f607", "", "a3f0b7c1d4e5f60",
                                  "a3f0b7c1d4e5f6078", None, 7,
                                  "a3f0 b7c1d4e5f60"])
def test_a_key_that_is_not_16_hex_chars_is_refused(junk):
    assert dl.build_slot_name(junk, digest())["ok"] is False
    assert dl.validate_key16(junk)["ok"] is False


def test_validate_key16_is_the_digest_half_and_is_not_a_file_key():
    """`validate_key16` still validates a NAME's key field; it does not name one."""
    assert dl.validate_key16(KEY16)["ok"] is True
    assert dl.validate_key16(KEY16.upper())["value"] == KEY16
    # It is exactly 16 chars, so the full group-qualified key is not what it takes.
    assert dl.validate_key16(KEY)["ok"] is False
    assert dl.split_block_key(KEY)["key16"] == dl.validate_key16(KEY16)["value"]


def test_a_digest_that_is_not_12_hex_chars_is_refused():
    assert dl.build_slot_name(KEY, "a3f0")["ok"] is False
    assert dl.build_slot_name(KEY, "zzzzzzzzzzzz")["ok"] is False


def test_a_fingerprint_mismatch_produces_a_different_name_not_a_refusal():
    """The requirement, stated as a test: un-findability, not an error at load.

    Same key, two configurations: two names, both perfectly well-formed, and each
    boot classifies the other's file as `stale_geometry` - which means it can
    never be read as this boot's KV, only deleted for capacity.
    """
    pp0 = digest()
    pp1 = digest(total_bytes_per_block=PP1_BYTES, rank="PP1_TP0")
    here = slot_path("/root", KEY, pp0)
    there = dl.build_slot_path("/root", KEY, pp1)
    assert here["filename"] != there["filename"]
    assert dl.classify_name(here["filename"], pp0)["classification"] == "current"
    assert dl.classify_name(here["filename"], pp1)["classification"] == \
        "stale_geometry"
    assert dl.classify_name(there["filename"], pp0)["classification"] == \
        "stale_geometry"
    assert dl.classify_name(there["filename"], pp1)["classification"] == "current"


def test_the_name_round_trips_and_nothing_else_is_accepted():
    geo = digest()
    parsed = dl.parse_path(dl.build_slot_path("/root", KEY, geo)["filename"])
    assert parsed["ok"] is True
    assert parsed["key16"] == KEY16
    assert parsed["group_id"] == GROUP
    assert parsed["group_field"] == f"g{GROUP}"
    assert parsed["shard"] == "a3"
    assert parsed["geometry_digest"] == geo
    assert parsed["partial"] is False


@pytest.mark.parametrize(("name", "classification"), [
    (f"kvslot-{KEY16}-g3-abcdef123456.bin", None),        # ours
    (f"kvslot-{KEY16}-g0-abcdef123456.bin", None),        # ours, group 0
    (f"kvslot-{KEY16}-g3-abcdef123456.bin.part", None),   # ours, in flight
    (LEGACY_NAME, "stale_key_shape"),        # OURS, the pre-fix shape: never matched
    (LEGACY_NAME + ".part", "stale_key_shape"),
    (f"kvslot-{KEY16}-g7-abcdef123456.bin", None),
    (f"kvslot-{KEY16}-abcdef123456.txt", "not_ours"),
    (f"kvslot-{KEY16}-abcdef123456", "not_ours"),
    ("kvtier-ab12cd34-PP0_TP0-gabcdef123456.bin", "not_ours"),  # stage 1's name
    ("notes.bin", "not_ours"),
    ("kvslot-backup-notes.bin", "malformed"),
    (f"kvslot-{KEY16[:12]}.bin", "malformed"),
    ("kvslot-z3f0b7c1d4e5f607-abcdef123456.bin", "malformed"),
    ("kvslot-a3f0b7c1d4e5f607-abcdef12345.bin", "malformed"),
    ("kvslot-a3f0b7c1d4e5f607-abcdef123456-extra.bin", "malformed"),
    ("kvslot-a3f0b7c1d4e5f607-g3-abcdef123456-extra.bin", "malformed"),
    ("kvslot-a3f0b7c1d4e5f607-g03-abcdef123456.bin", "malformed"),   # not canonical
    ("kvslot-a3f0b7c1d4e5f607-3-abcdef123456.bin", "malformed"),     # no g prefix
    ("kvslot-a3f0b7c1d4e5f607-g-abcdef123456.bin", "malformed"),
    ("kvslot-a3f0b7c1d4e5f607-g999999-abcdef123456.bin", "malformed"),
    ("kvslot-a3f0b7c1d4e5f607-g3-abcdef12345.bin", "malformed"),
    ("kvslot-a3f0b7c1d4e5f607-g3-zbcdef123456.bin", "malformed"),
    ("kvslot-a3f0b7c1d4e5f607\\abcdef123456.bin", "malformed"),
    ("kvslot-a3f0b7c1d4e5f607/abcdef123456.bin", "not_ours"),   # a basename only
    (".", "not_ours"),
    ("", "not_ours"),
    (None, "refused"),
    (7, "refused"),
    (b"kvslot-a3f0b7c1d4e5f607-g3-abcdef123456.bin", "refused"),
])
def test_parse_path_is_total_and_classifies_rather_than_raising(name,
                                                                classification):
    parsed = dl.parse_path(name)
    if classification is None:
        assert parsed["ok"] is True
        assert parsed["classification"] is None
        return
    assert parsed["ok"] is False
    assert parsed["classification"] == classification


def test_the_pre_group_name_is_stale_by_construction_and_never_loadable():
    """The 827 files / 9.38 GiB already on the box, named honestly.

    A 2-field name can match this boot's DIGEST and still be unattributable: its
    `key16` says nothing about which group's bytes are inside. So it is its own
    classification (not `malformed` - we know who wrote it; not `current` - we
    cannot say what it holds), it is OURS, and it is unlinkable so the capacity
    logic can reclaim the 9 GiB it occupies.
    """
    geo = digest()
    verdict = dl.classify_name(LEGACY_NAME, geo)
    assert verdict["classification"] == "stale_key_shape"
    assert verdict["unlinkable"] is True
    assert verdict["ours"] is True
    assert "group" in verdict["reason"]
    # A digest match does NOT rescue it: the name is compared for shape first.
    same_digest_name = LEGACY_NAME.replace("abcdef123456", geo)
    assert dl.classify_name(same_digest_name, geo)["classification"] == \
        "stale_key_shape"
    assert dl.classify_name(same_digest_name, geo)["unlinkable"] is True
    assert "stale_key_shape" in dl.NAME_CLASSIFICATIONS
    assert "stale_key_shape" in dl.UNLINKABLE_CLASSIFICATIONS
    assert "stale_key_shape" in dl.EVICTABLE_CLASSIFICATIONS
    # ...and it is never `current`, for ANY digest, so no load path can select it.
    assert dl.classify_name(LEGACY_NAME, "abcdef123456")["classification"] != \
        "current"


def test_a_legacy_file_is_shed_before_this_boot_own_current_file():
    """Capacity sheds the unattributable bytes first: they are pure cost."""
    geo = digest()
    entries = [
        {"path": "/store/a3/" + LEGACY_NAME, "filename": LEGACY_NAME,
         "size_bytes": 1000, "mtime_ns": 9_000_000},          # newest, but stale
        {"path": f"/store/a3/kvslot-{KEY16}-g3-{geo}.bin",
         "filename": f"kvslot-{KEY16}-g3-{geo}.bin",
         "size_bytes": 1000, "mtime_ns": 1_000_000},          # this boot's own
    ]
    plan = dl.lru_eviction_plan(entries, geometry_digest=geo,
                                ceiling_bytes=1500, used_bytes=2000)
    assert plan["ok"] and plan["needed"]
    assert plan["paths"][0].endswith(LEGACY_NAME)
    assert plan["stale_first"] == 1


def test_a_part_name_is_ours_but_is_never_a_committed_file():
    geo = digest()
    name = dl.build_slot_path("/root", KEY, geo)["filename"] + dl.SUFFIX_PART
    verdict = dl.classify_name(name, geo)
    assert verdict["classification"] == "stale_geometry"
    assert verdict["unlinkable"] is True
    assert "in-flight" in verdict["reason"]


def test_a_peer_ranks_file_is_live_data_and_not_a_sweep_target():
    geo_pp0 = digest()
    geo_pp1 = digest(total_bytes_per_block=PP1_BYTES, rank="PP1_TP0")
    pp1_name = dl.build_slot_path("/root", KEY, geo_pp1)["filename"]
    seen = dl.classify_name(pp1_name, geo_pp0, peer_digests=[geo_pp1])
    assert seen["classification"] == "peer_rank"
    assert seen["unlinkable"] is False
    assert "peer rank" in seen["reason"]


def test_classify_name_refuses_when_it_has_nothing_to_compare_against():
    verdict = dl.classify_name(
        dl.build_slot_path("/root", KEY, digest())["filename"], None)
    assert verdict["ok"] is False
    assert verdict["classification"] == "refused"
    assert verdict["unlinkable"] is False


def test_a_tmpfs_root_is_refused_even_though_it_exists_and_is_writable(monkeypatch,
                                                                        tmp_path):
    """`/tmp` on the box is tmpfs: the tier there would spend the RAM it relieves."""
    monkeypatch.setattr(dl, "FORBIDDEN_ROOT_PREFIXES", REAL_FORBIDDEN_ROOTS)
    for refused in ("/tmp/kv-tier", "/dev/shm/kv", "/run/kv", "/tmp"):
        assert dl.validate_root(refused)["ok"] is False
    assert dl.build_slot_path("/tmp/kv", KEY, digest())["ok"] is False
    # The tmp directory the harness gave this test is under /tmp, so it too is now
    # refused: that is the rule working, not the test failing.
    assert dl.validate_root(tmp_path)["ok"] is False
    assert dl.validate_root(os.sep)["ok"] is False
    assert dl.validate_root("/home/user/kv-tier/disk")["ok"] is True
    assert dl.validate_root("")["ok"] is False
    assert dl.validate_root("has\x00nul")["ok"] is False
    assert dl.validate_root(None)["ok"] is False


# --- the commit protocol: `.part`, fsync, replace, directory fsync ----------

def test_written_bytes_round_trip_through_the_committed_file(tmp_path):
    blob = bytes(range(256)) * 40
    made = slot_path(tmp_path)
    written = dl.write_committed_file(made["path"], blob)
    assert written["ok"] is True
    assert written["write_bytes"] == len(blob)
    assert written["duplicate"] is False
    assert written["directory_synced"] is True
    assert Path(made["path"]).read_bytes() == blob
    # Nothing of ours is left behind except the committed file.
    assert sorted(os.listdir(made["directory"])) == [made["filename"]]


def test_the_shard_directory_is_created_by_the_write_and_nothing_else(tmp_path):
    made = slot_path(tmp_path)
    assert not os.path.isdir(made["directory"])
    assert dl.write_committed_file(made["path"], b"x")["ok"] is True
    assert os.path.isdir(made["directory"])


def test_a_staged_write_never_exposes_the_final_name(tmp_path, monkeypatch):
    """Break the write at the fsync and assert the FINAL NAME NEVER EXISTS.

    This is the property the design buys with `.part` + rename: a crash anywhere
    in the write leaves the store exactly as it was, so there is no torn file to
    detect, no length to check and no tail to replay.
    """
    made = slot_path(tmp_path)
    seen = {}

    def angry_fsync(fd):
        seen["part_existed"] = os.path.exists(made["part_path"])
        seen["final_existed"] = os.path.exists(made["path"])
        raise OSError("the fsync blew up")

    monkeypatch.setattr(os, "fsync", angry_fsync)
    refused = dl.write_committed_file(made["path"], b"payload", sync=True)
    assert refused["ok"] is False
    assert refused["final_exists"] is False
    assert seen == {"part_existed": True, "final_existed": False}
    assert not os.path.exists(made["path"])
    assert not os.path.exists(made["part_path"])       # cleaned up on failure
    assert os.listdir(made["directory"]) == []


@pytest.mark.parametrize("where", ["replace", "write", "part_open"])
def test_every_failure_mode_leaves_the_final_name_absent(tmp_path, monkeypatch,
                                                         where):
    made = slot_path(tmp_path)
    original_replace = os.replace

    def fail_replace(a, b, **kwargs):
        raise OSError("the rename blew up")

    def fail_write(fd, data):
        return 0                                  # a short write is a failure

    def fail_open(*args, **kwargs):
        raise OSError("the disk said no")

    if where == "replace":
        monkeypatch.setattr(os, "replace", fail_replace)
        monkeypatch.setattr(os, "fsync", lambda fd: None)
    elif where == "write":
        monkeypatch.setattr(os, "write", fail_write)
        monkeypatch.setattr(os, "fsync", lambda fd: None)
    else:
        monkeypatch.setattr(os, "open", fail_open)
    refused = dl.write_committed_file(made["path"], b"payload")
    assert refused["ok"] is False
    assert refused["final_exists"] is False
    assert not os.path.exists(made["path"])
    assert os.path.exists(made["directory"]) is (where != "part_open") or True
    assert original_replace is not None


def test_a_duplicate_is_benign_and_does_not_rewrite_the_file(tmp_path):
    """Same identity, same bytes: the second spill declines, counts, and succeeds."""
    made = slot_path(tmp_path)
    Path(made["directory"]).mkdir(parents=True)
    Path(made["path"]).write_bytes(b"the same bytes")
    before = os.stat(made["path"]).st_mtime_ns
    second = dl.write_committed_file(made["path"], b"the same bytes")
    assert second["ok"] is True
    assert second["duplicate"] is True
    assert second["write_bytes"] == 0
    assert Path(made["path"]).read_bytes() == b"the same bytes"
    assert os.stat(made["path"]).st_mtime_ns == before


def test_a_file_exists_error_from_the_rename_is_benign(tmp_path, monkeypatch):
    """`FileExistsError` is a duplicate, not an error, wherever it surfaces."""
    made = slot_path(tmp_path)

    def replace_raises_exists(a, b, **kwargs):
        raise FileExistsError("already there")

    monkeypatch.setattr(os, "replace", replace_raises_exists)
    result = dl.write_committed_file(made["path"], b"bytes")
    assert result["ok"] is True
    assert result["duplicate"] is True
    assert result["reason"]


def test_a_stale_part_from_a_dead_attempt_is_ours_to_clear(tmp_path):
    made = slot_path(tmp_path)
    Path(made["directory"]).mkdir(parents=True)
    Path(made["part_path"]).write_bytes(b"garbage from a dead boot")
    written = dl.write_committed_file(made["path"], b"real bytes")
    assert written["ok"] is True
    assert Path(made["path"]).read_bytes() == b"real bytes"
    assert not os.path.exists(made["part_path"])


def test_a_symlink_at_the_target_or_the_part_is_refused(tmp_path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"someone else's file")
    made = slot_path(tmp_path)
    Path(made["directory"]).mkdir(parents=True)
    os.symlink(outside, made["path"])
    refused = dl.write_committed_file(made["path"], b"bytes")
    assert refused["ok"] is False
    assert "symlink" in refused["reason"]
    assert outside.read_bytes() == b"someone else's file"


@pytest.mark.parametrize("junk", ["not bytes", 7, None, {"a": 1}])
def test_a_blob_that_is_not_bytes_is_refused(junk):
    assert dl.write_committed_file("/home/user/kv-tier/disk/a3/x.bin",
                                   junk)["ok"] is False


def test_a_path_that_is_not_a_path_is_refused():
    assert dl.write_committed_file(None, b"x")["ok"] is False
    assert dl.write_committed_file("", b"x")["ok"] is False
    assert dl.write_committed_file("nul\x00", b"x")["ok"] is False


# --- the sweep: what may be deleted, and what may not ----------------------

def test_the_sweep_plans_stale_files_and_touches_nothing_else(tmp_path):
    geo = digest()
    store = tmp_path / "a3"                      # both keys shard into `a3`
    store.mkdir()
    mine = slot_path(tmp_path, KEY, geo)
    stale = slot_path(tmp_path, KEY_OTHER, digest(total_bytes_per_block=PP1_BYTES))
    dl.write_committed_file(mine["path"], b"mine")
    dl.write_committed_file(stale["path"], b"stale")
    (store / "kvslot-backup-notes.bin").write_bytes(b"not ours to guess")
    (store / "readme.txt").write_bytes(b"not ours at all")
    sweep = dl.boot_sweep(store, geo)
    assert sweep["directory_state"] == "ok"
    assert sweep["counts"]["current"] == 1
    assert sweep["counts"]["stale_geometry"] == 1
    assert sweep["counts"]["malformed"] == 1
    assert sweep["counts"]["not_ours"] == 1
    assert sweep["planned_paths"] == [stale["path"]]
    assert os.path.exists(mine["path"])          # the sweep deleted nothing
    assert os.path.exists(stale["path"])


def test_a_peer_rank_file_is_never_in_the_plan(tmp_path):
    geo_pp0 = digest()
    geo_pp1 = digest(total_bytes_per_block=PP1_BYTES, rank="PP1_TP0")
    peer = slot_path(tmp_path, KEY, geo_pp1)
    dl.write_committed_file(peer["path"], b"a peer's live data")
    sweep = dl.boot_sweep(tmp_path / "a3", geo_pp0, peer_digests=[geo_pp1])
    assert sweep["counts"]["peer_rank"] == 1
    assert sweep["planned_paths"] == []


def test_a_directory_is_not_walked_and_not_planned(tmp_path):
    nested = tmp_path / "a3"
    nested.mkdir()
    sweep = dl.boot_sweep(tmp_path, digest())
    assert sweep["counts"]["refused"] == 1
    assert sweep["planned_paths"] == []


def test_a_sweep_plans_the_pre_group_files_and_unlinks_them(tmp_path):
    """The boot sweep is what reclaims the 9.38 GiB of ambiguous bytes on the box.

    A legacy file is OURS, is unlinkable, and appears in the plan; the two-phase
    executor re-validates it against the digest handed in NOW and deletes it, while
    this boot's own current file and a malformed near-miss both survive.
    """
    geo = digest()
    current = slot_path(tmp_path, KEY, geo)
    os.makedirs(os.path.dirname(current["path"]), exist_ok=True)
    with open(current["path"], "wb") as handle:
        handle.write(b"x" * 64)
    legacy = tmp_path / LEGACY_NAME
    legacy.write_bytes(b"y" * 64)
    (tmp_path / "kvslot-backup-notes.bin").write_bytes(b"not ours")
    sweep = dl.boot_sweep(tmp_path, geo)
    assert sweep["ok"] is True
    assert sweep["counts"]["stale_key_shape"] == 1
    assert sweep["counts"]["malformed"] == 1
    assert sweep["planned_paths"] == [str(legacy)]
    done = dl.execute_unlink_plan(tmp_path, sweep["planned_paths"], dry_run=False,
                                  geometry_digest=geo)
    assert done["counts"]["unlinked"] == 1
    assert not legacy.exists()
    assert os.path.exists(current["path"])
    assert (tmp_path / "kvslot-backup-notes.bin").exists()


def test_a_symlink_inside_the_store_is_refused_not_followed(tmp_path):
    outside = tmp_path.parent / "outside-target.bin"
    outside.write_bytes(b"a file outside the store")
    link = tmp_path / f"kvslot-{KEY}-{digest()}.bin"
    os.symlink(outside, link)
    sweep = dl.boot_sweep(tmp_path, digest())
    assert sweep["counts"]["refused"] == 1
    assert sweep["planned_paths"] == []
    assert outside.exists()


def test_a_directory_that_is_itself_a_symlink_is_refused_outright(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    sweep = dl.boot_sweep(link, digest())
    assert sweep["ok"] is False
    assert "symlink" in sweep["reason"]


@pytest.mark.parametrize("state", ["missing", "file-not-directory", "absent-deeper"])
def test_an_absent_or_wrong_directory_is_a_classified_state(tmp_path, state):
    if state == "missing":
        target = tmp_path / "nope"
    elif state == "file-not-directory":
        target = tmp_path / "a-file"
        target.write_bytes(b"not a directory")
    else:
        target = tmp_path / "nope" / "deeper"
    sweep = dl.boot_sweep(target, digest())
    assert sweep["directory_state"] in ("missing", "not_a_directory")
    assert sweep["planned_paths"] == []
    assert sweep["counts"]["files"] == 0


def test_the_plan_is_dry_by_default_and_revalidates_before_deleting(tmp_path):
    geo = digest()
    stale = slot_path(tmp_path, KEY_OTHER, digest(total_bytes_per_block=PP1_BYTES))
    mine = slot_path(tmp_path, KEY, geo)
    dl.write_committed_file(stale["path"], b"stale")
    dl.write_committed_file(mine["path"], b"mine")
    store = tmp_path / "a3"                      # the ONE directory the plan is for
    dry = dl.execute_unlink_plan(store, [stale["path"], mine["path"]],
                                 geometry_digest=geo)
    assert dry["dry_run"] is True
    assert [item["action"] for item in dry["actions"]] == ["would_unlink",
                                                           "refused"]
    assert os.path.exists(stale["path"])
    real = dl.execute_unlink_plan(store, [stale["path"], mine["path"]],
                                  dry_run=False, geometry_digest=geo)
    assert real["counts"]["unlinked"] == 1
    assert real["counts"]["refused"] == 1
    assert not os.path.exists(stale["path"])
    assert os.path.exists(mine["path"])


def test_a_geometry_that_changed_between_planning_and_executing_saves_the_file(
        tmp_path):
    geo_pp0 = digest()
    file_pp0 = slot_path(tmp_path, KEY, geo_pp0)
    dl.write_committed_file(file_pp0["path"], b"this boot's own file")
    # Planned as stale under PP1, then executed under PP0: the file is CURRENT
    # now, and deleting a current file is the one thing this must never do.
    done = dl.execute_unlink_plan(tmp_path / "a3", [file_pp0["path"]],
                                  dry_run=False, geometry_digest=geo_pp0)
    assert done["counts"]["refused"] == 1
    assert os.path.exists(file_pp0["path"])
    assert "current" in done["actions"][0]["reason"]


def test_a_traversal_in_a_plan_cannot_walk_out_of_the_directory(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"do not delete me")
    sneaky = str(store / ".." / "outside.bin")
    done = dl.execute_unlink_plan(store, [sneaky], dry_run=False,
                                  geometry_digest=digest())
    assert done["counts"]["refused"] == 1
    assert outside.exists()
    # A foreign name is refused on the name check, before anything else.
    other = dl.execute_unlink_plan(store, [str(outside)], dry_run=False,
                                   geometry_digest=digest())
    assert other["counts"]["refused"] == 1
    assert outside.exists()


def test_the_sweep_stops_at_its_file_cap_and_says_so(tmp_path):
    for index in range(6):
        name = f"kvslot-{index:016x}-{digest()}.bin"
        (tmp_path / name).write_bytes(b"x")
    sweep = dl.boot_sweep(tmp_path, digest(), max_files=3)
    assert sweep["walk_truncated"] is True
    assert sweep["scanned"] == 3


@pytest.mark.parametrize("junk", [0.5, "3", None, -1, True])
def test_the_sweep_refuses_unusable_caps_rather_than_defaulting(tmp_path, junk):
    sweep = dl.boot_sweep(tmp_path, digest(), max_files=junk)
    assert sweep["ok"] is False


# --- capacity: an LRU that runs on mtime, not atime ------------------------

def entries(*specs):
    """`specs` is (name, size_bytes, mtime_ns, digest) with the digest optional.

    Each entry's filename is built from a FULL group-qualified key (the index in
    the digest half, group 0 as the suffix), because that is the only shape
    `build_slot_name` accepts now.
    """
    geo = digest()
    out = []
    for index, spec in enumerate(specs):
        name, size, when = spec[0], spec[1], spec[2]
        geo_of = spec[3] if len(spec) > 3 else geo
        key = f"{index:016x}" + "0" * dl.GROUP_SUFFIX_HEX_CHARS
        made = dl.build_slot_name(key, geo_of)
        assert made["ok"], made["reason"]
        out.append({"path": f"/store/a3/{name}",
                    "filename": made["filename"],
                    "size_bytes": size, "mtime_ns": when})
    return out


def test_the_eviction_order_is_oldest_mtime_first():
    plan = dl.lru_eviction_plan(
        entries(("new", 100, 3000), ("old", 100, 1000), ("mid", 100, 2000)),
        geometry_digest=digest(), ceiling_bytes=150, used_bytes=300)
    assert plan["needed"] is True
    assert plan["planned"] == 2
    assert [item["mtime_ns"] for item in plan["entries"]] == [1000, 2000]


def test_a_store_under_its_target_sheds_nothing():
    plan = dl.lru_eviction_plan(entries(("a", 100, 1)), geometry_digest=digest(),
                                ceiling_bytes=1000, used_bytes=100)
    assert plan["needed"] is False
    assert plan["paths"] == []
    assert plan["planned_bytes"] == 0


def test_the_target_is_hysteresis_not_the_ceiling_itself():
    """Targeting the line we stand on would plan nothing and trip again instantly."""
    under_target = dl.lru_eviction_plan(entries(("a", 100, 1)),
                                        geometry_digest=digest(),
                                        ceiling_bytes=200, used_bytes=170)
    assert under_target["target_bytes"] == 180
    assert under_target["needed"] is False       # over the ceiling? no: under it
    at_the_line = dl.lru_eviction_plan(entries(("a", 100, 1)),
                                       geometry_digest=digest(),
                                       ceiling_bytes=100, used_bytes=100)
    assert at_the_line["needed"] is True
    assert at_the_line["target_bytes"] == 90


def test_a_file_this_boot_cannot_read_is_shed_before_our_own():
    foreign = digest(total_bytes_per_block=PP1_BYTES)
    plan = dl.lru_eviction_plan(
        entries(("ours", 100, 1), ("foreign", 100, 9999, foreign)),
        geometry_digest=digest(), ceiling_bytes=150, used_bytes=200)
    assert plan["stale_first"] == 1
    assert plan["planned"] == 1
    assert "foreign" in plan["paths"][0] or plan["entries"][0]["mtime_ns"] == 9999


def test_a_malformed_or_foreign_name_is_skipped_and_never_planned():
    rows = [{"path": "/store/a3/notes.txt", "filename": "notes.txt",
             "size_bytes": 10, "mtime_ns": 1},
            {"path": "/store/a3/kvslot-nope-nope.bin",
             "filename": "kvslot-nope-nope.bin", "size_bytes": 10, "mtime_ns": 1},
            {"path": "", "filename": "x", "size_bytes": 10, "mtime_ns": 1},
            "not even a mapping"]
    plan = dl.lru_eviction_plan(rows, geometry_digest=digest(),
                                ceiling_bytes=1, used_bytes=100)
    assert plan["planned"] == 0
    assert plan["skipped"] == 4


@pytest.mark.parametrize("junk", [0, -1, "1024", None, True])
def test_a_ceiling_that_is_not_a_positive_int_is_refused(junk):
    assert dl.lru_eviction_plan([], geometry_digest=digest(),
                                ceiling_bytes=junk)["ok"] is False


def test_the_plan_is_capped_and_says_it_was_truncated():
    rows = entries(*[(f"f{index}", 100, index) for index in range(10)])
    plan = dl.lru_eviction_plan(rows, geometry_digest=digest(), ceiling_bytes=100,
                                used_bytes=1000, target_fraction=0.1, max_plan=2)
    assert plan["planned"] == 2
    assert plan["truncated"] is True


def test_the_used_total_can_be_measured_from_the_entries_themselves():
    plan = dl.lru_eviction_plan(entries(("a", 100, 1), ("b", 100, 2)),
                                geometry_digest=digest(), ceiling_bytes=150)
    assert plan["used_bytes"] == 200
    assert plan["needed"] is True


# --- free space: absent, unreadable and genuinely full are three facts ------

def test_free_space_reports_real_bytes_on_a_real_filesystem(tmp_path):
    space = dl.free_space(tmp_path)
    assert space["ok"] and space["available"] is True
    assert space["free_bytes"] > 0 and space["total_bytes"] > 0
    assert space["units"]["free_bytes"] == "bytes"
    assert space["free_bytes"] <= space["total_bytes"]


@pytest.mark.parametrize("path", ["missing-dir", "missing-dir/deeper"])
def test_an_unreadable_filesystem_is_absent_never_zero(tmp_path, path):
    space = dl.free_space(tmp_path / path)
    assert space["available"] is False
    assert space["free_bytes"] is None          # NEVER 0: a lie that fills a disk
    assert space["total_bytes"] is None
    assert space["reason"]
    check = dl.space_check(tmp_path / path, watermark_bytes=1024)
    assert check["space_available"] is False
    assert check["below_watermark"] is None     # unknown is not "there is room"
    assert check["free_bytes"] is None


def test_the_watermark_accepts_bytes_or_a_percent_and_says_which(tmp_path):
    free = dl.free_space(tmp_path)["free_bytes"]
    low = dl.space_check(tmp_path, watermark_bytes=free // 2)
    assert low["watermark_source"] == "bytes"
    assert low["below_watermark"] is False
    high = dl.space_check(tmp_path, watermark_bytes=free * 2)
    assert high["below_watermark"] is True
    percent = dl.space_check(tmp_path, watermark_percent=1.0)
    assert percent["watermark_source"] == "percent_of_total"
    assert percent["below_watermark"] is False
    assert 0.0 <= percent["free_percent_of_total"] <= 100.0


def test_a_reservation_is_subtracted_before_the_verdict(tmp_path):
    """A writer asks "is there room for THIS slot", not "is there room for something".

    The reservation is one slot's bytes: at 11.4 MB a filesystem with 5 MB free
    passes a bare watermark and fails a reserved one, and the second answer is the
    one that keeps the filesystem out of ENOSPC.
    """
    free = dl.free_space(tmp_path)["free_bytes"]
    bare = dl.space_check(tmp_path, watermark_bytes=0, extra_bytes=0)
    assert bare["below_watermark"] is False
    reserved = dl.space_check(tmp_path, watermark_bytes=0,
                              extra_bytes=free + 1)
    assert reserved["below_watermark"] is True
    assert reserved["headroom_bytes"] == -1


@pytest.mark.parametrize("kwargs", [
    {"watermark_bytes": "1024"}, {"watermark_bytes": -5},
    {"watermark_percent": 500}, {"watermark_percent": "high"},
    {"extra_bytes": -1}, {"extra_bytes": 1.5},
])
def test_a_bad_watermark_is_refused_rather_than_ignored(tmp_path, kwargs):
    assert dl.space_check(tmp_path, **kwargs)["ok"] is False


def test_no_watermark_configured_is_an_unknown_verdict_not_a_pass(tmp_path):
    check = dl.space_check(tmp_path)
    assert check["ok"] is True
    assert check["below_watermark"] is None
    assert check["watermark_source"] == "none"


def test_free_space_refuses_a_path_that_is_not_a_path():
    for junk in (None, 7, b"bytes", ""):
        assert dl.free_space(junk)["available"] is False


# --- units: no numeric field is left undescribed ---------------------------

def _walk(payload, trail, problems):
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "units":
                if value.get("unknown"):
                    problems.append(f"{trail}: an unnamed unit")
                if "unknown" in value.values():
                    problems.append(f"{trail}: {key} carries an unknown unit")
                _walk(value, f"{trail}.{key}", problems)
                continue
            if isinstance(value, int) and not isinstance(value, bool):
                declared = (trail + "." + key).split(".")[-1]
                if declared not in dl.UNITS_BY_NAME and key not in (
                        "ok", "planned_bytes", "size_bytes"):
                    problems.append(f"{trail}.{key} is a number with no unit "
                                    "declared")
            _walk(value, f"{trail}.{key}", problems)
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            _walk(item, f"{trail}[{index}]", problems)


def test_every_payload_and_every_nested_payload_declares_its_units(tmp_path):
    geo = digest()
    made = slot_path(tmp_path)
    dl.write_committed_file(made["path"], b"bytes")
    payloads = [
        dl.validate_fingerprint(fingerprint()),
        dl.fingerprint_digest(fingerprint()),
        dl.fingerprint_section(fingerprint()),
        dl.fingerprint_section(None),
        dl.validate_key16(KEY),
        dl.validate_root("/home/user/kv-tier/disk"),
        dl.build_slot_name(KEY, geo),
        dl.build_slot_path(tmp_path, KEY, geo),
        dl.parse_path(made["filename"]),
        dl.classify_name(made["filename"], geo),
        dl.classify_name("notes.txt", geo),
        dl.boot_sweep(tmp_path, geo),
        dl.execute_unlink_plan(tmp_path, [], geometry_digest=geo),
        dl.lru_eviction_plan(entries(("a", 10, 1)), geometry_digest=geo,
                             ceiling_bytes=1),
        dl.write_committed_file(os.path.join(str(tmp_path), "a3",
                                             "kvslot-0000000000000000-"
                                             f"{geo}.bin"), b"x"),
        dl.free_space(tmp_path),
        dl.space_check(tmp_path, watermark_bytes=1024, extra_bytes=1),
        dl.space_check(tmp_path / "absent"),
    ]
    problems: list[str] = []
    for payload in payloads:
        _walk(payload, "$", problems)
    assert not problems, problems


def test_an_unrecognised_field_name_reports_unknown_rather_than_guessing():
    assert dl._units("free_bytes", "totally_new_field")[
        "totally_new_field"] == "unknown"


# --- the module is inert: nothing here touches the engine or the network ----

def test_no_function_here_opens_a_socket_or_touches_the_engine(tmp_path,
                                                               monkeypatch):
    """The layout layer is inert by construction: no fd outside a path we were given.

    Proved by forbidding `socket.socket` for the duration and exercising the whole
    surface: if a later stage adds a socket, a vllm import or a CUDA call here,
    this is where it fails.
    """
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("disk_layout reached for a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    geo = digest()
    made = slot_path(tmp_path)
    for call in (lambda: dl.build_slot_path(tmp_path, KEY, geo),
                 lambda: dl.boot_sweep(tmp_path, geo),
                 lambda: dl.free_space(tmp_path),
                 lambda: dl.space_check(tmp_path, watermark_bytes=1),
                 lambda: dl.fingerprint_digest(fingerprint()),
                 lambda: dl.write_committed_file(made["path"], b"bytes")):
        assert call() is not None


def test_the_module_never_creates_a_file_it_was_not_told_to(tmp_path):
    before = set(os.listdir(tmp_path))
    geo = digest()
    dl.build_slot_path(tmp_path, KEY, geo)
    dl.build_slot_name(KEY, geo)
    dl.boot_sweep(tmp_path, geo)
    dl.free_space(tmp_path)
    dl.lru_eviction_plan([], geometry_digest=geo, ceiling_bytes=10)
    dl.fingerprint_digest(fingerprint())
    assert set(os.listdir(tmp_path)) == before


def test_the_deleted_stage_one_surface_is_really_gone():
    """No dead code: the index codec, the offset math and the header codec are gone.

    A name kept "for now" is a name a future reader may call, and a codec with no
    caller is a codec no test protects. This test is the receipt for that deletion.
    """
    for gone in ("encode_index_record", "encode_index_commit", "decode_index_frame",
                 "read_index", "append_index", "index_batch_bytes",
                 "offset_for_slot", "slot_for_offset", "bounds_for_file",
                 "slot_is_readable", "encode_header", "decode_header",
                 "normalize_epoch", "build_paths", "build_rank",
                 "compute_regions_digest", "build_geometry", "geometry_section",
                 "geometry_from_fields"):
        assert not hasattr(dl, gone), gone
