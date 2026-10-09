#!/usr/bin/env bash
# =============================================================================
# apply-kvtier-patches.sh - apply, check and reverse the vLLM patch series in
# this directory against a LOCAL vLLM checkout. No SSH, no box, no engine.
#
#   status              read-only: per-patch verdicts, base-tree check, current tree
#   apply               bring the checkout up to the whole series (idempotent)
#   revert              take the whole series back off (reverse-apply, in order)
#   help                this text
#
# WHY THIS LIVES HERE AND NOT NEXT TO THE TRANSPORT
# The four hunks in `series` are the canonical artifact: they are what the engine
# source has to look like for the host-RAM KV tier to work, and they are a
# property of THIS repo's design. What is a property of the llm-api repo is the
# trip to the GPU box - `config/box/kvtier-patch.sh` reads the very same
# directories (see patches/README.md) and adds the SSH, the marker, the backups
# and the engine-interpreter verification. Two appliers, one set of bytes: this
# one for a checkout on the machine you are standing at, that one for the box.
#
# THE REFUSAL MODEL, which is the whole point of the tool
# Every patch directory ships a `patch.manifest` carrying four hashes: the patch
# file's own sha256, the target file's sha256 BEFORE (`file_sha_pre`) and AFTER
# (`file_sha_post`), and the per-method fingerprints. A verdict is only ever
# computed FROM those hashes, never inferred from "the hunk looked like it
# fitted":
#
#   * A patch file that does not hash to its manifest's `patch_sha256` is refused
#     before anything else is read. Both hashes are printed. This is the same
#     rule `mamba_anchor.py` and `backport.py` apply to a source they are about
#     to patch: a stale fingerprint is a REFUSAL, not a hint - and here, unlike
#     at runtime, there is no tier to keep serving, so the only right answer is
#     to stop and say which hash moved.
#   * A target file that is neither a patch's pre-image nor any post-image in its
#     chain is REFUSED with three hashes printed: what is installed, what this
#     patch expects to be given, and what it would have produced. The tool does
#     not "try with fuzz", does not try `-3`, and has NO --force, for the reason
#     kvtier-patch.sh states: it tracks a patch plus fingerprints, so the only
#     post-image it can prove is the one `git apply` reproduces from the
#     pre-image. Forcing could only install bytes this repo cannot vouch for.
#   * Every build is checked against `file_sha_post` BEFORE it is installed. A
#     manifest whose patch does not reproduce its own post-image aborts with both
#     hashes and writes nothing at all.
#
# IDEMPOTENCE, AND WHY IT IS NOT "APPLY IT AGAIN"
# Each target file has a CHAIN of post-images (0001's post is 0002's pre, etc.).
# The tool finds the highest post-image the file already carries, reports every
# patch up to and including it as ALREADY (subsumed by the chain), and applies
# only the rest. So a fully-patched checkout says "nothing to do" per patch
# rather than double-applying - and a checkout that has the FIRST three hunks but
# not the fourth gets exactly the fourth, which is the state a half-finished
# bisect actually leaves you in.
#
# WHAT IT PRINTS AT THE END
# Per-patch verdicts, then the checkout's HEAD commit and HEAD tree, then the
# sha256 of each patched file, then the TREE the working copy would have if the
# patched files were committed - computed in a scratch index
# (GIT_INDEX_FILE + read-tree + update-index + write-tree), so your own index and
# working tree are never touched. `--expect-tree <hash>` turns that into a
# verdict: this is how you prove a reconstruction IS the box
# (`live_tree` in base.manifest).
#
# SAFETY
#   * Never boots, restarts, stops or signals anything, and never sends a request.
#     A patched file is INERT: the running engine keeps the code it imported.
#   * Never writes outside the checkout's tracked target files, and leaves no
#     backup, no .pyc and no index change behind. Originals are held in a temp
#     dir and restored if a later patch in the same run fails.
#   * `git hash-object -w` (used to compute the resulting tree) writes loose
#     objects into the checkout's object store. That is the only write outside the
#     target files; it creates no ref, touches no branch, and is removed by the
#     repository's own gc.
#
# Exit codes: 0 up-to-date / done / verified | 1 would-change (status: not
#             applied, or a pending apply in dry-run) | 2 refusal or error.
# =============================================================================
set -uo pipefail

TOOL=${BASH_SOURCE[0]##*/}
TOOL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

EXIT_OK=0; EXIT_PENDING=1; EXIT_ERR=2
SERIES_DIR=${KVTIER_PATCH_SERIES_DIR:-$TOOL_DIR}
CHECKOUT=${KVTIER_VLLM_CHECKOUT:-}
DRY=0
EXPECT_TREE=""
VERB=""
TMP=""

# ---- argument parsing -------------------------------------------------------
# Options may appear anywhere; the positional arguments are "the checkout, then
# the verb", with either omitted piece defaulted (a leading non-verb names the
# checkout, mirroring kvtier-patch.sh's "a leading non-verb names the thing", so
# muscle memory transfers between the two tools).
declare -a POS=()
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run)     DRY=1 ;;
    --checkout)    shift; [ $# -gt 0 ] || { echo "--checkout needs a value" >&2; exit "$EXIT_ERR"; }; CHECKOUT=$1 ;;
    --checkout=*)  CHECKOUT=${1#--checkout=} ;;
    --series)      shift; [ $# -gt 0 ] || { echo "--series needs a value" >&2; exit "$EXIT_ERR"; }; SERIES_DIR=$1 ;;
    --series=*)    SERIES_DIR=${1#--series=} ;;
    --expect-tree) shift; [ $# -gt 0 ] || { echo "--expect-tree needs a value" >&2; exit "$EXIT_ERR"; }; EXPECT_TREE=$1 ;;
    --expect-tree=*) EXPECT_TREE=${1#--expect-tree=} ;;
    -h|--help)     HELP=1 ;;
    --*)           printf '%s\n' "$TOOL: unknown option '$1'" >&2; exit "$EXIT_ERR" ;;
    *)             POS+=("$1") ;;
  esac
  shift
done
VERB=""
for a in ${POS[@]+"${POS[@]}"}; do
  case $a in
    status|apply|revert|help)
      [ -n "$VERB" ] && { printf '%s\n' "$TOOL: two verbs given ('$VERB' and '$a')" >&2; exit "$EXIT_ERR"; }
      VERB=$a ;;
    *)
      [ -z "$CHECKOUT" ] || { printf '%s\n' "$TOOL: two checkouts given ('$CHECKOUT' and '$a')" >&2; exit "$EXIT_ERR"; }
      CHECKOUT=$a ;;
  esac
done
[ -n "$VERB" ] || VERB=status
[ -n "${HELP:-}" ] || HELP=0
[ "$HELP" = 1 ] && VERB=help

usage() {
  cat <<EOF
${TOOL} - apply, check and reverse the vLLM patch series in ${SERIES_DIR}
          against a LOCAL vLLM checkout (no SSH; the box transport lives in the
          llm-api repo's config/box/kvtier-patch.sh, which reads these directories)

usage: ${TOOL} [<checkout-dir>] <command> [options]

  status              read-only: base-tree check, one verdict per patch, the
                      per-file sha256s, HEAD and the resulting tree hash
  apply               bring the checkout up to the whole series. Idempotent: a file
                      already carrying a patch's post-image reports ALREADY and is
                      left alone. --dry-run prints the verdicts and writes nothing.
  revert              take the series back off, chain order downward, verifying each
                      result against file_sha_pre. --dry-run available.
  help                this text

options
  --checkout DIR      the vLLM source checkout (or a leading positional argument,
                      or KVTIER_VLLM_CHECKOUT)
  --series DIR        where the series lives (default: this script's directory,
                      or KVTIER_PATCH_SERIES_DIR)
  --expect-tree HASH  fail unless the resulting tree hash equals HASH; the series
                      tip is ${LIVE_TREE:-?}
  --dry-run           print, write nothing

WHAT IS REFUSED, and what is not
  A patch file that does not hash to its own manifest's patch_sha256. A target
  file that is neither a patch's file_sha_pre nor any file_sha_post in its chain.
  A built image that does not hash to file_sha_post (or file_sha_pre when
  reverting) - that one aborts the whole run and undoes what it already wrote.
  Every refusal prints the hashes it compared. There is no --force, no fuzz and
  no -3: this repo ships a patch plus fingerprints, so the only post-image it can
  prove is the one 'git apply' reproduces from the pre-image.

exit codes: 0 up-to-date / done / verified | 1 would-change | 2 refusal or error
EOF
}

note() { printf '%s\n' "$*"; }
act()  { printf '   -> %s\n' "$*"; }
err()  { printf '%s\n' "$TOOL: $*" >&2; }
die()  { err "$*"; exit "$EXIT_ERR"; }
cleanup() { [ -n "${TMP:-}" ] && [ -d "${TMP:-}" ] && rm -rf -- "$TMP"; return 0; }
sha_of() { sha256sum -- "$1" | awk '{print $1}'; }

# ---- the series, and the base it sits on ------------------------------------
SERIES_FILE=${SERIES_DIR}/series
BASE_FILE=${SERIES_DIR}/base.manifest
[ -f "$SERIES_FILE" ] || die "no series file at ${SERIES_FILE}
  This script is the applier for the patch series that ships IN THIS REPO, under
  patches/. It expects to be run from that directory (it is at ${TOOL_DIR}); if
  you copied it elsewhere, point it at the series with --series DIR or
  KVTIER_PATCH_SERIES_DIR=DIR."
[ -f "$BASE_FILE" ] || die "no base manifest at ${BASE_FILE} - the series directory
  at ${SERIES_DIR} is not the one from this repo (it must carry base.manifest,
  series, README.md and the NNNN-* patch directories)."
# shellcheck disable=SC1090
set -a; . "$BASE_FILE"; set +a

BASE_TREE=${base_tree:-}
LIVE_TREE=${live_tree:-}

# help needs no checkout and no patch loading, only the series it prints.
if [ "$VERB" = help ]; then usage; exit "$EXIT_OK"; fi

declare -a PDIRS=()
while IFS= read -r line; do
  line=${line%%#*}
  line=$(printf '%s' "$line" | tr -d '[:space:]')
  [ -n "$line" ] && PDIRS+=("$line")
done < "$SERIES_FILE"
[ "${#PDIRS[@]}" -gt 0 ] || die "series file ${SERIES_FILE} lists no patch directories"

# Order is enforced HERE, twice over: the numbering must be 0001..NN in the exact
# order the file lists, and every numbered directory on disk must be listed. A
# directory that exists but is not in the series is the way a half-patched engine
# happens silently.
declare -A SEEN=()
i=0
for d in "${PDIRS[@]}"; do
  i=$((i + 1))
  want=$(printf '%04d' "$i")
  case $d in
    "$want-"*) : ;;
    *) die "series order is broken: entry ${i} is '${d}', which does not start with ${want}- (the series must be numbered 0001, 0002, ... with no gaps and in the order it applies)" ;;
  esac
  [ -d "${SERIES_DIR}/${d}" ] || die "series lists '${d}' but ${SERIES_DIR}/${d} does not exist"
  SEEN[$d]=1
done
for d in "${SERIES_DIR}"/[0-9][0-9][0-9][0-9]-*/; do
  [ -d "$d" ] || continue
  b=${d%/}; b=${b##*/}
  [ -n "${SEEN[$b]:-}" ] || die "'${b}' is a patch directory in ${SERIES_DIR} that the series does not list. Add it to ${SERIES_FILE} in the position it must apply, or remove it: an unlisted patch is a patch nobody applies."
done

# ---- load every patch's manifest and verify the artifact ITSELF -------------
declare -a P_ID=() P_TITLE=() P_REL=() P_PRE=() P_POST=() P_PATCH=() P_STAT=()
N=${#PDIRS[@]}
for i in $(seq 0 $((N - 1))); do
  d=${SERIES_DIR}/${PDIRS[$i]}
  mf=${d}/patch.manifest
  [ -f "$mf" ] || die "no patch.manifest in ${d}"
  # shellcheck disable=SC1090
  set -a; . "$mf"; set +a
  for k in patch_id title git_rel file_sha_pre file_sha_post patch_file patch_sha256; do
    v=${!k:-}
    [ -n "$v" ] || die "${mf} does not set ${k} - refusing to apply a patch whose manifest is incomplete"
  done
  [ "$file_sha_pre" != "$file_sha_post" ] || die "${mf}: file_sha_pre == file_sha_post, which cannot describe a change"
  pp=${d}/${patch_file}
  [ -f "$pp" ] || die "${mf} names patch_file=${patch_file}, which is not in ${d}"
  got=$(sha_of "$pp")
  if [ "$got" != "$patch_sha256" ]; then
    err "REFUSED: ${PDIRS[$i]}/${patch_file} is not the patch this manifest describes."
    err "  on disk   sha256=${got}"
    err "  manifest  sha256=${patch_sha256}"
    err "  A stale fingerprint is a refusal, not a hint: the tool will not apply bytes
  it cannot tie to a manifest, and it will not "fix" the manifest either. Re-make the
  patch against the file that actually exists, re-pin every hash in the manifest, and
  commit both together."
    exit "$EXIT_ERR"
  fi
  # The diff must be a -p1 diff of exactly the path the manifest patches: a patch
  # aimed at another file would apply to a different tree and be believed.
  ptext=$(cat -- "$pp")
  case $ptext in
    *"--- a/${git_rel}"*"+++ b/${git_rel}"*) : ;;
    *) die "${mf}: patch_file=${patch_file} is not a -p1 diff of '${git_rel}' - the manifest and the patch disagree about what is being patched" ;;
  esac
  P_ID+=("$patch_id"); P_TITLE+=("$title"); P_REL+=("$git_rel")
  P_PRE+=("$file_sha_pre"); P_POST+=("$file_sha_post"); P_PATCH+=("$pp")
  P_STAT+=("+${patch_added:-?}/-${patch_removed:-?}, ${patch_hunks:-?} hunks")
done

# ---- the checkout -----------------------------------------------------------
[ -n "$CHECKOUT" ] || die "no vLLM checkout given.
  usage: ${TOOL} <checkout-dir> <status|apply|revert> [--dry-run] [--expect-tree HASH]
  e.g.   ${TOOL} ~/vllm-workspace status
  (KVTIER_VLLM_CHECKOUT=... works too)"
CHECKOUT=$(cd -- "$CHECKOUT" 2>/dev/null && pwd) || die "cannot enter the checkout at ${CHECKOUT}"
git -C "$CHECKOUT" rev-parse --git-dir >/dev/null 2>&1 \
  || die "${CHECKOUT} is not a git checkout. The series is tracked against a
  specific TREE (${BASE_TREE:-unknown}) and this tool reports the tree it produces,
  so it needs HEAD: point it at the vLLM source checkout the engine imports
  (an editable install's checkout), not at an installed site-packages directory."

HEAD_COMMIT=$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null || printf 'unknown')
HEAD_TREE=$(git -C "$CHECKOUT" rev-parse 'HEAD^{tree}' 2>/dev/null || printf 'unknown')

# The resulting-tree computation substitutes the working copy of each target file
# into a scratch index built from HEAD. The caller's index is never opened.
compute_tree() {
  local idx mode out
  idx=$(mktemp) || return 1
  if ! GIT_INDEX_FILE=$idx git -C "$CHECKOUT" read-tree HEAD >/dev/null 2>&1; then
    rm -f -- "$idx"; return 1
  fi
  for i in $(seq 0 $((N - 1))); do
    rel=${P_REL[$i]}
    [ -f "${CHECKOUT}/${rel}" ] || continue
    mode=$(git -C "$CHECKOUT" ls-tree HEAD -- "$rel" | awk '{print $1}')
    [ -n "$mode" ] || mode=100644
    blob=$(git -C "$CHECKOUT" hash-object -w -- "${CHECKOUT}/${rel}") || { rm -f -- "$idx"; return 1; }
    GIT_INDEX_FILE=$idx git -C "$CHECKOUT" update-index --cacheinfo "$mode,$blob,$rel" \
      || { rm -f -- "$idx"; return 1; }
  done
  out=$(GIT_INDEX_FILE=$idx git -C "$CHECKOUT" write-tree 2>/dev/null) || { rm -f -- "$idx"; return 1; }
  rm -f -- "$idx"
  printf '%s' "$out"
}

# ---- per-target chain state -------------------------------------------------
# A target file is addressed by a numeric index (T_INDEX maps the repo-relative
# path to it): one file carries a CHAIN of patches, and the chain is what makes
# 'already applied' a computable fact instead of a guess.
declare -A T_INDEX=()
declare -a T_PATH=() T_PATCHES=()
for i in $(seq 0 $((N - 1))); do
  rel=${P_REL[$i]}
  if [ -z "${T_INDEX[$rel]:-}" ]; then
    n=${#T_PATH[@]}
    T_INDEX[$rel]=$n; T_PATH+=("$rel"); T_PATCHES[$n]=""
  fi
  t=${T_INDEX[$rel]}
  T_PATCHES[$t]="${T_PATCHES[$t]} $i"
done

CUR=()      # current sha256 of each target file ("" = missing)
COVER=()    # highest patch index whose post-image the file already carries, else -1
VIRT=()     # the hash a SEQUENTIAL walk expects the file to hold at this point
REAL=()     # 1 = VIRT is the file's actual hash; 0 = it is a walk prediction
for t in $(seq 0 $(( ${#T_PATH[@]} - 1 ))); do
  rel=${T_PATH[$t]}
  if [ -f "${CHECKOUT}/${rel}" ]; then CUR[$t]=$(sha_of "${CHECKOUT}/${rel}"); else CUR[$t]=""; fi
  COVER[$t]=-1
  for i in ${T_PATCHES[$t]}; do
    [ -n "${CUR[$t]}" ] && [ "${CUR[$t]}" = "${P_POST[$i]}" ] && COVER[$t]=$i
  done
done

# ---- the refusal text, which is the deliverable -----------------------------
refuse_absent() { # $1 = patch index
  local i=$1
  err "REFUSED ${PDIRS[$i]} (${P_ID[$i]}): the checkout has no ${P_REL[$i]}"
  err "  expected pre  sha256=${P_PRE[$i]}
  expected post sha256=${P_POST[$i]}
  The file this patch was made for is not there. This is the fork moving (or the
  wrong checkout): refusing to guess, and refusing to create it."
}
refuse_state() { # $1 = patch index, $2 = the hash the walk actually compared
  local i=$1 t=${T_INDEX[${P_REL[$i]}]} compared=${2:-}
  [ -n "$compared" ] || compared=${CUR[$t]:-missing}
  err "REFUSED ${PDIRS[$i]} (${P_ID[$i]}): ${P_REL[$i]} is not the file this patch was made for"
  err "  compared    sha256=${compared}   (the state the series walk expected to be standing in)"
  err "  installed   sha256=${CUR[$t]:-missing}"
  err "  this patch expects to be given  sha256=${P_PRE[$i]}  (file_sha_pre)"
  err "  this patch would produce        sha256=${P_POST[$i]}  (file_sha_post)"
  err "  so it is neither 'ready to apply' nor 'already applied'. Refusing to guess:
  no fuzz, no -3, no --force. Either the base moved off ${BASE_TREE:-?} (see
  patches/README.md for what these hunks sit on), or an untracked edit or a
  different patch is already on this file. Look with 'status', or:
      git -C ${CHECKOUT} diff -- ${P_REL[$i]}"
}

# ---- the write path, shared by apply and revert -----------------------------
ORIG=()   # temp copies of the originals, for one-shot undo of a failed run
restore_origins() {
  local t
  for t in $(seq 0 $(( ${#T_PATH[@]} - 1 ))); do
    [ -f "${ORIG[$t]:-}" ] || continue
    cp -- "${ORIG[$t]}" "${CHECKOUT}/${T_PATH[$t]}" \
      && note "   [undo] restored ${T_PATH[$t]} from the copy taken before this run" \
      || err "could not restore ${T_PATH[$t]} from ${ORIG[$t]} - restore it by hand: git -C ${CHECKOUT} checkout -- ${T_PATH[$t]}"
  done
}
save_origin() { # $1 = target index
  local t=$1 safe
  [ -n "${ORIG[$t]:-}" ] && return 0
  safe=${T_PATH[$t]//\//_}
  cp -p -- "${CHECKOUT}/${T_PATH[$t]}" "${TMP}/orig/${safe}" || die "could not snapshot ${T_PATH[$t]}"
  ORIG[$t]="${TMP}/orig/${safe}"
}

# install_builtin: build the post-image in a scratch tree, HASH it against the
# manifest, and only then put it in place. Never installs bytes it cannot name.
apply_one() { # $1 = patch index, $2 = "forward"|"reverse"
  local i=$1 dir=$2 rel=${P_REL[$i]} t=${T_INDEX[${P_REL[$i]}]}
  local want got why out
  rel=${P_REL[$i]}
  out="${TMP}/work/${PDIRS[$i]}"
  mkdir -p "${out}/a/$(dirname -- "$rel")" || return 1
  cp -- "${CHECKOUT}/${rel}" "${out}/a/${rel}" || return 1
  if [ "$dir" = forward ]; then want=${P_POST[$i]}; else want=${P_PRE[$i]}; fi
  if command -v git >/dev/null 2>&1; then
    if [ "$dir" = forward ]; then
      why=$( cd "${out}/a" && git apply --check -p1 "${P_PATCH[$i]}" 2>&1 ) || {
        err "REFUSED ${PDIRS[$i]}: git apply --check rejected the tracked patch:"
        printf '%s\n' "$why" | sed 's/^/      /' >&2
        return 1
      }
      ( cd "${out}/a" && git apply -p1 "${P_PATCH[$i]}" ) || { err "${PDIRS[$i]}: git apply failed after --check passed"; return 1; }
    else
      why=$( cd "${out}/a" && git apply --check -R -p1 "${P_PATCH[$i]}" 2>&1 ) || {
        err "REFUSED ${PDIRS[$i]}: git apply -R --check rejected the tracked patch:"
        printf '%s\n' "$why" | sed 's/^/      /' >&2
        return 1
      }
      ( cd "${out}/a" && git apply -R -p1 "${P_PATCH[$i]}" ) || { err "${PDIRS[$i]}: git apply -R failed after --check passed"; return 1; }
    fi
  elif command -v patch >/dev/null 2>&1; then
    if [ "$dir" = forward ]; then
      why=$( cd "${out}/a" && patch -s -p1 < "${P_PATCH[$i]}" 2>&1 ) || { err "REFUSED ${PDIRS[$i]}: patch -p1 said: ${why}"; return 1; }
    else
      why=$( cd "${out}/a" && patch -s -R -p1 < "${P_PATCH[$i]}" 2>&1 ) || { err "REFUSED ${PDIRS[$i]}: patch -R -p1 said: ${why}"; return 1; }
    fi
  else
    err "neither git nor patch is available locally"
    return 1
  fi
  got=$(sha_of "${out}/a/${rel}")
  if [ "$got" != "$want" ]; then
    err "REFUSED ${PDIRS[$i]}: the patch applied, but the result is not the image the manifest names."
    err "  built       sha256=${got}"
    err "  manifest    sha256=${want}"
    err "  Nothing from this patch was installed. The patch and its manifest disagree -
  re-generate the patch and re-pin the manifest together."
    return 1
  fi
  save_origin "$t"
  cp -- "${out}/a/${rel}" "${CHECKOUT}/${rel}" || { err "could not write ${CHECKOUT}/${rel}"; return 1; }
  got=$(sha_of "${CHECKOUT}/${rel}")
  [ "$got" = "$want" ] || { err "installed ${rel} does not hash to ${want}"; return 1; }
  CUR[$t]=$got
  return 0
}

# ---- verbs ------------------------------------------------------------------
header() {
  note "== vLLM patch series at ${SERIES_DIR}"
  note "   base      ${base_repo:-?} ${base_version:-?} @ ${base_commit:-?}"
  note "             + ${series_patches:-?} patches from ${series_repo:-?} (${series_dir:-?}), built by ${series_builder:-?}"
  note "             base tree ${BASE_TREE:-?}"
  note "   checkout  ${CHECKOUT}"
  note "   HEAD      ${HEAD_COMMIT}  (tree ${HEAD_TREE})"
  if [ -n "$BASE_TREE" ] && [ "$HEAD_TREE" != "unknown" ] && [ "$HEAD_TREE" != "$BASE_TREE" ]; then
    note "   WARNING   the checkout's tree is NOT the recorded base tree:"
    note "               checkout ${HEAD_TREE}"
    note "               recorded ${BASE_TREE}"
    note "             The per-file fingerprints below are still the real gate, so a"
    note "             clean run here means these hunks landed exactly - but it no"
    note "             longer means the tree underneath them is the one this series was"
    note "             made against. That difference is yours to explain, not the tool's."
  elif [ -n "$BASE_TREE" ] && [ "$HEAD_TREE" = "$BASE_TREE" ]; then
    note "             [ok] the checkout sits exactly on the recorded base tree"
  fi
  note "   patches   ${N} (${PDIRS[*]})"
  note ""
}

# Walk the series forward and print one verdict per patch. Sets PENDING/REFUSED;
# on a refusal it stops, because the next patch's pre-image cannot exist either
# and printing more verdicts would be a guess.
PENDING=0
REFUSED=0
declare -a VERDICT=()
walk() { # forward pass: $1 = "status" | "apply"
  local mode=$1 i t rel
  VERDICT=(); PENDING=0; REFUSED=0
  for t in $(seq 0 $(( ${#T_PATH[@]} - 1 ))); do
    VIRT[$t]=${CUR[$t]}; REAL[$t]=1
  done
  for i in $(seq 0 $((N - 1))); do
    t=${T_INDEX[${P_REL[$i]}]}; rel=${P_REL[$i]}   # t = the numeric target index
    if [ -z "${VIRT[$t]}" ]; then
      VERDICT+=("ABSENT   ${PDIRS[$i]}  ${P_ID[$i]}  (no ${rel} in the checkout)")
      refuse_absent "$i"; REFUSED=$((REFUSED + 1)); return 1
    fi
    if [ "$i" -le "${COVER[$t]}" ]; then
      VERDICT+=("ALREADY  ${PDIRS[$i]}  ${P_ID[$i]}  (${P_STAT[$i]} - its post-image is already on ${rel}, subsumed by the chain)")
      continue
    fi
    if [ "${REAL[$t]}" = 1 ] && [ "${VIRT[$t]}" = "${P_POST[$i]}" ]; then
      VERDICT+=("ALREADY  ${PDIRS[$i]}  ${P_ID[$i]}  (${P_STAT[$i]} - ${rel} is already its post-image)")
      continue
    fi
    if [ "${VIRT[$t]}" = "${P_PRE[$i]}" ]; then
      # A chain means the file CANNOT carry patch i's pre-image until the patch
      # before it lands. So a walk describes the state a sequential run would be
      # in, not a snapshot: VIRT carries the expected hash forward, and a status
      # run reports each later link as PENDING instead of refusing on a file that
      # is merely not patched yet.
      if [ "$mode" = status ] || [ "$DRY" = 1 ]; then
        VERDICT+=("PENDING  ${PDIRS[$i]}  ${P_ID[$i]}  ${rel}  (${P_STAT[$i]})")
        PENDING=$((PENDING + 1)); VIRT[$t]=${P_POST[$i]}; REAL[$t]=0
        continue
      fi
      VERDICT+=("PENDING  ${PDIRS[$i]}  ${P_ID[$i]}  ${rel}  (${P_STAT[$i]})")
      PENDING=$((PENDING + 1))
      if apply_one "$i" forward; then
        VERDICT[$(( ${#VERDICT[@]} - 1 ))]="APPLIED  ${PDIRS[$i]}  ${P_ID[$i]}  ${rel} = file_sha_post (${P_STAT[$i]})"
        VIRT[$t]=${P_POST[$i]}; REAL[$t]=1; COVER[$t]=$i
      else
        VERDICT[$(( ${#VERDICT[@]} - 1 ))]="FAILED   ${PDIRS[$i]}  ${P_ID[$i]}"
        REFUSED=$((REFUSED + 1)); return 1
      fi
      continue
    fi
    VERDICT+=("REFUSE   ${PDIRS[$i]}  ${P_ID[$i]}")
    refuse_state "$i" "${VIRT[$t]}"; REFUSED=$((REFUSED + 1)); return 1
  done
  return 0
}

# Revert walks each target's chain DOWNWARD: you cannot take 0001 off before
# 0004, because 0001's post-image IS 0004's pre-image. The chain applies upward
# and unapplies downward; the other order would be a refusal on every file but
# the first. Files are independent of each other, so the per-file order is all
# that has to be right here.
declare -a TOP=() VCUR=()
walk_revert() {
  local t i rel
  VERDICT=(); PENDING=0; REFUSED=0
  for t in $(seq 0 $(( ${#T_PATH[@]} - 1 ))); do
    rel=${T_PATH[$t]}
    if [ -z "${CUR[$t]}" ]; then
      for i in ${T_PATCHES[$t]}; do
        VERDICT+=("ABSENT   ${PDIRS[$i]}  ${P_ID[$i]}  (no ${rel} in the checkout)")
        refuse_absent "$i"
      done
      REFUSED=$((REFUSED + 1)); return 1
    fi
    TOP[$t]=${COVER[$t]}
    # VCUR is the revert walk's notion of "what the file holds now": in a dry run
    # nothing is written, so a chain would refuse at its second link (0001's
    # pre-image only exists once 0002 has come off). A dry run must describe the
    # same sequence a real run performs, or --dry-run is a different command.
    VCUR[$t]=${CUR[$t]}
    for i in $(printf '%s\n' ${T_PATCHES[$t]} | sort -rn); do
      if [ "$i" -gt "${TOP[$t]}" ]; then
        VERDICT+=("CLEAN    ${PDIRS[$i]}  ${P_ID[$i]}  (${rel} carries neither this patch nor a later one - nothing to take off)")
        continue
      fi
      if [ "${VCUR[$t]}" = "${P_PRE[$i]}" ]; then
        VERDICT+=("CLEAN    ${PDIRS[$i]}  ${P_ID[$i]}  (${rel} is already this patch's pre-image - nothing to take off)")
        TOP[$t]=$((i - 1))
        continue
      fi
      if [ "${VCUR[$t]}" = "${P_POST[$i]}" ]; then
        VERDICT+=("TAKE-OFF ${PDIRS[$i]}  ${P_ID[$i]}  ${rel} -> file_sha_pre (${P_STAT[$i]})")
        PENDING=$((PENDING + 1))
        if [ "$DRY" = 1 ]; then
          VCUR[$t]=${P_PRE[$i]}; TOP[$t]=$((i - 1)); continue
        fi
        if apply_one "$i" reverse; then
          VERDICT[$(( ${#VERDICT[@]} - 1 ))]="REVERTED ${PDIRS[$i]}  ${P_ID[$i]}  ${rel} = file_sha_pre"
          VCUR[$t]=${CUR[$t]}; TOP[$t]=$((i - 1))
        else
          VERDICT[$(( ${#VERDICT[@]} - 1 ))]="FAILED   ${PDIRS[$i]}  ${P_ID[$i]}"
          REFUSED=$((REFUSED + 1)); return 1
        fi
        continue
      fi
      VERDICT+=("REFUSE   ${PDIRS[$i]}  ${P_ID[$i]}")
      refuse_state "$i" "${VCUR[$t]}"; REFUSED=$((REFUSED + 1)); return 1
    done
  done
  return 0
}

report_verdicts() {
  note "   per-patch verdicts:"
  if [ "${#VERDICT[@]}" = 0 ]; then note "     (none)"; fi
  for v in "${VERDICT[@]}"; do note "     ${v}"; done
  note ""
}

report_result() { # the final hashes, and the --expect-tree verdict
  local t tree rc=0
  note "   final hashes"
  for t in $(seq 0 $(( ${#T_PATH[@]} - 1 ))); do
    note "     ${T_PATH[$t]}  sha256=${CUR[$t]:-missing}"
  done
  note "   HEAD       ${HEAD_COMMIT}"
  note "   HEAD tree  ${HEAD_TREE}"
  tree=$(compute_tree) || { err "could not compute the working-tree tree hash"; return "$EXIT_ERR"; }
  note "   tree now   ${tree}   (HEAD with these files substituted; your index was not touched)"
  if [ -n "$LIVE_TREE" ]; then
    if [ "$tree" = "$LIVE_TREE" ]; then
      note "             [= ${LIVE_TREE}: this checkout's source IS the box's tree]"
    else
      note "             (base.manifest's live_tree is ${LIVE_TREE} - this tree is not it)"
    fi
  fi
  if [ -n "$EXPECT_TREE" ]; then
    if [ "$tree" = "$EXPECT_TREE" ]; then
      note "   EXPECTED   PASS  tree ${tree} == ${EXPECT_TREE}"
    else
      err "EXPECTED FAIL: tree ${tree} != ${EXPECT_TREE}"
      rc=$EXIT_ERR
    fi
  fi
  return "$rc"
}

cmd_status() {
  header
  walk status || true
  report_verdicts
  report_result || return "$EXIT_ERR"
  note ""
  if [ "$REFUSED" -gt 0 ]; then
    note "RESULT  REFUSE: ${REFUSED} patch(es) cannot be explained by their fingerprints. Read the hashes above."
    return "$EXIT_ERR"
  fi
  if [ "$PENDING" -gt 0 ]; then
    note "RESULT  ${PENDING} patch(es) would apply: ${TOOL} ${CHECKOUT} apply"
    return "$EXIT_PENDING"
  fi
  note "RESULT  the series is fully applied; nothing to do"
  return "$EXIT_OK"
}

cmd_apply() {
  header
  if [ "$DRY" = 1 ]; then
    note "   DRY RUN - nothing will be written."
    walk status || true
    report_verdicts
    note "   planned: write ${PENDING} file state change(s) in series order, verifying"
    note "   every built image against its manifest's file_sha_post before installing it."
    note "   Nothing is started, stopped or restarted by this tool at any point; a"
    note "   patched file is INERT until the operator boots the engine."
    [ "$REFUSED" -gt 0 ] && { note "RESULT  REFUSE: see the hashes above"; return "$EXIT_ERR"; }
    [ "$PENDING" -gt 0 ] && { note "RESULT  would apply ${PENDING} patch(es)"; return "$EXIT_PENDING"; }
    note "RESULT  nothing to do"; return "$EXIT_OK"
  fi
  walk apply
  report_verdicts
  if [ "$REFUSED" -gt 0 ]; then
    restore_origins
    note ""
    note "RESULT  REFUSED and UNDONE - this run installed nothing."
    return "$EXIT_ERR"
  fi
  report_result || { restore_origins; return "$EXIT_ERR"; }
  note ""
  if [ "$PENDING" = 0 ]; then
    note "APPLIED  nothing was written: every patch reported ALREADY (or the checkout"
    note "         was already at the series tip). Re-running apply is idempotent."
    return "$EXIT_OK"
  fi
  note "APPLIED  ${PENDING} patch state(s) written. The engine was NOT started, stopped"
  note "         or signalled by this tool, and no request was sent to it: a patched"
  note "         file is inert until the operator boots it."
  return "$EXIT_OK"
}

cmd_revert() {
  header
  if [ "$DRY" = 1 ]; then
    note "   DRY RUN - nothing will be written."
    walk_revert || true
    report_verdicts
    note "   planned: reverse-apply in reverse chain order, verifying each result"
    note "   against its manifest's file_sha_pre before installing it."
    [ "$REFUSED" -gt 0 ] && { note "RESULT  REFUSE: see the hashes above"; return "$EXIT_ERR"; }
    [ "$PENDING" -gt 0 ] && { note "RESULT  would take off ${PENDING} patch state(s)"; return "$EXIT_PENDING"; }
    note "RESULT  nothing to do"; return "$EXIT_OK"
  fi
  walk_revert
  report_verdicts
  if [ "$REFUSED" -gt 0 ]; then
    restore_origins
    note ""
    note "RESULT  REFUSED and UNDONE - this run installed nothing."
    return "$EXIT_ERR"
  fi
  report_result || { restore_origins; return "$EXIT_ERR"; }
  note ""
  note "REVERTED  to the recorded base tree where the chain allowed; the engine was"
  note "          not restarted, stopped or signalled by this tool."
  return "$EXIT_OK"
}



TMP=$(mktemp -d) || die "mktemp -d failed"
trap cleanup EXIT
mkdir -p "${TMP}/orig" "${TMP}/work"

case "$VERB" in
  status) cmd_status ;;
  apply)  cmd_apply ;;
  revert) cmd_revert ;;
esac
