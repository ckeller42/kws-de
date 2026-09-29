#!/usr/bin/env bash
# Sync the kws-data tree between this machine and a remote, then cross-check it
# by sha256 manifest (design §3). No host or path is baked in: the remote is an
# ssh target you pass, the local root comes from KWS_DATA_ROOT / --data-root /
# the same per-OS default kws_de.config resolves. Dry-run by default.
#
#   scripts/sync-data.sh --to <ssh-host>     # push local -> remote (migrate)
#   scripts/sync-data.sh --from <ssh-host>   # pull remote -> local (restore/backup)
#   add --run to actually transfer (default is --dry-run)
#   add --verify to sha256-manifest both ends and diff after a --run
#
# The remote path mirrors the local one's basename under the remote home
# (e.g. ~/kws-data), or pass --remote-root.
set -euo pipefail

die() { echo "sync-data: $*" >&2; exit 2; }

direction="" host="" run="--dry-run" verify=0 remote_root=""
data_root="${KWS_DATA_ROOT:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --to)          direction=push; host="${2:?--to needs an ssh host}"; shift 2;;
    --from)        direction=pull; host="${2:?--from needs an ssh host}"; shift 2;;
    --data-root)   data_root="${2:?}"; shift 2;;
    --remote-root) remote_root="${2:?}"; shift 2;;
    --run)         run=""; shift;;
    --dry-run)     run="--dry-run"; shift;;
    --verify)      verify=1; shift;;
    -h|--help)     sed -n '2,17p' "$0"; exit 0;;
    *)             die "unknown arg: $1";;
  esac
done
[ -n "$direction" ] || die "need --to <host> or --from <host>"

# Resolve the local root the same way kws_de.config does when unset.
if [ -z "$data_root" ]; then
  data_root="$(python3 -c 'from kws_de import config; print(config._DATA_ROOT)' 2>/dev/null)" \
    || die "no --data-root/KWS_DATA_ROOT and could not import kws_de.config"
fi
[ -d "$data_root" ] || die "local data root not a directory: $data_root"
remote_root="${remote_root:-$(basename "$data_root")}"   # ~/<basename> on the remote

# Trailing slashes: copy the CONTENTS of the root into the remote root.
local_spec="${data_root%/}/"
remote_spec="${host}:${remote_root%/}/"

echo "sync-data: $direction  local=$data_root  remote=$remote_spec  mode=${run:-RUN}"
common=(-a --partial --human-readable --stats --exclude='.DS_Store')
if [ "$direction" = push ]; then
  ssh "$host" "mkdir -p '${remote_root%/}'"
  rsync "${common[@]}" $run "$local_spec" "$remote_spec"
else
  rsync "${common[@]}" $run "$remote_spec" "$local_spec"
fi

if [ "$verify" = 1 ] && [ -z "$run" ]; then
  echo "sync-data: verifying by sha256 manifest…"
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  python3 -m kws_de.verify manifest "$data_root" -o "$tmp/local.manifest"
  ssh "$host" "cd '${remote_root%/}' && python3 -m kws_de.verify manifest . " > "$tmp/remote.manifest" \
    || die "remote manifest failed (is kws_de importable on $host?)"
  python3 -m kws_de.verify diff "$tmp/local.manifest" "$tmp/remote.manifest"
fi
