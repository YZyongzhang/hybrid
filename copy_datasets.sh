#!/usr/bin/env bash
# Copy the five selected directories using rsync; rerun to resume.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
用法：
  ./copy_datasets.sh                         # 交互输入 IP 和远程绝对路径
  ./copy_datasets.sh 用户名@IP /远程/绝对路径
  ./copy_datasets.sh --dry-run 用户名@IP /远程/绝对路径
  ./copy_datasets.sh --jump 中转用户名@中转IP:22 目标用户名@目标IP /远程/绝对路径
  ./copy_datasets.sh --dry-run --jump 中转用户名@中转IP:22 目标用户名@目标IP /远程/绝对路径
  ./copy_datasets.sh --check-connection --jump 中转用户名@中转IP:22 目标用户名@目标IP

目标目录结构（复制以下五个目录的完整内容）：
  <目标>/soundspaces/data/                    /hdd-data3/yongzhang/finnal_exp/data
  <目标>/hybrid_offline_RL_datasets/getuanhui/ /data2/getuanhui
  <目标>/hybrid_offline_RL_datasets/avnav-hybrid-mutiple/
                                             /data/Getuanhui/avnav-hybrid-mutiple
  <目标>/result/2026-10-28-avnav-codex_attemp/
  <目标>/result/2026-10-28-avnav-codex_attemp_rgb/
  <目标>/dataset_copy_sources.txt             本次复制的数据来源说明

保留原始 split、manifest 和 episode/scene ID；训练、验证、测试数据不得混用。

可选环境变量：SSH_PORT（默认 22）、REMOTE_USER（默认当前用户名）、
SSH_JUMP（中转机，格式为 [用户名@]主机[:端口]，也可使用 SSH 配置中的主机别名）、
SSH_CONNECT_TIMEOUT（连接超时秒数，默认 20）、
SOUNDSPACES_SOURCE、DATA2_SOURCE、AVNAV_SOURCE、ATTEMP_SOURCE、ATTEMP_RGB_SOURCE。
--jump 优先于 SSH_JUMP；省略中转端口时使用 SSH 配置或默认端口 22。
SSH_PORT 是目标服务器端口，中转机必须能通过 SSH 访问目标服务器。
中转机仅转发连接，无需安装 rsync 或存储数据；认证使用本机密钥或交互输入密码。
--dry-run 只显示复制计划，不连接远程机器。
--check-connection 逐段检查 SSH 和目标 rsync，不读取或复制数据，无需目标路径。
--verbose 输出 SSH 详细连接日志，可与传输或连接检查一起使用。
使用 rsync：本机需要 ssh 和 rsync，远程需要 SSH 服务和 rsync。
保留未完成文件；中断后重跑可续传，并跳过已完成且未变化的文件。
仅复制数据；新机器仍需准备项目代码和 Python/Habitat/SoundSpaces 环境。
EOF
}

die() { printf '错误：%s\n' "$*" >&2; exit 1; }
shell_quote() { printf "'%s'" "${1//\'/\'\\\'\'}"; }

dry_run=false
check_connection=false
verbose=false
jump_host=${SSH_JUMP:-}
while (( $# > 0 )); do
  case $1 in
    --help|-h) usage; exit 0 ;;
    --dry-run) dry_run=true; shift ;;
    --check-connection) check_connection=true; shift ;;
    --verbose) verbose=true; shift ;;
    --jump)
      (( $# >= 2 )) && [[ -n $2 && $2 != -* ]] || die '--jump 后需要提供 [用户名@]中转主机[:端口]。'
      jump_host=$2
      shift 2
      ;;
    --) shift; break ;;
    -*) die "未知选项：$1" ;;
    *) break ;;
  esac
done
(( $# <= 2 )) || die '最多提供用户名@IP 和目标路径两个参数。'
remote=${1:-}
destination=${2:-}
if $dry_run && $check_connection; then die '--dry-run 与 --check-connection 不能同时使用。'; fi
if [[ -z $remote ]]; then read -r -p '远程 IP（也可输入 用户名@IP）：' remote; fi
if ! $check_connection && [[ -z $destination ]]; then read -r -p '远程存放数据的绝对路径：' destination; fi
[[ $remote =~ ^([a-zA-Z0-9_][a-zA-Z0-9_.-]*@)?[a-zA-Z0-9][a-zA-Z0-9.-]*$ ]] || die 'IP/主机名格式不正确。'
if [[ $remote != *@* ]]; then
  remote_user=${REMOTE_USER:-$(id -un)}
  [[ $remote_user =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ ]] || die 'REMOTE_USER 格式不正确。'
  remote="$remote_user@$remote"
fi
if ! $check_connection; then
  [[ $destination == /* && $destination != *$'\n'* && $destination != *$'\r'* ]] || die '请使用远程绝对路径，例如 /data/my_datasets。'
  destination=${destination%/}
  [[ -n $destination ]] || die '请指定一个数据目录，不能直接使用 /。'
fi
ssh_port=${SSH_PORT:-22}
[[ $ssh_port =~ ^[0-9]{1,5}$ ]] || die 'SSH_PORT 必须为 1 到 65535 的整数。'
(( 10#$ssh_port >= 1 && 10#$ssh_port <= 65535 )) || die 'SSH_PORT 超出范围。'
if [[ -n $jump_host ]]; then
  [[ $jump_host =~ ^([a-zA-Z0-9_][a-zA-Z0-9_.-]*@)?[a-zA-Z0-9][a-zA-Z0-9.-]*(:[0-9]{1,5})?$ ]] || die 'SSH_JUMP/--jump 格式不正确，应为 [用户名@]主机[:端口]。'
  if [[ $jump_host == *:* ]]; then
    jump_port=${jump_host##*:}
    (( 10#$jump_port >= 1 && 10#$jump_port <= 65535 )) || die '中转 SSH 端口必须为 1 到 65535 的整数。'
  fi
fi
connect_timeout=${SSH_CONNECT_TIMEOUT:-20}
[[ $connect_timeout =~ ^[0-9]{1,5}$ ]] || die 'SSH_CONNECT_TIMEOUT 必须为 1 到 65535 的整数。'
(( 10#$connect_timeout >= 1 && 10#$connect_timeout <= 65535 )) || die 'SSH_CONNECT_TIMEOUT 超出范围。'
# Reuse the same connection options for checks, mkdir, and every rsync call.
ssh_args=(-p "$ssh_port" -o "ConnectTimeout=$connect_timeout")
if [[ -n $jump_host ]]; then ssh_args+=(-J "$jump_host"); fi
if $verbose; then ssh_args+=(-vvv); fi
# All values in ssh_args are validated above and contain no shell metacharacters.
rsync_ssh="ssh ${ssh_args[*]}"

check_remote() {
  local status
  printf '\n检查目标服务器：%s（端口 %s，中转 %s）\n' "$remote" "$ssh_port" "${jump_host:-无}"
  if ssh "${ssh_args[@]}" "$remote" 'if command -v rsync >/dev/null; then exit 0; else exit 127; fi'; then
    printf '目标 SSH 登录成功，rsync 可用。\n'
  else
    status=$?
    if (( status == 127 )); then die '目标 SSH 登录成功，但目标服务器未安装 rsync 或 rsync 不在 PATH 中。'; fi
    die "目标 SSH 检查失败（退出码 $status）。请用 --check-connection --verbose 定位；检查中转到目标的网络、SSH 端口及中转机 TCP 转发权限。"
  fi
}

if $check_connection; then
  command -v ssh >/dev/null || die '本机未安装 ssh。'
  if [[ -n $jump_host ]]; then
    jump_args=(-o "ConnectTimeout=$connect_timeout")
    jump_remote=$jump_host
    if [[ $jump_host == *:* ]]; then
      jump_remote=${jump_host%:*}
      jump_args+=(-p "$jump_port")
    fi
    if $verbose; then jump_args+=(-vvv); fi
    printf '检查本机到中转机：%s\n' "$jump_host"
    ssh "${jump_args[@]}" "$jump_remote" true || die '本机到中转机 SSH 检查失败，请检查中转机地址、端口、网络和认证。'
    printf '中转机 SSH 登录成功。\n'
  fi
  check_remote
  exit 0
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
soundspaces_source=${SOUNDSPACES_SOURCE:-/hdd-data3/yongzhang/finnal_exp/data}
data2_source=${DATA2_SOURCE:-/data2/getuanhui}
avnav_source=${AVNAV_SOURCE:-/data/Getuanhui/avnav-hybrid-mutiple}
attemp_source=${ATTEMP_SOURCE:-$script_dir/result/2026-10-28-avnav-codex_attemp}
attemp_rgb_source=${ATTEMP_RGB_SOURCE:-$script_dir/result/2026-10-28-avnav-codex_attemp_rgb}
sources=("$soundspaces_source" "$data2_source" "$avnav_source" "$attemp_source" "$attemp_rgb_source")
targets=("$destination/soundspaces/data" \
  "$destination/hybrid_offline_RL_datasets/getuanhui" \
  "$destination/hybrid_offline_RL_datasets/avnav-hybrid-mutiple" \
  "$destination/result/2026-10-28-avnav-codex_attemp" \
  "$destination/result/2026-10-28-avnav-codex_attemp_rgb")

for i in "${!sources[@]}"; do
  [[ -d ${sources[i]} && -r ${sources[i]} && -x ${sources[i]} ]] || die "源数据目录不存在或不可读：${sources[i]}"
  sources[i]=$(cd -- "${sources[i]}" && pwd -P)
done
printf '远程：%s（SSH 端口 %s）\n' "$remote" "$ssh_port"
if [[ -n $jump_host ]]; then printf '中转机：%s（SSH ProxyJump）\n' "$jump_host"; fi
for i in "${!sources[@]}"; do
  printf '\n来源：%s/\n目标：%s:%s/\n' "${sources[i]}" "$remote" "${targets[i]}"
done
printf '\n复制符号链接指向的实际数据，保留原始 split 目录和文件名。\n'
if $dry_run; then exit 0; fi

command -v ssh >/dev/null || die '本机未安装 ssh。'
command -v rsync >/dev/null || die '本机未安装 rsync，请先安装。'
check_remote
rsync_args=(-aL --no-owner --no-group --partial --partial-dir=.rsync-partial \
  --info=progress2 --protect-args -e "$rsync_ssh")

manifest=$(mktemp)
trap 'rm -f -- "$manifest"' EXIT
{
  printf 'Copied at UTC: %s\nRemote: %s\n' "$(date -u +%FT%TZ)" "$remote"
  printf 'SSH port: %s\nSSH jump: %s\n' "$ssh_port" "${jump_host:-direct}"
  for i in "${!sources[@]}"; do printf '%s -> %s\n' "${sources[i]}" "${targets[i]}"; done
  printf '\nTrain only: training-split data identified by the original manifests.\n'
  printf 'Validation data is for checkpoint selection; test data is for final evaluation only.\n'
  printf 'Preserve original split manifests and episode/scene IDs. Audit split overlap before training.\n'
  printf 'SoundSpaces test splits are for final evaluation only.\n'
} > "$manifest"

for i in "${!sources[@]}"; do
  printf '\n正在复制 %s\n' "${sources[i]}"
  ssh "${ssh_args[@]}" "$remote" "mkdir -p -- $(shell_quote "${targets[i]}")"
  # Copy contents (including hidden files) without adding a nested source directory.
  rsync "${rsync_args[@]}" -- "${sources[i]}/" "$remote:${targets[i]}/"
done
rsync "${rsync_args[@]}" -- "$manifest" "$remote:$destination/dataset_copy_sources.txt"
printf '\n数据复制完成：%s:%s\n' "$remote" "$destination"
printf '新机器的项目 data 路径应指向：%s/soundspaces/data\n' "$destination"
printf '训练和验证配置中的数据路径也需要改成上述远程目标路径。\n'
