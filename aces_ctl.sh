#!/usr/bin/env bash
# =========================================================================
#  ACES-v2 实验控制中心  (aces_ctl.sh)
#
#  一站式脚本：启动/停止/监视 Web 服务、SSH 端口转发、运行实验
#
#  用法:  ./aces_ctl.sh <命令> [选项]
#
#  命令:
#    start     启动 Web 服务（后台）
#    stop      停止 Web 服务
#    restart   重启 Web 服务
#    status    查看服务 + 进程状态
#    log       实时查看服务日志
#    forward   打印 SSH 端口转发命令（方便复制）
#    share-start  启动临时+稳定公网隧道
#    share-status 查看公网隧道状态和访问链接
#    share-stop   停止公网隧道
#    run       运行 Agent (透传参数给 run_browser_agent.py)
#    conditions 列出可用的实验条件文件
#    datasets  列出可用的数据集 / 商品类目
#    health    健康检查（返回JSON）
# =========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 默认配置 ----
PORT="${ACES_PORT:-5000}"
HOST="${ACES_HOST:-0.0.0.0}"
DATASETS_DIR="${ACES_DATASETS:-datasets_unified}"
MAX_PAGES="${ACES_MAX_PAGES:-5}"    # 最多 N 页，每页 8 个 = 最多 40 个商品
LOG_FILE="${ACES_LOG:-/tmp/aces_web_server.log}"
PROCESS_MARK="start_web_server.py"
SIMPLE_SEARCH="${ACES_SIMPLE_SEARCH:-}"
SHARE_DIR="${ACES_SHARE_DIR:-/tmp/aces_share_${PORT}}"
LT_LOG="${SHARE_DIR}/localtunnel.log"
CF_LOG="${SHARE_DIR}/cloudflared.log"
LT_SESSION="${ACES_LT_SESSION:-aces_lt_${PORT}}"
CF_SESSION="${ACES_CF_SESSION:-aces_cf_${PORT}}"

# ---- 颜色 ----
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
_err()   { echo -e "${RED}[ERR]${NC}   $*" >&2; }
_head()  { echo -e "\n${BOLD}${CYAN}$*${NC}"; }

# ---- 辅助函数 ----
_is_running() { pgrep -f "$PROCESS_MARK" > /dev/null 2>&1; }

_wait_healthy() {
    local timeout=${1:-${ACES_START_TIMEOUT:-90}}
    for (( i=1; i<=timeout; i++ )); do
        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            return 0
        fi
        if [[ $((i % 10)) -eq 0 ]]; then
            _info "  等待中... ${i}s/${timeout}s (构建索引/加载模型可能需要 1-2 分钟)"
        fi
        sleep 1
    done
    return 1
}

_get_local_ip() {
    hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1"
}

_urlencode() {
    python3 - "$1" <<'PY'
import sys
from urllib.parse import quote
print(quote(sys.argv[1], safe=""))
PY
}

_tmux_session_exists() {
    local name="$1"
    tmux has-session -t "$name" 2>/dev/null
}

_resolve_cmd() {
    local name="$1"
    command -v "$name" 2>/dev/null || return 1
}

_localtunnel_url() {
    [[ -f "$LT_LOG" ]] || return 1
    grep -Eo 'https://[a-zA-Z0-9.-]+\.loca\.lt' "$LT_LOG" | tail -1
}

_cloudflared_url() {
    [[ -f "$CF_LOG" ]] || return 1
    grep -Eo 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "$CF_LOG" | tail -1
}

_print_forward_commands() {
    local ip
    ip=$(_get_local_ip)
    echo "  # 基本转发 (搜索页 + 查看器)"
    echo "  ssh -L ${PORT}:localhost:${PORT} $(whoami)@${ip}"
    echo ""
    echo "  # 后台转发 (不占用终端)"
    echo "  ssh -fNL ${PORT}:localhost:${PORT} $(whoami)@${ip}"
}

# =========================================================================
# 命令实现
# =========================================================================

cmd_start() {
    _head "启动 ACES-v2 Web 服务"

    if _is_running; then
        _warn "Web 服务已在运行中 (PID: $(pgrep -f "$PROCESS_MARK" | head -1))"
        _info "如需重启: ./aces_ctl.sh restart"
        cmd_health
        return 0
    fi

    local extra_args=""
    [[ -n "$SIMPLE_SEARCH" ]] && extra_args="--simple-search"

    # 支持从命令行传入 --simple-search
    for arg in "$@"; do
        [[ "$arg" == "--simple-search" ]] && extra_args="--simple-search"
        [[ "$arg" == "-p" || "$arg" == "--port" ]] && { shift; PORT="$1"; }
    done

    _info "端口: $PORT | 数据集: $DATASETS_DIR | 最多页: $MAX_PAGES | 日志: $LOG_FILE"

    nohup python3 start_web_server.py \
        --host "$HOST" \
        --port "$PORT" \
        --datasets-dir "$DATASETS_DIR" \
        --max-pages "$MAX_PAGES" \
        $extra_args \
        >> "$LOG_FILE" 2>&1 &

    _info "正在启动 (LlamaIndex 建索引+加载模型约 30-90 秒，首次可能更久)..."

    if _wait_healthy 90; then
        _info "服务已就绪!"
        echo ""
        _print_access_info
    else
        _err "启动超时，查看日志: tail -50 $LOG_FILE"
        _info "若首次启动慢，可尝试: ACES_SIMPLE_SEARCH=1 ./aces_ctl.sh start (跳过 RAG，秒启)"
        return 1
    fi
}

cmd_stop() {
    _head "停止 ACES-v2 Web 服务"

    if _is_running; then
        pkill -f "$PROCESS_MARK" || true
        sleep 1
        if _is_running; then
            _warn "进程仍在运行，尝试强制终止..."
            pkill -9 -f "$PROCESS_MARK" || true
        fi
        _info "已停止。"
    else
        _info "未检测到运行中的 Web 服务。"
    fi
}

cmd_restart() {
    cmd_stop
    sleep 1
    cmd_start "$@"
}

cmd_status() {
    _head "ACES-v2 状态"

    if _is_running; then
        local pid
        pid=$(pgrep -f "$PROCESS_MARK" | head -1)
        _info "Web 服务: ${GREEN}运行中${NC} (PID: $pid)"

        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            _info "健康检查: ${GREEN}OK${NC}"
            curl -s "http://localhost:${PORT}/health" | python3 -m json.tool 2>/dev/null || true
        else
            _warn "健康检查: 未响应 (可能仍在启动)"
        fi

        # 内存占用
        local mem
        mem=$(ps -o rss= -p "$pid" 2>/dev/null || echo "0")
        _info "内存占用: $((mem / 1024)) MB"
    else
        _warn "Web 服务: ${RED}未运行${NC}"
    fi

    # 数据集统计
    echo ""
    _info "数据集目录: $DATASETS_DIR"
    local count
    count=$(python3 - "$DATASETS_DIR" <<'PY'
import json
import sys
from pathlib import Path
base = Path(sys.argv[1])
n = 0
for fp in base.glob("*.json"):
    try:
        data = json.load(open(fp, "r", encoding="utf-8"))
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            n += 1
    except Exception:
        pass
print(n)
PY
)
    _info "商品类目数: $count"
}

cmd_log() {
    _head "实时日志 (Ctrl+C 退出)"
    tail -f "$LOG_FILE"
}

cmd_forward() {
    _head "SSH 端口转发命令 (在你的本地电脑执行)"
    echo ""
    _print_forward_commands
    echo ""
    echo "  然后在浏览器打开:"
    echo "    http://localhost:${PORT}/              搜索页"
    echo "    http://localhost:${PORT}/viewer         实时查看器"
    echo "    http://localhost:${PORT}/health         健康检查"
    echo ""
}

cmd_share_start() {
    _head "启动公网分享隧道"

    local tmux_bin=""
    local npx_bin=""
    local cloudflared_bin=""
    tmux_bin=$(_resolve_cmd tmux || true)
    npx_bin=$(_resolve_cmd npx || true)
    cloudflared_bin=$(_resolve_cmd cloudflared || true)

    if [[ -z "$tmux_bin" ]]; then
        _err "未检测到 tmux，请先安装 tmux"
        return 1
    fi
    if [[ -z "$npx_bin" ]]; then
        _err "未检测到 npx，无法启动 localtunnel"
        return 1
    fi

    local token="${ACES_VIEWER_TOKEN:-}"
    if [[ -z "$token" ]]; then
        _err "ACES_VIEWER_TOKEN 未设置，拒绝启动分享。示例: export ACES_VIEWER_TOKEN='your-secret-token'"
        return 1
    fi

    mkdir -p "$SHARE_DIR"
    : > "$LT_LOG"
    : > "$CF_LOG"

    if _tmux_session_exists "$LT_SESSION"; then
        _warn "检测到旧 localtunnel 会话，正在重启: $LT_SESSION"
        tmux kill-session -t "$LT_SESSION" || true
    fi
    if _tmux_session_exists "$CF_SESSION"; then
        _warn "检测到旧 cloudflared 会话，正在重启: $CF_SESSION"
        tmux kill-session -t "$CF_SESSION" || true
    fi

    "$tmux_bin" new-session -d -s "$LT_SESSION" \
        "cd '$SCRIPT_DIR' && '$npx_bin' --yes localtunnel --port '$PORT' > '$LT_LOG' 2>&1"
    _info "localtunnel 已启动 (tmux: $LT_SESSION)"

    if [[ -n "$cloudflared_bin" ]]; then
        "$tmux_bin" new-session -d -s "$CF_SESSION" \
            "cd '$SCRIPT_DIR' && '$cloudflared_bin' tunnel --url 'http://localhost:${PORT}' > '$CF_LOG' 2>&1"
        _info "cloudflared 已启动 (tmux: $CF_SESSION)"
    else
        _warn "未检测到 cloudflared，仅保留临时外链。安装后可获得稳定外链:"
        echo "  brew install cloudflared   或   apt-get install cloudflared"
    fi

    sleep 4
    cmd_share_status
}

cmd_share_status() {
    _head "公网分享状态"

    local token="${ACES_VIEWER_TOKEN:-}"
    local token_enc=""
    if [[ -n "$token" ]]; then
        token_enc=$(_urlencode "$token")
    fi

    local lt_running="no"
    local cf_running="no"
    _tmux_session_exists "$LT_SESSION" && lt_running="yes"
    _tmux_session_exists "$CF_SESSION" && cf_running="yes"

    local lt_url=""
    local cf_url=""
    [[ "$lt_running" == "yes" ]] && lt_url=$(_localtunnel_url 2>/dev/null || true)
    [[ "$cf_running" == "yes" ]] && cf_url=$(_cloudflared_url 2>/dev/null || true)

    echo "  端口: $PORT"
    echo "  token: $([[ -n "$token" ]] && echo "已设置" || echo "未设置")"
    echo "  tmux localtunnel: $lt_running ($LT_SESSION)"
    echo "  tmux cloudflared: $cf_running ($CF_SESSION)"
    echo ""

    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        _info "本地服务健康检查: OK"
    else
        _warn "本地服务健康检查失败: http://localhost:${PORT}/health"
    fi
    echo ""

    _head "本地转发命令"
    _print_forward_commands
    echo ""

    _head "公网链接"
    if [[ -n "$lt_url" ]]; then
        if [[ -n "$token_enc" ]]; then
            echo "  临时(localtunnel) viewer: ${lt_url}/viewer?token=${token_enc}"
        else
            echo "  临时(localtunnel) viewer: ${lt_url}/viewer  (未附带 token)"
        fi
        echo "  临时(localtunnel) 首页:   ${lt_url}/"
    elif [[ "$lt_running" == "yes" ]]; then
        echo "  临时(localtunnel): 正在启动，尚未获取 URL，查看日志: tail -n 50 $LT_LOG"
    else
        echo "  临时(localtunnel): 未运行"
    fi

    if [[ -n "$cf_url" ]]; then
        if [[ -n "$token_enc" ]]; then
            echo "  稳定(cloudflared) viewer: ${cf_url}/viewer?token=${token_enc}"
        else
            echo "  稳定(cloudflared) viewer: ${cf_url}/viewer  (未附带 token)"
        fi
        echo "  稳定(cloudflared) 首页:   ${cf_url}/"
    elif [[ "$cf_running" == "yes" ]]; then
        echo "  稳定(cloudflared): 正在启动，尚未获取 URL，查看日志: tail -n 80 $CF_LOG"
    elif command -v cloudflared >/dev/null 2>&1; then
        echo "  稳定(cloudflared): 已安装但未运行（请执行 ./aces_ctl.sh share-start）"
    else
        echo "  稳定(cloudflared): 未安装 cloudflared"
    fi
    echo ""
}

cmd_share_stop() {
    _head "停止公网分享隧道"
    local stopped=0
    if _tmux_session_exists "$LT_SESSION"; then
        tmux kill-session -t "$LT_SESSION" || true
        _info "已停止 localtunnel 会话: $LT_SESSION"
        stopped=1
    fi
    if _tmux_session_exists "$CF_SESSION"; then
        tmux kill-session -t "$CF_SESSION" || true
        _info "已停止 cloudflared 会话: $CF_SESSION"
        stopped=1
    fi
    if [[ "$stopped" -eq 0 ]]; then
        _info "未检测到运行中的分享隧道会话"
    fi
}

cmd_health() {
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        curl -s "http://localhost:${PORT}/health" | python3 -m json.tool 2>/dev/null
    else
        _err "服务未响应 (端口: $PORT)"
        return 1
    fi
}

cmd_run() {
    _head "运行实验"

    # 确保服务已启动（如果实验用的是 web_renderer 或 llamaindex 在线模式）
    # 对于 offline 模式无需 web 服务

    # 透传所有参数给 run_browser_agent.py
    _info "执行: python3 run_browser_agent.py $*"
    echo ""
    python3 run_browser_agent.py "$@"
}

cmd_conditions() {
    _head "可用的实验条件文件"
    echo ""
    for f in configs/experiments/*.yaml configs/experiments/*.yml configs/experiments/*.json; do
        [[ -f "$f" ]] || continue
        local name
        name=$(grep -m1 "^name:" "$f" 2>/dev/null | sed 's/name: *//' || basename "$f")
        echo "  $f"
        echo "    名称: $name"
        # 列出条件名
        local conds
        conds=$(grep "^  - name:" "$f" 2>/dev/null | sed 's/.*name: */  /' || echo "    (无法解析)")
        echo "    条件:$conds"
        echo ""
    done
}

cmd_datasets() {
    _head "可用的商品数据集 ($DATASETS_DIR)"
    echo ""
    printf "  %-25s %s\n" "类目 (查询词)" "商品数"
    printf "  %-25s %s\n" "────────────────────" "──────"
    for f in "$DATASETS_DIR"/*.json; do
        [[ -f "$f" ]] || continue
        local name count is_product
        name=$(basename "$f" .json)
        is_product=$(python3 - "$f" <<'PY'
import json
import sys
try:
    data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
except Exception:
    print("0")
    raise SystemExit
ok = isinstance(data, list) and all(isinstance(x, dict) for x in data)
print("1" if ok else "0")
PY
)
        [[ "$is_product" == "1" ]] || continue
        count=$(python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null || echo "?")
        printf "  %-25s %s\n" "$name" "$count"
    done
    echo ""
}

_print_access_info() {
    local ip
    ip=$(_get_local_ip)
    echo "  ┌───────────────────────────────────────────────────┐"
    echo "  │  本机访问:                                         │"
    echo "  │    http://localhost:${PORT}/               搜索页   │"
    echo "  │    http://localhost:${PORT}/viewer          查看器   │"
    echo "  │    http://localhost:${PORT}/health          健康检查 │"
    echo "  ├───────────────────────────────────────────────────┤"
    echo "  │  远程访问 (先在本地执行 SSH 转发):                  │"
    echo "  │    ssh -L ${PORT}:localhost:${PORT} $(whoami)@${ip}  │"
    echo "  │    然后浏览器打开 http://localhost:${PORT}/         │"
    echo "  ├───────────────────────────────────────────────────┤"
    echo "  │  日志: tail -f $LOG_FILE              │"
    echo "  │  停止: ./aces_ctl.sh stop                          │"
    echo "  └───────────────────────────────────────────────────┘"
}

# =========================================================================
# 主入口
# =========================================================================

cmd_help() {
    echo ""
    echo "ACES-v2 实验控制中心"
    echo ""
    echo "用法: ./aces_ctl.sh <命令> [选项]"
    echo ""
    echo "服务管理:"
    echo "  start [--simple-search]   启动 Web 服务（后台运行）"
    echo "  stop                      停止 Web 服务"
    echo "  restart                   重启 Web 服务"
    echo "  status                    查看服务状态 + 系统信息"
    echo "  log                       实时查看服务日志"
    echo "  health                    健康检查（JSON 输出）"
    echo ""
    echo "网络:"
    echo "  forward                   打印 SSH 端口转发命令"
    echo "  share-start               启动 localtunnel + cloudflared 外链"
    echo "  share-status              查看外链状态/URL/健康检查"
    echo "  share-stop                关闭所有外链 tmux 会话"
    echo ""
    echo "实验:"
    echo "  run [...]                 运行 Agent（参数透传给 run_browser_agent.py）"
    echo "  conditions                列出可用的实验条件文件"
    echo "  datasets                  列出可用的商品数据集"
    echo ""
    echo "环境变量:"
    echo "  ACES_PORT          端口 (默认: 5000)"
    echo "  ACES_HOST          监听地址 (默认: 0.0.0.0)"
    echo "  ACES_DATASETS      数据集目录 (默认: datasets_unified)"
    echo "  ACES_MAX_PAGES     最多页数 (默认: 5，每页 8 个 = 最多 40 商品)"
    echo "  ACES_LOG           日志文件 (默认: /tmp/aces_web_server.log)"
    echo "  ACES_SIMPLE_SEARCH 设为1则禁用 LlamaIndex RAG"
    echo "  ACES_START_TIMEOUT 健康检查等待秒数 (默认: 90)"
    echo "  ACES_VIEWER_TOKEN  viewer 访问令牌（share-start 必填）"
    echo "  ACES_SHARE_DIR     外链日志目录 (默认: /tmp/aces_share_<port>)"
    echo ""
    echo "示例:"
    echo "  ./aces_ctl.sh start                          # 启动服务"
    echo "  ./aces_ctl.sh status                         # 查看状态"
    echo "  ./aces_ctl.sh share-start                    # 启动双外链"
    echo "  ./aces_ctl.sh share-status                   # 查看双外链 URL"
    echo "  ./aces_ctl.sh run --query mousepad            # 运行 Agent"
    echo "  ./aces_ctl.sh run --llm qwen --perception visual --query ski"
    echo ""
}

CMD="${1:-help}"
shift 2>/dev/null || true

case "$CMD" in
    start)      cmd_start "$@" ;;
    stop)       cmd_stop ;;
    restart)    cmd_restart "$@" ;;
    status)     cmd_status ;;
    log)        cmd_log ;;
    forward)    cmd_forward ;;
    share-start) cmd_share_start ;;
    share-status) cmd_share_status ;;
    share-stop) cmd_share_stop ;;
    health)     cmd_health ;;
    run)        cmd_run "$@" ;;
    conditions) cmd_conditions ;;
    datasets)   cmd_datasets ;;
    help|-h|--help) cmd_help ;;
    *)
        _err "未知命令: $CMD"
        cmd_help
        exit 1
        ;;
esac
