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
#    run       运行实验 (透传参数给 run_experiment.py)
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
LOG_FILE="${ACES_LOG:-/tmp/aces_web_server.log}"
PROCESS_MARK="start_web_server.py"
SIMPLE_SEARCH="${ACES_SIMPLE_SEARCH:-}"

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
    local timeout=${1:-30}
    for (( i=1; i<=timeout; i++ )); do
        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

_get_local_ip() {
    hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1"
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

    _info "端口: $PORT | 数据集: $DATASETS_DIR | 日志: $LOG_FILE"

    nohup python3 start_web_server.py \
        --host "$HOST" \
        --port "$PORT" \
        --datasets-dir "$DATASETS_DIR" \
        $extra_args \
        >> "$LOG_FILE" 2>&1 &

    _info "正在启动 (构建索引可能需要 10-20 秒)..."

    if _wait_healthy 30; then
        _info "服务已就绪!"
        echo ""
        _print_access_info
    else
        _err "启动超时，查看日志: tail -50 $LOG_FILE"
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
    count=$(ls -1 "$DATASETS_DIR"/*.json 2>/dev/null | wc -l)
    _info "商品类目数: $count"
}

cmd_log() {
    _head "实时日志 (Ctrl+C 退出)"
    tail -f "$LOG_FILE"
}

cmd_forward() {
    _head "SSH 端口转发命令 (在你的本地电脑执行)"
    local ip
    ip=$(_get_local_ip)
    echo ""
    echo "  # 基本转发 (搜索页 + 查看器)"
    echo "  ssh -L ${PORT}:localhost:${PORT} $(whoami)@${ip}"
    echo ""
    echo "  # 后台转发 (不占用终端)"
    echo "  ssh -fNL ${PORT}:localhost:${PORT} $(whoami)@${ip}"
    echo ""
    echo "  然后在浏览器打开:"
    echo "    http://localhost:${PORT}/              搜索页"
    echo "    http://localhost:${PORT}/viewer         实时查看器"
    echo "    http://localhost:${PORT}/health         健康检查"
    echo ""
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

    # 透传所有参数给 run_experiment.py
    _info "执行: python3 run_experiment.py $*"
    echo ""
    python3 run_experiment.py "$@"
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
        local name count
        name=$(basename "$f" .json)
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
    echo ""
    echo "实验:"
    echo "  run [...]                 运行实验（参数透传给 run_experiment.py）"
    echo "  conditions                列出可用的实验条件文件"
    echo "  datasets                  列出可用的商品数据集"
    echo ""
    echo "环境变量:"
    echo "  ACES_PORT          端口 (默认: 5000)"
    echo "  ACES_HOST          监听地址 (默认: 0.0.0.0)"
    echo "  ACES_DATASETS      数据集目录 (默认: datasets_unified)"
    echo "  ACES_LOG           日志文件 (默认: /tmp/aces_web_server.log)"
    echo "  ACES_SIMPLE_SEARCH 设为1则禁用 LlamaIndex RAG"
    echo ""
    echo "示例:"
    echo "  ./aces_ctl.sh start                          # 启动服务"
    echo "  ./aces_ctl.sh status                         # 查看状态"
    echo "  ./aces_ctl.sh run --mode simple --trials 10  # 运行简单实验"
    echo "  ./aces_ctl.sh run --llm qwen --query mousepad \\"
    echo "    -C configs/experiments/example_price_anchoring.yaml"
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
