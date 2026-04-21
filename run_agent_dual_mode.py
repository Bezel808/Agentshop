#!/usr/bin/env python3
"""
运行 Verbal + Visual 双感知模式 Agent 选品

顺序执行 verbal 和 visual 模式，带超时监控，卡住时自动中断并尝试 debug。
"""

import os
import sys
import signal
import subprocess
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# 超时配置（秒）
VERBAL_TIMEOUT = 120
VISUAL_TIMEOUT = 180
SERVER_WAIT_TIMEOUT = 30


def wait_for_server(url: str = "http://localhost:5000", timeout: int = SERVER_WAIT_TIMEOUT) -> bool:
    """等待 Web 服务器就绪"""
    for i in range(timeout):
        try:
            r = requests.get(f"{url}/api/search?q=smart_watch&page_size=1", timeout=5)
            if r.status_code == 200:
                print(f"[OK] 服务器已就绪 ({url})")
                return True
        except Exception:
            pass
        time.sleep(1)
        if (i + 1) % 5 == 0:
            print(f"  等待服务器... {(i+1)}s/{timeout}s")
    return False


def run_with_timeout(cmd: list, timeout: int, label: str) -> tuple[int, str]:
    """带超时运行命令，超时则 kill。返回 (exit_code, output)"""
    print(f"\n{'='*60}")
    print(f"🚀 [{label}] 启动 (timeout={timeout}s)")
    print(f"   命令: {' '.join(cmd)}")
    print("="*60)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines = []
        start = time.time()
        while proc.poll() is None:
            if time.time() - start > timeout:
                proc.kill()
                proc.wait()
                print(f"\n⚠️ [{label}] 超时 ({timeout}s)，已终止")
                return -1, "\n".join(output_lines)
            line = proc.stdout.readline()
            if line:
                line = line.rstrip()
                print(line)
                output_lines.append(line)
            else:
                time.sleep(0.1)
        # 读取剩余输出
        rest = proc.stdout.read()
        if rest:
            for l in rest.splitlines():
                print(l)
                output_lines.append(l)
        return proc.returncode, "\n".join(output_lines)
    except Exception as e:
        print(f"[ERROR] {e}")
        return -2, str(e)


def _load_query_preset(preset: str) -> str:
    """从 configs/queries.yaml 加载预设 query"""
    cfg_path = Path(__file__).parent / "configs" / "queries.yaml"
    if not cfg_path.exists():
        return ""
    try:
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data:
            return ""
        if preset == "default":
            return data.get("default", "smart watch under 100 dollars")
        if preset in data and isinstance(data[preset], list) and data[preset]:
            import random
            return random.choice(data[preset])
        if preset in data and isinstance(data[preset], str):
            return data[preset]
    except Exception:
        pass
    return ""


def _load_env():
    """加载 .env 到环境变量（与 run_browser_agent 一致）"""
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        for line in _env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip("'\"")
                if k and v and k not in os.environ:
                    os.environ[k] = v


def main():
    _load_env()
    import argparse
    parser = argparse.ArgumentParser(description="Verbal + Visual 双模式 Agent 选品")
    parser.add_argument("--query", default=None, help="用户购物需求（自然语言）")
    parser.add_argument("--query-preset", default="default",
        help="从 configs/queries.yaml 加载: default, multi_attribute_budget, scenario_based, multi_constraint, preference_exclude, verbal_vs_visual_divergent")
    parser.add_argument("--server", default="http://localhost:5000", help="Web 服务器 URL")
    parser.add_argument("--llm", choices=["openai", "qwen", "kimi"], default="qwen")
    parser.add_argument("--verbal-only", action="store_true", help="仅运行 verbal")
    parser.add_argument("--visual-only", action="store_true", help="仅运行 visual")
    parser.add_argument("--verbal-timeout", type=int, default=VERBAL_TIMEOUT)
    parser.add_argument("--visual-timeout", type=int, default=VISUAL_TIMEOUT)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if args.llm == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
    elif args.llm == "kimi":
        api_key = api_key or os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        print("错误: 需要 API Key。设置 QWEN_API_KEY 或 --api-key")
        sys.exit(1)

    query = args.query
    if query is None:
        query = _load_query_preset(args.query_preset) or "smart watch under 100 dollars"
    print(f"📋 Query: {query}")

    if not wait_for_server(args.server, SERVER_WAIT_TIMEOUT):
        print("\n错误: Web 服务器未就绪。请先启动:")
        print("  python start_web_server.py -d datasets_unified --simple-search -p 5000")
        sys.exit(1)

    base_cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_browser_agent.py"),
        "--api-key", api_key,
        "--llm", args.llm,
        "--query", query,
        "--server", args.server,
        "--once",
    ]

    results = {}
    run_verbal = not args.visual_only
    run_visual = not args.verbal_only

    if run_verbal:
        cmd = base_cmd + ["--perception", "verbal"]
        code, out = run_with_timeout(cmd, args.verbal_timeout, "Verbal")
        results["verbal"] = (code, out)
        if code != 0 and "推荐商品" not in out:
            print("\n[DEBUG] Verbal 可能卡住，检查: 1) API 是否正常 2) prompt 是否导致 LLM 输出格式异常")

    if run_visual:
        cmd = base_cmd + ["--perception", "visual"]
        code, out = run_with_timeout(cmd, args.visual_timeout, "Visual")
        results["visual"] = (code, out)
        if code != 0 and "推荐完成" not in out:
            print("\n[DEBUG] Visual 可能卡住，检查: 1) Playwright 是否安装 2) VLM 是否支持图像 3) 截图是否过大")

    print("\n" + "="*60)
    print("📊 运行结果汇总")
    print("="*60)
    for mode, (code, _) in results.items():
        status = "✓ 完成" if code == 0 else f"✗ 退出码 {code}"
        print(f"  {mode}: {status}")
    print()


if __name__ == "__main__":
    main()
