"""
Browser Interaction Tools using MCP

封装 MCP Browser 工具，让 Agent 可以操作网页。
"""

from typing import Any, Dict, Optional, List
import logging
import re
from urllib.parse import quote
import requests

from aces.tools.base_tool import BaseTool
from aces.core.protocols import ToolResult


logger = logging.getLogger(__name__)


class HTTPMCPCaller:
    """
    Minimal MCP caller over HTTP JSON.

    Expected endpoint contract:
    POST {endpoint} with JSON:
    {
      "server": "<server-name>",
      "tool": "<tool-name>",
      "arguments": {...}
    }
    """

    def __init__(self, endpoint: str, timeout: int = 30):
        self.endpoint = endpoint
        self.timeout = timeout

    def call_tool(self, server: str, tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(
            self.endpoint,
            json={"server": server, "tool": tool, "arguments": arguments},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data


class BrowserNavigateTool(BaseTool):
    """导航到指定 URL"""
    
    def __init__(self, mcp_caller):
        super().__init__(
            name="browser_navigate",
            description="Navigate to a URL in the browser",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to"
                    }
                },
                "required": ["url"]
            }
        )
        self.mcp = mcp_caller
    
    def _execute_impl(self, parameters: Dict[str, Any]) -> Any:
        result = self.mcp.call_tool("cursor-ide-browser", "browser_navigate", {
            "url": parameters["url"]
        })
        return result


class BrowserSnapshotTool(BaseTool):
    """获取当前页面的快照（DOM结构和截图）"""
    
    def __init__(self, mcp_caller):
        super().__init__(
            name="browser_snapshot",
            description="Get a snapshot of the current browser page (DOM structure and screenshot)",
            input_schema={
                "type": "object",
                "properties": {},
            }
        )
        self.mcp = mcp_caller
    
    def _execute_impl(self, parameters: Dict[str, Any]) -> Any:
        result = self.mcp.call_tool("cursor-ide-browser", "browser_snapshot", {})
        return result


class BrowserClickTool(BaseTool):
    """点击页面元素"""
    
    def __init__(self, mcp_caller):
        super().__init__(
            name="browser_click",
            description="Click an element on the page",
            input_schema={
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Element reference from snapshot"
                    }
                },
                "required": ["ref"]
            }
        )
        self.mcp = mcp_caller
    
    def _execute_impl(self, parameters: Dict[str, Any]) -> Any:
        result = self.mcp.call_tool("cursor-ide-browser", "browser_click", {
            "ref": parameters["ref"]
        })
        return result


class BrowserTypeTool(BaseTool):
    """在输入框中输入文字"""
    
    def __init__(self, mcp_caller):
        super().__init__(
            name="browser_type",
            description="Type text into an input field",
            input_schema={
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Element reference for input field"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type"
                    }
                },
                "required": ["ref", "text"]
            }
        )
        self.mcp = mcp_caller
    
    def _execute_impl(self, parameters: Dict[str, Any]) -> Any:
        result = self.mcp.call_tool("cursor-ide-browser", "browser_type", {
            "ref": parameters["ref"],
            "text": parameters["text"]
        })
        return result


class SimpleBrowserController:
    """
    简化的浏览器控制器，用于 ACES-v2 场景。
    
    封装常用的浏览器操作流程。
    """
    
    def __init__(self, mcp_caller, base_url: str = "http://192.168.8.240:5000"):
        self.mcp = mcp_caller
        self.base_url = base_url
        
        # 创建工具
        self.navigate = BrowserNavigateTool(mcp_caller)
        self.snapshot = BrowserSnapshotTool(mcp_caller)
        self.click = BrowserClickTool(mcp_caller)
        self.type = BrowserTypeTool(mcp_caller)
    
    def goto_search_page(self, query: str = ""):
        """导航到搜索页面"""
        url = f"{self.base_url}/search?q={query}" if query else f"{self.base_url}/"
        return self.navigate.execute({"url": url})
    
    def get_current_state(self):
        """获取当前页面状态（包含截图）"""
        return self.snapshot.execute({})

    def _flatten_snapshot(self, snapshot_data: Any) -> List[Dict[str, Any]]:
        """
        Flatten unknown snapshot structures into a list of candidate nodes.
        Each node should include text-ish fields and a clickable ref-ish field.
        """
        nodes: List[Dict[str, Any]] = []

        def _walk(obj: Any):
            if isinstance(obj, dict):
                nodes.append(obj)
                for v in obj.values():
                    _walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    _walk(item)

        _walk(snapshot_data)
        return nodes

    @staticmethod
    def _pick_ref(node: Dict[str, Any]) -> Optional[str]:
        for key in ("ref", "id", "elementId", "nodeId"):
            val = node.get(key)
            if isinstance(val, (str, int)):
                return str(val)
        return None

    @staticmethod
    def _node_text(node: Dict[str, Any]) -> str:
        text_parts = []
        for key in ("text", "name", "label", "value", "role", "ariaLabel"):
            val = node.get(key)
            if isinstance(val, str):
                text_parts.append(val)
        return " ".join(text_parts).strip()

    def _find_search_input_ref(self, snapshot_data: Any) -> Optional[str]:
        candidates = self._flatten_snapshot(snapshot_data)
        # Prefer actual input/search nodes.
        for node in candidates:
            text = self._node_text(node).lower()
            role = str(node.get("role", "")).lower()
            if "search" in text or "search" in role:
                ref = self._pick_ref(node)
                if ref:
                    return ref
        # Fallback: first input-like node.
        for node in candidates:
            text = self._node_text(node).lower()
            if "input" in text or "textbox" in text:
                ref = self._pick_ref(node)
                if ref:
                    return ref
        return None

    def _find_product_refs(self, snapshot_data: Any) -> List[str]:
        candidates = self._flatten_snapshot(snapshot_data)
        refs: List[str] = []
        seen = set()
        product_like = re.compile(r"(product|item|detail|view)", re.IGNORECASE)
        for node in candidates:
            text = self._node_text(node)
            href = str(node.get("href", "") or "")
            if not (product_like.search(text) or "/product/" in href):
                continue
            ref = self._pick_ref(node)
            if ref and ref not in seen:
                seen.add(ref)
                refs.append(ref)
        return refs
    
    def search_products(self, query: str):
        """在搜索框输入并搜索"""
        snapshot_result = self.get_current_state()
        if not snapshot_result.success:
            return snapshot_result

        search_ref = self._find_search_input_ref(snapshot_result.data)
        if search_ref:
            type_result = self.type.execute({"ref": search_ref, "text": query})
            if type_result.success:
                return type_result

        # Fallback: direct URL navigation if input ref cannot be inferred.
        return self.navigate.execute({"url": f"{self.base_url}/search?q={quote(query)}"})
    
    def click_product(self, product_index: int):
        """点击第 N 个商品"""
        snapshot_result = self.get_current_state()
        if not snapshot_result.success:
            return snapshot_result

        refs = self._find_product_refs(snapshot_result.data)
        if product_index < 1 or product_index > len(refs):
            return ToolResult(
                success=False,
                data=None,
                error=f"Product index out of range: {product_index}, max={len(refs)}",
            )

        return self.click.execute({"ref": refs[product_index - 1]})
