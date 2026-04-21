"""
Base Agent Implementation

This module provides a concrete implementation of the Agent interface
that composes the three key slots: LLM, Perception, and Tools.
"""

from typing import List, Optional, Dict, Any
import logging
import json
import re
import ast

from aces.core.protocols import (
    Agent,
    LLMBackend,
    PerceptionMode,
    Tool,
    Observation,
    Action,
    Message,
    AgentState,
    ToolResult,
)


logger = logging.getLogger(__name__)


class ComposableAgent(Agent):
    """
    A modular agent composed of pluggable components.
    
    This is the "Dependency Injection Container" that wires together:
    - An LLM backend (the brain)
    - A perception mode (the sensor)
    - A toolset (the capabilities)
    
    Usage:
        agent = ComposableAgent(
            llm=OpenAIBackend(model="gpt-4o"),
            perception=VisualPerception(),
            tools=[SearchTool(), CartTool()],
            system_prompt="You are a helpful shopping assistant."
        )
    """
    
    def __init__(
        self,
        llm: LLMBackend,
        perception: PerceptionMode,
        tools: List[Tool],
        system_prompt: Optional[str] = None,
        max_history: int = 50,
        legacy_fallback: bool = False,
    ):
        """
        Initialize the agent.
        
        Args:
            llm: The LLM backend to use for decision making
            perception: The perception mode for processing observations
            tools: List of tools available to the agent
            system_prompt: Optional system prompt to guide behavior
            max_history: Maximum number of messages to keep in history
        """
        self.llm = llm
        self.perception = perception
        self.tools = {tool.get_schema().name: tool for tool in tools}
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.legacy_fallback = legacy_fallback
        
        # Agent state
        self.message_history: List[Message] = []
        self.observations: List[Observation] = []
        self.step_count = 0
        
        # Initialize with system prompt
        if system_prompt:
            self.message_history.append(
                Message(role="system", content=system_prompt)
            )
        
        logger.info(
            f"Initialized agent with LLM={llm.model_name}, "
            f"Perception={perception.get_modality()}, "
            f"Tools={list(self.tools.keys())}"
        )
    
    def act(self, observation: Observation) -> Action:
        """
        Decide on an action given an observation.
        
        This is the core decision-making loop:
        1. Validate observation matches perception mode
        2. Convert observation to message
        3. Query LLM with tools
        4. Parse LLM response into action
        """
        # Validate observation
        if not self.perception.validate_observation(observation):
            raise ValueError(
                f"Observation modality mismatch: expected {self.perception.get_modality()}, "
                f"got {observation.modality}"
            )
        
        # Store observation
        self.observations.append(observation)
        
        # Convert observation to message
        obs_message = self._observation_to_message(observation)
        self.message_history.append(obs_message)
        # Append context prompt if provided and different from data.
        # For visual mode, prompt is embedded in the same multimodal message.
        prompt = observation.metadata.get("prompt") if observation.metadata else None
        if prompt and prompt != observation.data and observation.modality != "visual":
            self.message_history.append(Message(role="user", content=prompt))
        
        # Trim history if needed
        self._trim_history()
        
        # Get tool schemas for LLM
        tool_schemas = [tool.get_schema() for tool in self.tools.values()]
        
        # Query LLM
        response = self.llm.generate(
            messages=self.message_history,
            tools=tool_schemas,
        )
        
        # Store LLM response
        self.message_history.append(response)
        
        # Parse response into action
        actions = self._parse_actions(response)
        action = actions[0] if actions else Action(tool_name="noop", parameters={}, reasoning="No action")
        
        self.step_count += 1
        
        return action

    def act_batch(self, observation: Observation) -> List[Action]:
        """
        Decide one or multiple actions for a single observation.

        Compatible with tool_calls[] style responses and falls back to single action.
        """
        if not self.perception.validate_observation(observation):
            raise ValueError(
                f"Observation modality mismatch: expected {self.perception.get_modality()}, "
                f"got {observation.modality}"
            )

        self.observations.append(observation)
        obs_message = self._observation_to_message(observation)
        self.message_history.append(obs_message)
        prompt = observation.metadata.get("prompt") if observation.metadata else None
        if prompt and prompt != observation.data and observation.modality != "visual":
            self.message_history.append(Message(role="user", content=prompt))

        self._trim_history()
        tool_schemas = [tool.get_schema() for tool in self.tools.values()]
        response = self.llm.generate(messages=self.message_history, tools=tool_schemas)
        self.message_history.append(response)

        actions = self._parse_actions(response)
        self.step_count += 1
        return actions
    
    async def aact(self, observation: Observation) -> Action:
        """Async version of act (implementation similar to act)."""
        # For brevity, implementation omitted but would mirror act()
        # with async LLM calls
        raise NotImplementedError("Async act not yet implemented")
    
    def reset(self) -> None:
        """Reset agent to initial state."""
        self.message_history = []
        self.observations = []
        self.step_count = 0
        
        # Re-add system prompt
        if self.system_prompt:
            self.message_history.append(
                Message(role="system", content=self.system_prompt)
            )
        
        logger.info("Agent reset")
    
    def get_state(self) -> AgentState:
        """Get current agent state for checkpointing."""
        return AgentState(
            message_history=self.message_history.copy(),
            observations=self.observations.copy(),
            step_count=self.step_count,
            metadata={
                "llm": self.llm.model_name,
                "perception": self.perception.get_modality(),
                "tools": list(self.tools.keys()),
            }
        )
    
    def load_state(self, state: AgentState) -> None:
        """Restore agent from checkpoint."""
        self.message_history = state.message_history
        self.observations = state.observations
        self.step_count = state.step_count
        
        logger.info(f"Loaded state at step {self.step_count}")
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _observation_to_message(self, observation: Observation) -> Message:
        """
        Convert an observation to a message format the LLM understands.
        
        This is where perception mode matters:
        - Visual mode: Convert to image message (data URL)
        - Verbal mode: Convert to text message
        """
        if observation.modality == "visual":
            prompt = observation.metadata.get("prompt") if observation.metadata else None
            if prompt and prompt != observation.data:
                # Keep visual context and image in one message to avoid "extra user turn"
                # that can bias models toward natural-language replies.
                content = {
                    "type": "image_with_prompt",
                    "image_data": observation.data,
                    "text": prompt,
                }
            else:
                # For visual, observation.data is already a data URL string
                # Pass it directly so LLM backend can handle it
                content = observation.data
        elif observation.modality == "verbal":
            # For verbal, extract text content
            content = observation.data
        else:
            # Default: stringify
            content = str(observation.data)
        
        return Message(
            role="user",
            content=content,
            metadata={"timestamp": observation.timestamp}
        )
    
    def _resolve_tool_name(self, tool_name: str) -> str:
        """Resolve tool name with exact then fuzzy match."""
        if tool_name in self.tools:
            return tool_name
        logger.warning(f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}")
        for available_tool in self.tools.keys():
            if available_tool in tool_name or tool_name in available_tool:
                return available_tool
        return tool_name

    def _parse_actions(self, response: Message) -> List[Action]:
        """
        Parse LLM response into one or more actions.
        
        Supported response formats:
        - {"tool_calls":[{"name":"...", "parameters": {...}}, ...], "reasoning":"..."}
        - {"tool_call":{"name":"...", "parameters": {...}}, "reasoning":"..."} (legacy)
        - Plain text (only if legacy_fallback=True)
        """
        content = response.content
        if isinstance(content, str):
            parsed = self._try_parse_structured_content(content)
            if parsed is not None:
                content = parsed

        if isinstance(content, list) and content:
            # Common variant: root is an array of tool calls.
            content = {"tool_calls": content}

        if isinstance(content, dict):
            reasoning = content.get("reasoning")
            if "actions" in content and isinstance(content["actions"], list):
                content = {"tool_calls": content["actions"], "reasoning": reasoning}

            if "name" in content and ("parameters" in content or "arguments" in content):
                single_params = content.get("parameters", content.get("arguments"))
                if isinstance(single_params, str):
                    try:
                        single_params = json.loads(single_params)
                    except Exception:
                        single_params = {}
                return [Action(
                    tool_name=self._resolve_tool_name(str(content.get("name", ""))),
                    parameters=single_params or {},
                    reasoning=reasoning,
                )]

            if "tool_calls" in content and isinstance(content["tool_calls"], list):
                actions: List[Action] = []
                for tool_call in content["tool_calls"]:
                    if not isinstance(tool_call, dict):
                        continue
                    name = tool_call.get("name")
                    params_obj = tool_call.get("parameters")
                    if not name and isinstance(tool_call.get("function"), dict):
                        fn = tool_call["function"]
                        name = fn.get("name")
                        params_obj = params_obj or fn.get("parameters") or fn.get("arguments")
                    if isinstance(params_obj, str):
                        try:
                            params_obj = json.loads(params_obj)
                        except Exception:
                            params_obj = {}
                    name = self._resolve_tool_name(str(name or ""))
                    parameters = params_obj or {}
                    actions.append(Action(tool_name=name, parameters=parameters, reasoning=reasoning))
                if actions:
                    return actions

            if "tool_call" in content and isinstance(content["tool_call"], dict):
                tool_call = content["tool_call"]
                name = tool_call.get("name")
                params_obj = tool_call.get("parameters")
                if not name and isinstance(tool_call.get("function"), dict):
                    fn = tool_call["function"]
                    name = fn.get("name")
                    params_obj = params_obj or fn.get("parameters") or fn.get("arguments")
                if isinstance(params_obj, str):
                    try:
                        params_obj = json.loads(params_obj)
                    except Exception:
                        params_obj = {}
                name = self._resolve_tool_name(str(name or ""))
                parameters = params_obj or {}
                return [Action(tool_name=name, parameters=parameters, reasoning=reasoning)]

            if "function_call" in content and isinstance(content["function_call"], dict):
                fn = content["function_call"]
                name = self._resolve_tool_name(str(fn.get("name", "")))
                parameters = fn.get("arguments") or fn.get("parameters") or {}
                if isinstance(parameters, str):
                    try:
                        parameters = json.loads(parameters)
                    except Exception:
                        parameters = {}
                return [Action(tool_name=name, parameters=parameters or {}, reasoning=reasoning)]

        if self.legacy_fallback and isinstance(content, str):
            for tool_name in self.tools.keys():
                if tool_name.lower() in content.lower():
                    logger.info(f"Extracted tool '{tool_name}' from text response")
                    return [Action(tool_name=tool_name, parameters={}, reasoning=content)]

        if isinstance(content, str):
            extracted = self._extract_action_from_text(content)
            if extracted is not None:
                return [extracted]

        logger.warning(f"Could not parse action from response: {str(content)[:200]}")
        return [Action(tool_name="noop", parameters={}, reasoning="No valid action found")]

    def _try_parse_structured_content(self, text: str) -> Optional[Any]:
        """Try to parse JSON-like tool-call content from plain text."""
        t = (text or "").strip()
        if not t:
            return None

        # Direct JSON object/array
        if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
            try:
                obj = json.loads(t)
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                try:
                    # Some backends return Python-literal dict/list strings with single quotes.
                    obj = ast.literal_eval(t)
                    if isinstance(obj, (dict, list)):
                        return obj
                except Exception:
                    pass

        # JSON inside markdown fenced block
        m = re.search(r"```(?:json)?\s*([\{\[][\s\S]*[\}\]])\s*```", t, re.IGNORECASE)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                pass

        return None

    @staticmethod
    def _parse_kv_parameter_text(text: str) -> Dict[str, Any]:
        """
        Parse loose key/value parameter text emitted by some model backends.

        Supported input examples:
        - `"max": 100`
        - `{"max": 100}`
        - `index=3, viewed_index=2`
        """
        src = (text or "").strip()
        if not src:
            return {}

        candidate = src
        if not candidate.startswith("{"):
            candidate = "{" + candidate + "}"

        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue

        out: Dict[str, Any] = {}
        for key in ("index", "min", "max", "viewed_index", "product_id"):
            m = re.search(
                rf"\b{key}\b\s*(?::|=)\s*([\"'][^\"']*[\"']|-?\d+(?:\.\d+)?)",
                src,
                re.IGNORECASE,
            )
            if not m:
                continue
            raw = m.group(1).strip()
            if raw.startswith(("'", '"')) and raw.endswith(("'", '"')) and len(raw) >= 2:
                val: Any = raw[1:-1]
            else:
                try:
                    val = float(raw)
                    if key in ("index", "viewed_index"):
                        val = int(val)
                except Exception:
                    val = raw
            out[key] = val
        return out

    def _extract_action_from_text(self, text: str) -> Optional[Action]:
        """Heuristic parser for plain-text model replies that mention tool intents."""
        t = (text or "").strip()
        if not t:
            return None
        low = t.lower()

        def _make(name: str, params: Optional[Dict[str, Any]] = None) -> Action:
            return Action(tool_name=self._resolve_tool_name(name), parameters=params or {}, reasoning=t[:500])

        # Prefer the first explicit function-style tool call in text order.
        # Example:
        #   select_product(1)
        #   select_product(2)
        #   recommend()
        # Should execute select_product(1) first instead of jumping to recommend().
        first_call = re.search(
            r"\b(select_product|next_page|prev_page|back|recommend|filter_price|filter_rating)\s*\(([^)]*)\)",
            low,
            re.IGNORECASE,
        )
        if first_call:
            tool_name = first_call.group(1).lower()
            arg_text = (first_call.group(2) or "").strip()
            if tool_name in ("next_page", "prev_page", "back", "recommend"):
                return _make(tool_name)
            if tool_name == "select_product":
                m = re.search(r"\d+", arg_text)
                if m:
                    return _make("select_product", {"index": int(m.group(0))})
            if tool_name == "filter_price":
                nums = re.findall(r"-?\d+(?:\.\d+)?", arg_text)
                if len(nums) >= 2:
                    a, b = float(nums[0]), float(nums[1])
                    lo, hi = (a, b) if a <= b else (b, a)
                    return _make("filter_price", {"min": lo, "max": hi})
                if len(nums) == 1:
                    return _make("filter_price", {"max": float(nums[0])})
            if tool_name == "filter_rating":
                m = re.search(r"-?\d+(?:\.\d+)?", arg_text)
                if m:
                    return _make("filter_rating", {"min": float(m.group(0))})

        # functions.<tool>:<step>_${{...}} style (seen in some providers)
        fn_match = re.search(
            r"(?:functions\.)?"
            r"(select_product|next_page|prev_page|back|recommend|filter_price|filter_rating)"
            r"\s*(?::\d+)?"
            r"(?P<tail>[\s\S]{0,600})",
            t,
            re.IGNORECASE,
        )
        if fn_match:
            tool_name = fn_match.group(1)
            tail = fn_match.group("tail") or ""
            parsed_params: Dict[str, Any] = {}
            for pat in (
                r"_\$\{\{(?P<body>[\s\S]*?)\}\}",
                r"\$\{\{(?P<body>[\s\S]*?)\}\}",
                r"\$\{(?P<body>[\s\S]*?)\}",
                r"\((?P<body>[\s\S]*?)\)",
                r"\{(?P<body>[\s\S]*?)\}",
            ):
                m = re.search(pat, tail, re.IGNORECASE)
                if not m:
                    continue
                candidate = m.group("body") or ""
                parsed_params = self._parse_kv_parameter_text(candidate)
                if parsed_params:
                    break
            # Handle positional function-call args when key-value parsing fails.
            if not parsed_params:
                if tool_name == "select_product":
                    m = re.search(r"\d+", tail)
                    if m:
                        parsed_params = {"index": int(m.group(0))}
                elif tool_name == "filter_price":
                    nums = re.findall(r"-?\d+(?:\.\d+)?", tail)
                    if len(nums) >= 2:
                        a, b = float(nums[0]), float(nums[1])
                        lo, hi = (a, b) if a <= b else (b, a)
                        parsed_params = {"min": lo, "max": hi}
                    elif len(nums) == 1:
                        parsed_params = {"max": float(nums[0])}
                elif tool_name == "filter_rating":
                    m = re.search(r"-?\d+(?:\.\d+)?", tail)
                    if m:
                        parsed_params = {"min": float(m.group(0))}
            if not parsed_params:
                parsed_params = self._parse_kv_parameter_text(tail)
            if tool_name in ("next_page", "prev_page", "back", "recommend") and not parsed_params:
                return _make(tool_name)
            if parsed_params:
                return _make(tool_name, parsed_params)
            # For parameterized tools, fall through to other heuristics instead
            # of creating an empty-parameter action.

        # Explicitly listed actions without parameters.
        for name in ("next_page", "prev_page", "back", "recommend"):
            if re.search(rf"\b{name}\b", low):
                return _make(name)

        # select_product(index)
        m = re.search(r"\bselect_product\b(?:\s*:\s*\d+\b)?[^\d]{0,40}(\d+)", low)
        if m:
            return _make("select_product", {"index": int(m.group(1))})
        m = re.search(r"\b(?:select|choose|open|click)\b[^\d]{0,20}(?:product|item)?[^\d]{0,20}(\d+)\b", low)
        if m:
            return _make("select_product", {"index": int(m.group(1))})

        # filter_price(min,max) / under / above
        m = re.search(r"\bfilter_price\b(?:\s*:\s*\d+\b)?[^\d]{0,40}(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)", low)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
            return _make("filter_price", {"min": lo, "max": hi})
        m = re.search(r"\b(?:under|below|less than)\s*\$?\s*(\d+(?:\.\d+)?)", low)
        if m:
            return _make("filter_price", {"max": float(m.group(1))})
        m = re.search(r"\b(?:over|above|more than)\s*\$?\s*(\d+(?:\.\d+)?)", low)
        if m:
            return _make("filter_price", {"min": float(m.group(1))})

        # filter_rating(min)
        m = re.search(r"\bfilter_rating\b(?:\s*:\s*\d+\b)?[^\d]{0,40}(\d(?:\.\d+)?)", low)
        if m:
            return _make("filter_rating", {"min": float(m.group(1))})
        m = re.search(r"\b(\d(?:\.\d+)?)\s*(?:\+?\s*stars?|star\s*rating)\b", low)
        if m:
            return _make("filter_rating", {"min": float(m.group(1))})

        return None

    def _parse_action(self, response: Message) -> Action:
        """Backward-compatible single-action parser."""
        actions = self._parse_actions(response)
        return actions[0] if actions else Action(tool_name="noop", parameters={}, reasoning="No valid action found")
    
    def _trim_history(self) -> None:
        """Trim message history to stay within max_history limit."""
        if len(self.message_history) > self.max_history:
            # Keep system message and recent messages
            system_messages = [
                msg for msg in self.message_history if msg.role == "system"
            ]
            recent_messages = self.message_history[-self.max_history:]
            
            # Combine
            self.message_history = system_messages + recent_messages
            
            logger.debug(f"Trimmed history to {len(self.message_history)} messages")
