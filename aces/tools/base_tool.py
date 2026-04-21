"""
Base Tool Implementation

Provides a base class for MCP-compliant tools with parameter validation.
"""

from typing import Any, Dict
import logging
import jsonschema

from aces.core.protocols import Tool, ToolSchema, ToolResult


logger = logging.getLogger(__name__)


class BaseTool(Tool):
    """
    Base class for MCP-compliant tools.
    
    Provides common functionality like schema validation.
    Subclasses only need to implement _execute_impl().
    """
    
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        """
        Initialize base tool.
        
        Args:
            name: Tool name (used by LLM to call it)
            description: What the tool does
            input_schema: JSON Schema for parameters
        """
        self._name = name
        self._description = description
        self._input_schema = input_schema
        
        logger.info(f"Initialized tool: {name}")
    
    def get_schema(self) -> ToolSchema:
        """Return the tool's schema."""
        return ToolSchema(
            name=self._name,
            description=self._description,
            input_schema=self._input_schema,
        )
    
    def _coerce_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coerce parameters to match schema (e.g. LLM may return "1" for integer).
        Override in subclasses for custom coercion.
        """
        if not parameters:
            return parameters
        schema = self._input_schema
        props = schema.get("properties", {})
        result = dict(parameters)
        for key, spec in props.items():
            if key not in result:
                continue
            val = result[key]
            if val is None:
                continue
            type_spec = spec.get("type")
            if isinstance(type_spec, list):
                type_spec = type_spec[0] if type_spec else None
            if type_spec == "integer" and isinstance(val, str):
                try:
                    result[key] = int(float(val))
                except (ValueError, TypeError):
                    pass
            elif type_spec == "number" and isinstance(val, str):
                try:
                    result[key] = float(val)
                except (ValueError, TypeError):
                    pass
        return result

    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with validation.
        
        This method:
        1. Coerces parameters to match schema
        2. Validates parameters
        3. Calls the implementation
        4. Wraps result in ToolResult
        """
        parameters = self._coerce_parameters(parameters or {})
        # Validate parameters
        if not self.validate_parameters(parameters):
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid parameters for tool {self._name}",
            )
        
        try:
            # Call implementation
            data = self._execute_impl(parameters)

            # If tool returns a structured payload with explicit success flag,
            # propagate it to ToolResult.success so orchestrator error handling works.
            if isinstance(data, dict) and "success" in data:
                payload_success = bool(data.get("success"))
                err = data.get("error")
                return ToolResult(
                    success=payload_success,
                    data=data,
                    error=str(err) if (err and not payload_success) else None,
                    metadata={"tool": self._name}
                )

            return ToolResult(
                success=True,
                data=data,
                metadata={"tool": self._name}
            )
        except Exception as e:
            logger.error(f"Tool {self._name} failed: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
            )
    
    async def aexecute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Async version of execute."""
        # For sync tools, just call execute
        # Subclasses can override for true async
        return self.execute(parameters)
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters against JSON schema."""
        try:
            jsonschema.validate(
                instance=parameters,
                schema=self._input_schema,
            )
            return True
        except jsonschema.ValidationError as e:
            logger.error(f"Parameter validation failed for {self._name}: {e}. params={parameters}")
            return False
    
    # ========================================================================
    # Abstract Method (Subclasses Must Implement)
    # ========================================================================
    
    def _execute_impl(self, parameters: Dict[str, Any]) -> Any:
        """
        Implementation-specific execution logic.
        
        Subclasses MUST implement this method.
        
        Args:
            parameters: Validated parameters
            
        Returns:
            Result data (will be wrapped in ToolResult)
        """
        raise NotImplementedError(
            f"Tool {self._name} must implement _execute_impl()"
        )
