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
    
    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with validation.
        
        This method:
        1. Validates parameters
        2. Calls the implementation
        3. Wraps result in ToolResult
        """
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
            logger.error(f"Parameter validation failed: {e}")
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
