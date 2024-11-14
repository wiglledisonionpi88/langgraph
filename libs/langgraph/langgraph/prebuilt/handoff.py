from typing import Optional

from langchain_core.tools import StructuredTool

from langgraph.graph import GraphCommand


class HandoffTool(StructuredTool):
    goto: str

    @classmethod
    def from_function(cls, goto: str, *args, **kwargs):
        kwargs["goto"] = goto
        return super().from_function(*args, **kwargs)


def create_handoff_tool(
    goto: str, name: Optional[str] = None, description: Optional[str] = None
) -> HandoffTool:
    """Create a tool that can hand off control to another node / agent."""

    def func():
        return f"Transferred to '{goto}'!", GraphCommand(goto=goto)

    if description is None:
        description = f"Transfer to '{goto}'. Do not ask any details."

    if name is None:
        name = goto

    transfer_tool = HandoffTool.from_function(
        goto,
        func,
        name=name,
        description=description,
        response_format="content_and_artifact",
        args_schema=None,
        infer_schema=True,
        return_direct=True,
    )
    return transfer_tool
