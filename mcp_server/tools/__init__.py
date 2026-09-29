"""MCP tool registry. server.py registers every function in ALL_TOOLS onto
the FastMCP app — it never imports individual tool modules directly.

To add a new tool: write a module here (a plain function; its docstring
becomes the MCP tool description shown to agents), import it below, and
add it to ALL_TOOLS. server.py needs zero changes.
"""
from tools.write_file import write_file
from tools.run_shell_command import run_shell_command
from tools.run_tests import run_tests
from tools.check_webapp import check_webapp

ALL_TOOLS = [write_file, run_shell_command, run_tests, check_webapp]

__all__ = ["ALL_TOOLS", "write_file", "run_shell_command", "run_tests", "check_webapp"]
