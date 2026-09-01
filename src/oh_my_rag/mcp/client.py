import argparse
import asyncio
import json

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from oh_my_rag.config import MCP_HTTP_PORT


def stdio_transport() -> StdioTransport:
    """Spawn the server the same way an MCP host would."""
    return StdioTransport(command="uv", args=["run", "oh-my-rag-mcp"])


def make_client(http: bool = False, port: int = MCP_HTTP_PORT) -> Client:
    return Client(f"http://127.0.0.1:{port}/mcp" if http else stdio_transport())


async def call(tool: str, arguments: dict, http: bool = False, port: int = MCP_HTTP_PORT):
    async with make_client(http=http, port=port) as client:
        result = await client.call_tool(tool, arguments)
        return result.data


async def list_tools(http: bool = False, port: int = MCP_HTTP_PORT):
    async with make_client(http=http, port=port) as client:
        return [(t.name, (t.description or "").split("\n")[0]) for t in await client.list_tools()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the oh-my-rag MCP server.")
    parser.add_argument("tool", nargs="?", help="Tool name; omit to list tools.")
    parser.add_argument("args", nargs="*", help="key=value pairs.")
    parser.add_argument("--http", action="store_true", help="Connect over HTTP.")
    parser.add_argument("--port", type=int, default=MCP_HTTP_PORT)
    opts = parser.parse_args()

    if not opts.tool:
        for name, summary in asyncio.run(list_tools(opts.http, opts.port)):
            print(f"{name:<14} {summary}")
        return

    arguments = {}
    for pair in opts.args:
        key, _, value = pair.partition("=")
        if value.lstrip("-").isdigit():
            arguments[key] = int(value)
        elif value.lower() in {"true", "false"}:
            arguments[key] = value.lower() == "true"
        else:
            arguments[key] = value

    print(json.dumps(asyncio.run(call(opts.tool, arguments, opts.http, opts.port)),
                     indent=2, default=str))


if __name__ == "__main__":
    main()
