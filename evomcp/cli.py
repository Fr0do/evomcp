"""evomcp CLI — `evomcp <subcommand>`.

Usage:
    evomcp serve                       # stdio MCP server (for Claude Code)
    evomcp serve --http [--port 8765]  # HTTP/SSE MCP server

    evomcp run gepa   configs/gepa.yaml   [--resume] [--foreground]
    evomcp run evox   configs/evox.yaml   [--resume] [--foreground]
    evomcp run hybrid configs/hybrid.yaml [--resume] [--foreground]

    evomcp status [<run_id>]
    evomcp export <run_id>
    evomcp inspect <bundle_dir>
    evomcp slots [--project <dir>]
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="evomcp",
        description="GEPA + EvoX evolutionary optimizer MCP server / CLI",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # serve
    sp = sub.add_parser("serve", help="Start the MCP server")
    sp.add_argument("--http", action="store_true", help="HTTP/SSE transport instead of stdio")
    sp.add_argument("--port", type=int, default=8765)

    # run
    rp = sub.add_parser("run", help="Start an evolution run")
    rp.add_argument("mode", choices=["gepa", "evox", "hybrid"])
    rp.add_argument("config_path")
    rp.add_argument("--resume", action="store_true")
    rp.add_argument("--foreground", action="store_true", help="Block until complete")

    # status
    stp = sub.add_parser("status", help="Show run status")
    stp.add_argument("run_id", nargs="?", default="")

    # export
    ep = sub.add_parser("export", help="Export best candidate from a run")
    ep.add_argument("run_id")

    # inspect
    ip = sub.add_parser("inspect", help="Inspect a trace bundle")
    ip.add_argument("bundle_dir")

    # slots
    slp = sub.add_parser("slots", help="List registered search-space slots")
    slp.add_argument("--project", default=".", help="Project root directory")

    args = parser.parse_args()

    if args.cmd == "serve":
        from evomcp.server import serve
        serve(http=args.http, port=getattr(args, "port", 8765))

    elif args.cmd == "run":
        from evomcp.server import tool_evolve_run
        result = tool_evolve_run(
            args.mode,
            args.config_path,
            resume=args.resume,
            background=not args.foreground,
        )
        print(json.dumps(result, indent=2))

    elif args.cmd == "status":
        from evomcp.server import tool_evolve_status
        result = tool_evolve_status(args.run_id or None)
        print(json.dumps(result, indent=2))

    elif args.cmd == "export":
        from evomcp.server import tool_evolve_export
        result = tool_evolve_export(args.run_id)
        print(json.dumps(result, indent=2))

    elif args.cmd == "inspect":
        from evomcp.server import tool_evolve_inspect
        result = tool_evolve_inspect(args.bundle_dir)
        print(json.dumps(result, indent=2))

    elif args.cmd == "slots":
        from evomcp.server import tool_evolve_list_slots
        result = tool_evolve_list_slots(args.project)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
