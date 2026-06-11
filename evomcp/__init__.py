"""evomcp — GEPA + EvoX evolutionary optimizer, served as an MCP server.

This package is the **canonical home** of the optimization protocol.
Projects (e.g. hanfu-code) declare their project-specific search spaces and
evaluators; evomcp provides the shared infrastructure.

Quick install:
    pip install -e ~/experiments/evomcp

Quick start (MCP server):
    evomcp serve                        # stdio MCP server

Quick start (CLI):
    evomcp run gepa configs/gepa.yaml
    evomcp run evox configs/evox.yaml
    evomcp run hybrid configs/hybrid.yaml
    evomcp status
    evomcp export <run_id>
"""
__version__ = "0.1.0"
