"""
Client MCP de demo — se connecte au serveur mcp_server.py via stdio,
liste les outils exposes et appelle meteo_actuelle("Paris") pour preuve
d'interoperabilite.

Usage : python test_mcp_client.py
"""

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def demo():
    print("=" * 70)
    print("DEMO MCP CLIENT — appel a mcp_server.py via protocole stdio")
    print("=" * 70)

    # Spawn mcp_server.py comme sous-processus et parle-lui en stdio
    params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env=None,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # Handshake MCP
            await session.initialize()
            print("\n[OK] Handshake MCP reussi — client connecte au serveur\n")

            # Lister les outils exposes
            tools_resp = await session.list_tools()
            print(f"[OK] Outils exposes par le serveur : {len(tools_resp.tools)}")
            for t in tools_resp.tools:
                print(f"    - {t.name}")

            # Appeler un outil via le protocole MCP
            print("\n" + "-" * 70)
            print("APPEL MCP : meteo_actuelle(ville='Paris')")
            print("-" * 70)
            result = await session.call_tool("meteo_actuelle", {"ville": "Paris"})
            for item in result.content:
                if hasattr(item, "text"):
                    print(item.text)

            # Second exemple : inventaire du corpus
            print("\n" + "-" * 70)
            print("APPEL MCP : inventaire_corpus()")
            print("-" * 70)
            result = await session.call_tool("inventaire_corpus", {})
            for item in result.content:
                if hasattr(item, "text"):
                    print(item.text[:500])

    print("\n" + "=" * 70)
    print("DEMO TERMINEE — le serveur MCP est bien interoperable.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo())
