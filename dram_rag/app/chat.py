from __future__ import annotations

import argparse
from pathlib import Path

from ..agent.graph import AdaptiveRAGAgent
from ..config import load_settings
from ..index.stores import IndexBundle


def _main() -> None:
    p = argparse.ArgumentParser(description="Chat with the DRAM Adaptive RAG agent")
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "config" / "settings.yaml"), help="Path to settings.yaml")
    p.add_argument("--index", default=None, help="Index directory (overrides settings.paths.index_dir)")

    args = p.parse_args()

    settings = load_settings(args.config)
    index_dir = args.index or settings.paths.index_dir
    bundle = IndexBundle.load(index_dir)

    agent = AdaptiveRAGAgent(settings, bundle)

    print(f"Loaded index: {index_dir}")
    print("Type your question. Press Ctrl+C or enter empty line to exit.")

    while True:
        try:
            q = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye")
            return
        if not q:
            print("Bye")
            return

        state = agent.run(q)
        print("\nA>\n")
        print(state.generation)

        # Optional trace for debugging
        print("\n--- Trace (debug) ---")
        for t in state.trace:
            print(t)


if __name__ == "__main__":
    _main()
