"""
Hosted-mode example: CognisClient against the Lyzr memory service.

Setup:
    export LYZR_API_KEY="your-lyzr-studio-api-key"
    # optional, for on-prem deployments (default: https://memory.studio.lyzr.ai):
    # export BASE_MEMORY_URL="https://memory.your-company.internal"

Run:
    python examples/hosted_client.py
"""

from cognis import (
    CognisClient,
    CognisAuthenticationError,
    CognisConnectionError,
    CognisPermissionError,
)


def main():
    try:
        client = CognisClient(owner_id="demo_user", agent_id="demo_agent")
    except CognisAuthenticationError as e:
        print(f"Auth setup problem: {e}")
        return

    print(f"Connected to {client.base_url}")

    try:
        print("Health:", client.health())

        # Add messages — the service extracts memories server-side
        resp = client.add(
            [
                {"role": "user", "content": "My name is Alice and I work at Google."},
                {"role": "user", "content": "I love hiking on weekends."},
            ],
            sync_extraction=True,
        )
        print("Add:", resp)

        # Search
        resp = client.search("Where does Alice work?")
        for r in resp["results"]:
            print(f"  [{r.get('score')}] {r['content']}")

        # LLM-ready context
        ctx = client.get_context([{"role": "user", "content": "What do you know about me?"}])
        print("Context string:\n", ctx["context_string"])

        print("Total memories:", client.count())

    except CognisPermissionError as e:
        print(f"Your API key lacks a required permission: {e}")
    except CognisConnectionError as e:
        print(f"Could not reach the memory service: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
