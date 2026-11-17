"""Generate Pro licence keys.

    python tools/make_keys.py 20

Give the keys to buyers; paste the printed LICENSE_HASHES line into
.streamlit/secrets.toml (or the Streamlit Cloud "Secrets" box). Only hashes are
stored on the server, so a leaked secrets file does not leak usable keys.
"""
import hashlib
import secrets
import string
import sys

ALPHABET = "".join(c for c in string.ascii_uppercase + string.digits if c not in "O0I1")


def new_key() -> str:
    return "STZ-" + "-".join("".join(secrets.choice(ALPHABET) for _ in range(4)) for _ in range(3))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    keys = [new_key() for _ in range(n)]
    print("# Keys (give one to each buyer):")
    print("\n".join(keys))
    print("\n# Add to secrets.toml (append to any existing list):")
    print("LICENSE_HASHES = [" + ", ".join(f'"{hashlib.sha256(k.encode()).hexdigest()}"' for k in keys) + "]")
