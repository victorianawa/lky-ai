"""Verify Gemini API access, list available models, and test the configured GEMINI_MODEL."""
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    configured_model = os.getenv("GEMINI_MODEL")

    if not api_key or api_key == "your_api_key":
        print("ERROR: GEMINI_API_KEY is not set. Please add it to your .env file.", file=sys.stderr)
        sys.exit(1)

    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai is not installed. Run `pip install -r requirements.txt` first.", file=sys.stderr)
        sys.exit(1)

    print("Authenticating with Gemini API...")
    client = genai.Client(api_key=api_key)

    print("\nFetching models available for text generation...")
    try:
        models = list(client.models.list())
        text_models = [
            m for m in models
            if getattr(m, "supported_actions", None) and "generateContent" in m.supported_actions
            or "generateContent" in getattr(m, "supported_generation_methods", [])
            or "gemini" in (m.name or "").lower()
        ]
        for m in text_models:
            name = getattr(m, "name", str(m))
            display_name = getattr(m, "display_name", "")
            print(f"  - {name} ({display_name})" if display_name else f"  - {name}")
    except Exception as e:
        print(f"WARNING: Could not list models: {e}")

    if configured_model and configured_model != "your_selected_gemini_model":
        print(f"\nTesting configured model: '{configured_model}'...")
        try:
            resp = client.models.generate_content(
                model=configured_model,
                contents="Respond with 'OK' if you can read this.",
            )
            print(f"Success! Model response: {resp.text.strip()}")
        except Exception as e:
            print(f"ERROR: Failed to query model '{configured_model}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("\nNOTE: GEMINI_MODEL is not configured in .env yet. Pick one of the models listed above.")


if __name__ == "__main__":
    main()

