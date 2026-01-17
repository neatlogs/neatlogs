"""
Neatlogs Azure OpenAI Example
========================
This example demonstrates how to use Neatlogs with OpenAI API calls.
Traces will be written to a local file (neatlogs.jsonl).
"""

import os
import sys

# Add parent directory to path to import local neatlogs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
import neatlogs

load_dotenv()
neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY"),
    tags=["v3", "azure-openai", "demo"],
    instrumentations=["openai"],
)

# Initialize Azure OpenAI client if available
try:
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    )
    USE_AZURE = True
except Exception as e:
    print(f"Azure OpenAI not available: {e}")
    print("Using mock responses...")
    USE_AZURE = False


class ChatBot:
    def __init__(self):
        self.history = []

    def generate_response(self, prompt):
        """Generate a response using AI or fallback to simple replies"""
        try:
            if USE_AZURE:
                response = client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            else:
                # Simple mock responses
                prompt_lower = prompt.lower()
                if "hello" in prompt_lower or "hi" in prompt_lower:
                    return "Hello! How can I help you?"
                elif "how are you" in prompt_lower:
                    return "I'm doing well, thank you!"
                elif "python" in prompt_lower:
                    return "Python is a versatile programming language!"
                else:
                    return f"You said: '{prompt}'. Configure Azure OpenAI for real responses."

        except Exception as e:
            return f"Error: {str(e)}"

    def run(self):
        """Main chat loop"""
        print("ChatBot started. Type 'quit' to exit, 'help' for commands.")

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ["quit", "q", "exit"]:
                    break
                elif user_input.lower() in ["help", "h"]:
                    print("Commands: quit, clear, history")
                    continue
                elif user_input.lower() == "clear":
                    self.history.clear()
                    print("History cleared.")
                    continue
                elif user_input.lower() in ["history", "h"]:
                    if not self.history:
                        print("No history yet.")
                    else:
                        for i, msg in enumerate(self.history[-5:], 1):
                            print(f"{i}. {msg}")
                    continue

                # Generate response
                print("AI: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)

                # Keep last 10 messages
                self.history.append(f"You: {user_input}")
                self.history.append(f"AI: {response}")
                self.history = self.history[-20:]  # Keep last 10 exchanges

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

        print("Chat ended.")


def main():
    """Main entry point"""
    bot = ChatBot()
    bot.run()
    # neatlogs.shutdown()  # Ensure clean shutdown of telemetry


if __name__ == "__main__":
    main()
