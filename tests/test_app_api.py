import asyncio
import unittest
from unittest.mock import patch

from app import ChatRequest, chat_endpoint


class ChatEndpointFallbackTests(unittest.TestCase):
    @patch("app.create_llm")
    def test_chat_endpoint_returns_explicit_fallback_flag(self, mock_create_llm):
        mock_create_llm.side_effect = RuntimeError("quota exceeded")

        response = asyncio.run(
            chat_endpoint(
                ChatRequest(
                    message="How do I return an item?",
                    history=[],
                    provider="gemini",
                    api_key="bad-key",
                )
            )
        )

        self.assertIn("fallback", response)
        self.assertTrue(response["fallback"])


if __name__ == "__main__":
    unittest.main()
