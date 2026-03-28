"""Integration tests for MiniMax provider in prompt_optimize.py.

These tests verify end-to-end behavior with the actual MiniMax API.
They require a valid MINIMAX_API_KEY environment variable to run.

Usage:
    MINIMAX_API_KEY=your-key python -m pytest tests/test_integration_minimax.py -v
"""

import os
import sys
import unittest

# Add inference directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))

from prompt_optimize import PROVIDER_PRESETS, convert_prompt, get_system_instruction, get_user_assistant_pairs


MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")


@unittest.skipUnless(MINIMAX_API_KEY, "MINIMAX_API_KEY not set")
class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests that call the actual MiniMax API."""

    def test_minimax_prompt_enhancement_english(self):
        """Test that MiniMax can enhance an English prompt."""
        base_url, model = PROVIDER_PRESETS["minimax"]
        system_instruction = get_system_instruction("cogview4")
        user_assistant_pairs = get_user_assistant_pairs("cogview4")

        result = convert_prompt(
            api_key=MINIMAX_API_KEY,
            base_url=base_url,
            prompt="a cat sitting on a windowsill",
            system_instruction=system_instruction,
            model=model,
            user_assistant_pairs=user_assistant_pairs,
        )

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 20, "Enhanced prompt should be longer than the input")

    def test_minimax_prompt_enhancement_chinese(self):
        """Test that MiniMax can enhance a Chinese prompt."""
        base_url, model = PROVIDER_PRESETS["minimax"]
        system_instruction = get_system_instruction("cogview4")
        user_assistant_pairs = get_user_assistant_pairs("cogview4")

        result = convert_prompt(
            api_key=MINIMAX_API_KEY,
            base_url=base_url,
            prompt="一只猫坐在窗台上",
            system_instruction=system_instruction,
            model=model,
            user_assistant_pairs=user_assistant_pairs,
        )

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10, "Enhanced prompt should be longer than the input")

    def test_minimax_cogview3_mode(self):
        """Test MiniMax with CogView3 system instructions."""
        base_url, model = PROVIDER_PRESETS["minimax"]
        system_instruction = get_system_instruction("cogview3")
        user_assistant_pairs = get_user_assistant_pairs("cogview3")

        result = convert_prompt(
            api_key=MINIMAX_API_KEY,
            base_url=base_url,
            prompt="a sunset over the ocean",
            system_instruction=system_instruction,
            model=model,
            user_assistant_pairs=user_assistant_pairs,
        )

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 20)


if __name__ == "__main__":
    unittest.main()
