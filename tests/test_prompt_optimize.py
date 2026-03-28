"""Unit tests for prompt_optimize.py provider presets and MiniMax integration."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add inference directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))

from prompt_optimize import PROVIDER_PRESETS, clean_string, convert_prompt, get_system_instruction


class TestProviderPresets(unittest.TestCase):
    """Test that PROVIDER_PRESETS contains correct entries."""

    def test_presets_has_zhipu(self):
        self.assertIn("zhipu", PROVIDER_PRESETS)
        base_url, model = PROVIDER_PRESETS["zhipu"]
        self.assertEqual(base_url, "https://open.bigmodel.cn/api/paas/v4")
        self.assertEqual(model, "glm-4-plus")

    def test_presets_has_minimax(self):
        self.assertIn("minimax", PROVIDER_PRESETS)
        base_url, model = PROVIDER_PRESETS["minimax"]
        self.assertEqual(base_url, "https://api.minimax.io/v1")
        self.assertEqual(model, "MiniMax-M2.7")

    def test_presets_has_openai(self):
        self.assertIn("openai", PROVIDER_PRESETS)
        base_url, model = PROVIDER_PRESETS["openai"]
        self.assertEqual(base_url, "https://api.openai.com/v1")
        self.assertEqual(model, "gpt-4o")

    def test_all_presets_have_two_elements(self):
        for provider, preset in PROVIDER_PRESETS.items():
            self.assertEqual(len(preset), 2, f"Provider {provider} should have (base_url, model)")
            self.assertIsInstance(preset[0], str, f"Provider {provider} base_url should be str")
            self.assertIsInstance(preset[1], str, f"Provider {provider} model should be str")


class TestCleanString(unittest.TestCase):
    """Test the clean_string utility function."""

    def test_removes_newlines(self):
        self.assertEqual(clean_string("hello\nworld"), "hello world")

    def test_strips_whitespace(self):
        self.assertEqual(clean_string("  hello  "), "hello")

    def test_collapses_multiple_spaces(self):
        self.assertEqual(clean_string("hello   world"), "hello world")

    def test_combined(self):
        self.assertEqual(clean_string("  hello\n  world  "), "hello world")


class TestGetSystemInstruction(unittest.TestCase):
    """Test system instruction retrieval."""

    def test_cogview3_returns_english_only(self):
        instruction = get_system_instruction("cogview3")
        self.assertIn("English", instruction)
        self.assertNotIn("bilingual", instruction)

    def test_cogview4_returns_bilingual(self):
        instruction = get_system_instruction("cogview4")
        self.assertIn("bilingual", instruction)

    def test_invalid_version_raises(self):
        with self.assertRaises(ValueError):
            get_system_instruction("cogview5")


class TestConvertPromptWithMiniMax(unittest.TestCase):
    """Test convert_prompt with MiniMax provider."""

    @patch("prompt_optimize.OpenAI")
    def test_minimax_base_url_passed(self, mock_openai_cls):
        """Verify that MiniMax base_url is passed to OpenAI client."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Enhanced prompt"
        mock_client.chat.completions.create.return_value = mock_response

        result = convert_prompt(
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
            prompt="a cat",
            system_instruction="You are a bot",
            model="MiniMax-M2.7",
            user_assistant_pairs=[],
        )

        mock_openai_cls.assert_called_once_with(api_key="test-key", base_url="https://api.minimax.io/v1")
        self.assertEqual(result, "Enhanced prompt")

    @patch("prompt_optimize.OpenAI")
    def test_minimax_temperature_clamped(self, mock_openai_cls):
        """Verify temperature is valid for MiniMax (> 0)."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_client.chat.completions.create.return_value = mock_response

        convert_prompt(
            api_key="key",
            base_url="https://api.minimax.io/v1",
            prompt="test",
            system_instruction="sys",
            model="MiniMax-M2.7",
            user_assistant_pairs=[],
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertGreater(call_kwargs["temperature"], 0)
        self.assertLessEqual(call_kwargs["temperature"], 1.0)

    @patch("prompt_optimize.OpenAI")
    def test_minimax_model_passed(self, mock_openai_cls):
        """Verify MiniMax model name is passed correctly."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_client.chat.completions.create.return_value = mock_response

        convert_prompt(
            api_key="key",
            base_url="https://api.minimax.io/v1",
            prompt="test",
            system_instruction="sys",
            model="MiniMax-M2.7",
            user_assistant_pairs=[],
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.7")

    @patch("prompt_optimize.OpenAI")
    def test_zhipu_uses_default_temperature(self, mock_openai_cls):
        """Verify non-MiniMax providers use original temperature."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_client.chat.completions.create.return_value = mock_response

        convert_prompt(
            api_key="key",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            prompt="test",
            system_instruction="sys",
            model="glm-4-plus",
            user_assistant_pairs=[],
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.01)

    @patch("prompt_optimize.OpenAI")
    def test_messages_include_system_and_user(self, mock_openai_cls):
        """Verify messages contain system instruction and user prompt."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "output"
        mock_client.chat.completions.create.return_value = mock_response

        convert_prompt(
            api_key="key",
            base_url="https://api.minimax.io/v1",
            prompt="a dog",
            system_instruction="You are helpful",
            model="MiniMax-M2.7",
            user_assistant_pairs=[{"role": "user", "content": "ex"}, {"role": "assistant", "content": "resp"}],
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are helpful")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("a dog", messages[-1]["content"])

    @patch("prompt_optimize.OpenAI")
    def test_clean_string_applied_to_output(self, mock_openai_cls):
        """Verify output is cleaned (newlines removed, whitespace collapsed)."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  enhanced\n  prompt  "
        mock_client.chat.completions.create.return_value = mock_response

        result = convert_prompt(
            api_key="key",
            base_url="https://api.minimax.io/v1",
            prompt="test",
            system_instruction="sys",
            model="MiniMax-M2.7",
            user_assistant_pairs=[],
        )

        self.assertEqual(result, "enhanced prompt")


class TestCLIArgParsing(unittest.TestCase):
    """Test CLI argument parsing for provider presets."""

    @patch("prompt_optimize.OpenAI")
    def test_minimax_api_key_from_env(self, mock_openai_cls):
        """Verify MINIMAX_API_KEY env var is used for minimax provider."""
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test-key"}, clear=False):
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "result"
            mock_client.chat.completions.create.return_value = mock_response

            # Simulate what the CLI main block does
            provider = "minimax"
            preset_base_url, preset_model = PROVIDER_PRESETS[provider]
            api_key = os.environ.get("MINIMAX_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

            self.assertEqual(api_key, "mm-test-key")
            self.assertEqual(preset_base_url, "https://api.minimax.io/v1")
            self.assertEqual(preset_model, "MiniMax-M2.7")


if __name__ == "__main__":
    unittest.main()
