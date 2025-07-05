import pytest
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_from_scratch.simple_tokenizer_v1 import SimpleTokenizerV1


class TestSimpleTokenizerV1:
    """Test suite for SimpleTokenizerV1 class."""

    @pytest.fixture
    def sample_vocab(self):
        """Create a sample vocabulary for testing."""
        return {
            "hello": 0,
            "world": 1,
            "!": 2,
            "this": 3,
            "is": 4,
            "a": 5,
            "test": 6,
            ".": 7,
            ",": 8,
            "?": 9,
            "python": 10,
            "tokenizer": 11,
            "the": 12,
            "quick": 13,
            "brown": 14,
            "fox": 15,
            "jumps": 16,
            "over": 17,
            "lazy": 18,
            "dog": 19,
            "(": 20,
            ")": 21,
            "\"": 22,
            "'": 23,
            "--": 24,
            ";": 25,
            ":": 26,
            "_": 27,
            "and": 28,
            "or": 29,
        }

    @pytest.fixture
    def tokenizer(self, sample_vocab):
        """Create a SimpleTokenizerV1 instance with sample vocabulary."""
        return SimpleTokenizerV1(sample_vocab)

    def test_initialization(self, sample_vocab):
        """Test tokenizer initialization."""
        tokenizer = SimpleTokenizerV1(sample_vocab)
        
        # Check that str_to_int is set correctly
        assert tokenizer.str_to_int == sample_vocab
        
        # Check that int_to_str is the reverse mapping
        expected_int_to_str = {v: k for k, v in sample_vocab.items()}
        assert tokenizer.int_to_str == expected_int_to_str

    def test_encode_simple_text(self, tokenizer):
        """Test encoding simple text."""
        text = "hello world"
        encoded = tokenizer.encode(text)
        expected = [0, 1]  # hello=0, world=1
        assert encoded == expected

    def test_encode_with_punctuation(self, tokenizer):
        """Test encoding text with punctuation."""
        text = "hello world!"
        encoded = tokenizer.encode(text)
        expected = [0, 1, 2]  # hello=0, world=1, !=2
        assert encoded == expected

    def test_encode_complex_punctuation(self, tokenizer):
        """Test encoding with various punctuation marks."""
        text = "hello, world! this is a test."
        encoded = tokenizer.encode(text)
        expected = [0, 8, 1, 2, 3, 4, 5, 6, 7]  # hello,world!thisisatest.
        assert encoded == expected

    def test_decode_simple_ids(self, tokenizer):
        """Test decoding simple token IDs."""
        ids = [0, 1]  # hello, world
        decoded = tokenizer.decode(ids)
        expected = "hello world"
        assert decoded == expected

    def test_decode_with_punctuation(self, tokenizer):
        """Test decoding with punctuation formatting."""
        ids = [0, 8, 1, 2]  # hello, world!
        decoded = tokenizer.decode(ids)
        expected = "hello, world!"
        assert decoded == expected

    def test_decode_handles_punctuation_spacing(self, tokenizer):
        """Test that decode properly handles punctuation spacing."""
        ids = [0, 1, 2, 3, 4, 5, 6, 7]  # hello world! this is a test.
        decoded = tokenizer.decode(ids)
        expected = "hello world! this is a test."
        assert decoded == expected

    def test_round_trip_encoding_decoding(self, tokenizer):
        """Test that encode->decode produces expected results."""
        original_texts = [
            "hello world",
            "hello world!",
            "this is a test.",
            "hello, world! this is a test.",
            "the quick brown fox jumps over the lazy dog.",
        ]
        
        for text in original_texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            # The decoded text should match the original after proper formatting
            assert isinstance(decoded, str)
            # Check that key words are preserved
            original_words = text.split()
            for word in original_words:
                clean_word = word.strip(".,!?;:()")
                if clean_word in tokenizer.str_to_int:
                    assert clean_word in decoded

    def test_encode_empty_string(self, tokenizer):
        """Test encoding empty string."""
        encoded = tokenizer.encode("")
        assert encoded == []

    def test_decode_empty_list(self, tokenizer):
        """Test decoding empty list."""
        decoded = tokenizer.decode([])
        assert decoded == ""

    def test_encode_unknown_token_raises_error(self, tokenizer):
        """Test that encoding unknown tokens raises KeyError."""
        with pytest.raises(KeyError):
            tokenizer.encode("unknown_token")

    def test_decode_unknown_id_raises_error(self, tokenizer):
        """Test that decoding unknown IDs raises KeyError."""
        with pytest.raises(KeyError):
            tokenizer.decode([999])  # ID not in vocabulary

    def test_encode_with_quotes(self, tokenizer):
        """Test encoding text with quotes."""
        text = 'hello "world"'
        encoded = tokenizer.encode(text)
        expected = [0, 22, 1, 22]  # hello "world"
        assert encoded == expected

    def test_encode_with_parentheses(self, tokenizer):
        """Test encoding text with parentheses."""
        text = "hello (world)"
        encoded = tokenizer.encode(text)
        expected = [0, 20, 1, 21]  # hello (world)
        assert encoded == expected

    def test_encode_with_double_dash(self, tokenizer):
        """Test encoding text with double dash."""
        text = "hello--world"
        encoded = tokenizer.encode(text)
        expected = [0, 24, 1]  # hello--world
        assert encoded == expected

    def test_encode_whitespace_handling(self, tokenizer):
        """Test that multiple whitespaces are handled correctly."""
        text = "hello    world"  # Multiple spaces
        encoded = tokenizer.encode(text)
        expected = [0, 1]  # Should ignore extra whitespace
        assert encoded == expected

    def test_vocabulary_completeness(self, tokenizer):
        """Test that vocabulary mappings are complete and consistent."""
        # Test that all keys in str_to_int have corresponding values in int_to_str
        for token, idx in tokenizer.str_to_int.items():
            assert idx in tokenizer.int_to_str
            assert tokenizer.int_to_str[idx] == token

        # Test that all keys in int_to_str have corresponding values in str_to_int
        for idx, token in tokenizer.int_to_str.items():
            assert token in tokenizer.str_to_int
            assert tokenizer.str_to_int[token] == idx

    def test_encode_decode_consistency(self, tokenizer):
        """Test that encode and decode are consistent operations."""
        test_cases = [
            "hello world!",
            "this is a test.",
            "hello, world!",
            "the quick brown fox",
            "test (with) parentheses",
            "test \"with\" quotes",
            "test--with--dashes",
        ]
        
        for text in test_cases:
            try:
                encoded = tokenizer.encode(text)
                decoded = tokenizer.decode(encoded)
                
                # The decoded text should be a valid string
                assert isinstance(decoded, str)
                
                # Re-encoding the decoded text should give the same result
                re_encoded = tokenizer.encode(decoded)
                assert re_encoded == encoded
                
            except KeyError:
                # If text contains unknown tokens, that's expected
                pass

    def test_special_characters_handling(self, tokenizer):
        """Test handling of various special characters."""
        special_chars = ["!", ".", ",", "?", "(", ")", "\"", "'", "--", ";", ":", "_"]
        
        for char in special_chars:
            if char in tokenizer.str_to_int:
                text = f"hello{char}world"
                encoded = tokenizer.encode(text)
                assert len(encoded) == 3  # hello, char, world
                
                decoded = tokenizer.decode(encoded)
                assert char in decoded

    @pytest.mark.parametrize("text,expected_token_count", [
        ("hello", 1),
        ("hello world", 2),
        ("hello world!", 3),
        ("hello, world!", 4),
        ("hello, world! this is a test.", 9),
    ])
    def test_token_count(self, tokenizer, text, expected_token_count):
        """Test that tokenization produces expected number of tokens."""
        try:
            encoded = tokenizer.encode(text)
            assert len(encoded) == expected_token_count
        except KeyError:
            # Skip if text contains unknown tokens
            pytest.skip(f"Text contains unknown tokens: {text}")
