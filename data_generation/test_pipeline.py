"""
Test Script for Data Generation Pipeline

This script validates that the data generation pipeline works correctly
and produces the expected outputs as per the project proposal.

Tests include:
- Template loading and validation
- Word list loading and validation
- Clean pair generation (should produce 900 pairs)
- Fuzzy variant generation
- Path patching pair generation
- Output file creation and format validation
"""

import json
import os
import sys
from pathlib import Path
import tempfile
import shutil


class PipelineTestRunner:
    """Test runner for the data generation pipeline"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.temp_dir = None

    def setup(self):
        """Set up test environment"""
        print("=" * 60)
        print("Data Generation Pipeline Test Suite")
        print("=" * 60)
        print()

        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp(prefix="pipeline_test_")
        print(f"✓ Test directory created: {self.temp_dir}")

    def teardown(self):
        """Clean up test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"\n✓ Test directory cleaned up")

    def assert_true(self, condition, test_name, message=""):
        """Assert that condition is true"""
        if condition:
            print(f"✓ PASS: {test_name}")
            self.passed += 1
        else:
            print(f"✗ FAIL: {test_name}")
            if message:
                print(f"  Reason: {message}")
            self.failed += 1

    def assert_equal(self, actual, expected, test_name):
        """Assert that actual equals expected"""
        if actual == expected:
            print(f"✓ PASS: {test_name}")
            self.passed += 1
        else:
            print(f"✗ FAIL: {test_name}")
            print(f"  Expected: {expected}")
            print(f"  Actual: {actual}")
            self.failed += 1

    def test_templates_file_exists(self):
        """Test that templates.json exists and is valid"""
        print("\n--- Testing Templates File ---")

        # Check file exists
        self.assert_true(
            os.path.exists("templates.json"),
            "templates.json exists",
            "File not found in current directory",
        )

        if not os.path.exists("templates.json"):
            return

        # Check file is valid JSON
        try:
            with open("templates.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assert_true(True, "templates.json is valid JSON")
        except Exception as e:
            self.assert_true(False, "templates.json is valid JSON", str(e))
            return

        # Check structure
        self.assert_true("templates" in data, "templates.json contains 'templates' key")

        if "templates" not in data:
            return

        templates = data["templates"]

        # Check count (should be 15 as per proposal)
        self.assert_equal(len(templates), 15, "templates.json contains 15 templates")

        # Check each template has required fields
        all_valid = True
        for idx, template in enumerate(templates):
            if not all(key in template for key in ["english", "chinese"]):
                all_valid = False
                print(f"  Template {idx} missing required fields")
                break

        self.assert_true(
            all_valid, "All templates have required fields (english, chinese)"
        )

        # Check for {word} placeholder
        all_have_placeholder = all("{word}" in t.get("english", "") for t in templates)
        self.assert_true(
            all_have_placeholder, "All templates contain {word} placeholder"
        )

    def test_words_file_exists(self):
        """Test that words.json exists and is valid"""
        print("\n--- Testing Words File ---")

        # Check file exists
        self.assert_true(
            os.path.exists("words.json"),
            "words.json exists",
            "File not found in current directory",
        )

        if not os.path.exists("words.json"):
            return

        # Check file is valid JSON
        try:
            with open("words.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assert_true(True, "words.json is valid JSON")
        except Exception as e:
            self.assert_true(False, "words.json is valid JSON", str(e))
            return

        # Check structure
        self.assert_true("words" in data, "words.json contains 'words' key")
        self.assert_true(
            "chinese_translations" in data,
            "words.json contains 'chinese_translations' key",
        )

        if "words" not in data:
            return

        words = data["words"]

        # Check count (should be 60 as per proposal)
        self.assert_equal(len(words), 60, "words.json contains 60 words")

        # Check each word has required fields
        all_valid = True
        for idx, word_data in enumerate(words):
            if "word" not in word_data:
                all_valid = False
                print(f"  Word entry {idx} missing 'word' field")
                break

        self.assert_true(all_valid, "All word entries have 'word' field")

        # Check translations exist
        if "chinese_translations" in data:
            translations = data["chinese_translations"]
            has_translations = len(translations) > 0
            self.assert_true(
                has_translations,
                "chinese_translations dictionary is not empty",
            )

    def test_generate_dataset_import(self):
        """Test that generate_dataset.py can be imported"""
        print("\n--- Testing Module Import ---")

        try:
            sys.path.insert(0, os.getcwd())
            import generate_dataset

            self.assert_true(True, "generate_dataset.py can be imported")

            # Check main class exists
            self.assert_true(
                hasattr(generate_dataset, "DatasetGenerator"),
                "DatasetGenerator class exists",
            )

            return generate_dataset
        except Exception as e:
            self.assert_true(False, "generate_dataset.py can be imported", str(e))
            return None

    def test_dataset_generation(self, generate_dataset_module):
        """Test full dataset generation"""
        if not generate_dataset_module:
            print("\n--- Skipping Dataset Generation Test (import failed) ---")
            return

        print("\n--- Testing Dataset Generation ---")

        try:
            generator = generate_dataset_module.DatasetGenerator(
                output_dir=self.temp_dir
            )
            self.assert_true(True, "DatasetGenerator instantiated successfully")
        except Exception as e:
            self.assert_true(
                False, "DatasetGenerator instantiated successfully", str(e)
            )
            return

        # Test loading templates
        try:
            generator.load_templates("templates.json")
            self.assert_true(True, "Templates loaded successfully")
            self.assert_equal(len(generator.templates), 15, "Loaded 15 templates")
        except Exception as e:
            self.assert_true(False, "Templates loaded successfully", str(e))
            return

        # Test loading words
        try:
            generator.load_words("words.json")
            self.assert_true(True, "Words loaded successfully")
            self.assert_equal(len(generator.words), 60, "Loaded 60 words")
        except Exception as e:
            self.assert_true(False, "Words loaded successfully", str(e))
            return

        # Test clean pair generation
        try:
            clean_pairs = generator.generate_clean_pairs()
            self.assert_true(True, "Clean pairs generated successfully")

            # Should be 60 words × 15 templates = 900 pairs
            expected_pairs = 60 * 15
            self.assert_equal(
                len(clean_pairs),
                expected_pairs,
                f"Generated {expected_pairs} clean pairs (60 words × 15 templates)",
            )

            # Check structure of first pair
            if len(clean_pairs) > 0:
                first_pair = clean_pairs[0]
                has_fields = all(
                    hasattr(first_pair, field)
                    for field in ["english", "chinese", "word", "is_fuzzy"]
                )
                self.assert_true(has_fields, "Clean pairs have required fields")
        except Exception as e:
            self.assert_true(False, "Clean pairs generated successfully", str(e))
            return

        # Test fuzzy pair generation
        try:
            fuzzy_pairs = generator.generate_fuzzy_pairs(clean_pairs)
            self.assert_true(True, "Fuzzy pairs generated successfully")

            # Should be 900 × 3 fuzzy types = 2700 pairs
            expected_fuzzy = 900 * 3
            self.assert_equal(
                len(fuzzy_pairs),
                expected_fuzzy,
                f"Generated {expected_fuzzy} fuzzy pairs (900 × 3 fuzzy types)",
            )
        except Exception as e:
            self.assert_true(False, "Fuzzy pairs generated successfully", str(e))

        # Test path patching pair generation
        try:
            patching_pairs = generator.generate_path_patching_pairs(clean_pairs)
            self.assert_true(True, "Path patching pairs generated successfully")

            # Should be 900 pairs
            self.assert_equal(
                len(patching_pairs),
                900,
                "Generated 900 path patching pairs",
            )

            # Check structure
            if len(patching_pairs) > 0:
                first_pair = patching_pairs[0]
                has_structure = all(
                    key in first_pair for key in ["clean", "corrupted", "metadata"]
                )
                self.assert_true(
                    has_structure, "Path patching pairs have correct structure"
                )
        except Exception as e:
            self.assert_true(
                False, "Path patching pairs generated successfully", str(e)
            )

        # Test saving
        try:
            generator.save_dataset(clean_pairs, fuzzy_pairs, patching_pairs)
            self.assert_true(True, "Dataset saved successfully")

            # Check output files exist
            output_files = [
                "clean_pairs.json",
                "fuzzy_pairs.json",
                "path_patching_pairs.json",
                "dataset_stats.json",
            ]

            for filename in output_files:
                filepath = os.path.join(self.temp_dir, filename)
                self.assert_true(os.path.exists(filepath), f"{filename} created")

        except Exception as e:
            self.assert_true(False, "Dataset saved successfully", str(e))

    def test_output_format(self):
        """Test that output files have correct format"""
        print("\n--- Testing Output Format ---")

        output_dir = self.temp_dir
        clean_file = os.path.join(output_dir, "clean_pairs.json")

        if not os.path.exists(clean_file):
            print("  Skipping (no output file to test)")
            return

        try:
            with open(clean_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assert_true(isinstance(data, list), "Clean pairs is a list")

            if len(data) > 0:
                first_item = data[0]
                required_fields = [
                    "template_id",
                    "word_id",
                    "english",
                    "chinese",
                    "word",
                    "grammatical_feature",
                    "is_fuzzy",
                ]
                has_all_fields = all(field in first_item for field in required_fields)
                self.assert_true(
                    has_all_fields,
                    "Clean pair entries have all required fields",
                )

                # Check is_fuzzy is False for clean pairs
                self.assert_equal(
                    first_item["is_fuzzy"],
                    False,
                    "Clean pairs have is_fuzzy=False",
                )

        except Exception as e:
            self.assert_true(False, "Output format is valid", str(e))

    def test_computational_requirements(self):
        """Test that dataset meets computational requirements from proposal"""
        print("\n--- Testing Computational Requirements ---")

        # As per proposal: 900 pairs × 2 runs × 12 layers × 12 heads = 270,000 forward passes
        num_pairs = 900
        runs_per_pair = 2
        num_layers = 12
        num_heads = 12

        expected_forward_passes = num_pairs * runs_per_pair * num_layers * num_heads

        self.assert_equal(
            expected_forward_passes,
            259200,  # Note: 900*2*12*12 = 259,200, not 270,000
            "Calculated forward passes (900×2×12×12)",
        )

        print(f"  Note: Expected ~259,200 forward passes for path patching")
        print(f"  (900 pairs × 2 runs × 12 layers × 12 heads)")

    def run_all_tests(self):
        """Run all tests"""
        self.setup()

        try:
            # File existence tests
            self.test_templates_file_exists()
            self.test_words_file_exists()

            # Module import test
            generate_dataset = self.test_generate_dataset_import()

            # Dataset generation tests
            self.test_dataset_generation(generate_dataset)
            self.test_output_format()

            # Requirements test
            self.test_computational_requirements()

        finally:
            self.teardown()

        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total:  {self.passed + self.failed}")

        if self.failed == 0:
            print("\n✓ All tests passed!")
            return 0
        else:
            print(f"\n✗ {self.failed} test(s) failed")
            return 1


def main():
    """Main test execution"""
    runner = PipelineTestRunner()
    exit_code = runner.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
