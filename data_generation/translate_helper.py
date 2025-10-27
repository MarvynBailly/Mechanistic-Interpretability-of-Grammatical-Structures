"""
Translation Helper for Data Generation Pipeline

This script helps manage translations from English to Chinese using ChatGPT.
As per the project proposal, translations should be:
1. Generated using ChatGPT
2. Manually verified by a native Chinese speaker

This helper provides:
- Batch translation functions (if OpenAI API is configured)
- Manual translation workflow
- Translation verification utilities
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path


class TranslationHelper:
    """Helper class for managing English-Chinese translations"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the translation helper

        Args:
            api_key: Optional OpenAI API key for automated translation
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.use_api = self.api_key is not None

        if self.use_api:
            try:
                import openai

                self.client = openai.OpenAI(api_key=self.api_key)
                print("✓ OpenAI API configured for automated translation")
            except ImportError:
                print(
                    "⚠ OpenAI package not installed. Install with: pip install openai"
                )
                self.use_api = False
        else:
            print("ℹ No API key found. Using manual translation mode.")

    def translate_with_chatgpt(self, text: str, context: str = "") -> str:
        """
        Translate text from English to Chinese using ChatGPT API

        Args:
            text: English text to translate
            context: Optional context about the text (e.g., "singular noun", "IOI sentence")

        Returns:
            Chinese translation
        """
        if not self.use_api:
            return f"[TRANSLATE: {text}]"

        try:
            context_note = f" (Context: {context})" if context else ""

            prompt = f"""Translate the following English text to Chinese.
This is for a linguistics research project studying grammatical structures.
Provide a natural, grammatically correct Chinese translation.{context_note}

English: {text}
Chinese:"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional English-Chinese translator specializing in linguistic research.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            translation = response.choices[0].message.content.strip()
            return translation

        except Exception as e:
            print(f"Error translating '{text}': {e}")
            return f"[ERROR: {text}]"

    def translate_word_list(self, words_file: str, output_file: str):
        """
        Translate a word list from JSON file

        Args:
            words_file: Path to words.json file
            output_file: Path to save translated words
        """
        print("=" * 60)
        print("Word List Translation")
        print("=" * 60)

        # Load words
        with open(words_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        words = data.get("words", [])
        translations = data.get("chinese_translations", {})

        print(f"Found {len(words)} words to translate")

        # Translate each word
        for idx, word_data in enumerate(words):
            word = word_data["word"]
            feature = word_data.get("grammatical_feature", "")

            if word in translations and not translations[word].startswith("[TRANSLATE"):
                print(
                    f"[{idx + 1}/{len(words)}] {word} -> {translations[word]} (already translated)"
                )
                continue

            if self.use_api:
                print(f"[{idx + 1}/{len(words)}] Translating: {word} ({feature})...")
                translation = self.translate_with_chatgpt(word, feature)
                translations[word] = translation
                print(f"  -> {translation}")
            else:
                translations[word] = f"[TRANSLATE: {word}]"
                print(f"[{idx + 1}/{len(words)}] {word} -> [NEEDS TRANSLATION]")

        # Update data
        data["chinese_translations"] = translations

        # Save
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nTranslations saved to: {output_file}")
        print("\n⚠ IMPORTANT: Have a native Chinese speaker verify all translations!")

    def translate_templates(self, templates_file: str, output_file: str):
        """
        Translate sentence templates from JSON file

        Args:
            templates_file: Path to templates.json file
            output_file: Path to save translated templates
        """
        print("=" * 60)
        print("Template Translation")
        print("=" * 60)

        # Load templates
        with open(templates_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        templates = data.get("templates", [])
        print(f"Found {len(templates)} templates to translate")

        # Translate each template
        for idx, template in enumerate(templates):
            english = template.get("english", "")
            chinese = template.get("chinese", "")

            # Skip if already has non-placeholder Chinese
            if chinese and not chinese.startswith("[TRANSLATE"):
                print(
                    f"[{idx + 1}/{len(templates)}] Template {template.get('id', idx)} (already translated)"
                )
                continue

            if self.use_api:
                print(
                    f"[{idx + 1}/{len(templates)}] Translating template {template.get('id', idx)}..."
                )
                print(f"  English: {english}")
                translation = self.translate_with_chatgpt(
                    english, "IOI sentence template"
                )
                template["chinese"] = translation
                print(f"  Chinese: {translation}")
            else:
                template["chinese"] = f"[TRANSLATE: {english}]"
                print(
                    f"[{idx + 1}/{len(templates)}] Template {template.get('id', idx)} -> [NEEDS TRANSLATION]"
                )

        # Save
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nTranslations saved to: {output_file}")
        print("\n⚠ IMPORTANT: Have a native Chinese speaker verify all translations!")

    def create_verification_checklist(
        self, translated_file: str, output_file: str = "verification_checklist.txt"
    ):
        """
        Create a checklist file for manual verification of translations

        Args:
            translated_file: Path to translated JSON file
            output_file: Path to save verification checklist
        """
        with open(translated_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("TRANSLATION VERIFICATION CHECKLIST\n")
            f.write("=" * 70 + "\n\n")
            f.write("Instructions for Native Chinese Speaker:\n")
            f.write("1. Review each translation below\n")
            f.write("2. Mark [ ] as [✓] if translation is correct\n")
            f.write(
                "3. Mark [ ] as [✗] and provide correction if translation is wrong\n"
            )
            f.write("4. Check for natural phrasing and grammatical correctness\n\n")
            f.write("=" * 70 + "\n\n")

            # Check if it's words or templates
            if "words" in data:
                f.write("WORD TRANSLATIONS\n\n")
                translations = data.get("chinese_translations", {})
                for word_data in data["words"]:
                    word = word_data["word"]
                    feature = word_data.get("grammatical_feature", "unknown")
                    chinese = translations.get(word, "[NO TRANSLATION]")

                    f.write(f"[ ] English: {word}\n")
                    f.write(f"    Feature: {feature}\n")
                    f.write(f"    Chinese: {chinese}\n")
                    f.write(f"    Correction (if needed): _________________\n\n")

            elif "templates" in data:
                f.write("TEMPLATE TRANSLATIONS\n\n")
                for template in data["templates"]:
                    template_id = template.get("id", "unknown")
                    english = template.get("english", "")
                    chinese = template.get("chinese", "[NO TRANSLATION]")

                    f.write(f"[ ] Template ID: {template_id}\n")
                    f.write(f"    English: {english}\n")
                    f.write(f"    Chinese: {chinese}\n")
                    f.write(f"    Correction (if needed):\n")
                    f.write(
                        f"    _________________________________________________\n\n"
                    )

        print(f"Verification checklist created: {output_file}")
        print("Please share this file with a native Chinese speaker for review.")


def create_manual_translation_template(
    words_file: str, output_file: str = "manual_translation_template.txt"
):
    """
    Create a simple template for manual translation without API

    Args:
        words_file: Path to words.json file
        output_file: Path to save manual translation template
    """
    with open(words_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("MANUAL TRANSLATION TEMPLATE\n")
        f.write("=" * 70 + "\n\n")
        f.write("Instructions:\n")
        f.write("1. Copy this file\n")
        f.write("2. Paste into ChatGPT or share with a translator\n")
        f.write("3. Fill in the Chinese translations\n")
        f.write("4. Copy results back to words.json\n\n")
        f.write("=" * 70 + "\n\n")

        if "words" in data:
            f.write("WORDS TO TRANSLATE:\n\n")
            for word_data in data["words"]:
                word = word_data["word"]
                feature = word_data.get("grammatical_feature", "")
                f.write(f"English: {word} ({feature})\n")
                f.write(f"Chinese: ____________\n\n")

        elif "templates" in data:
            f.write("TEMPLATES TO TRANSLATE:\n\n")
            for template in data["templates"]:
                english = template.get("english", "")
                f.write(f"English: {english}\n")
                f.write(f"Chinese: ____________\n\n")

    print(f"Manual translation template created: {output_file}")


def main():
    """Main function for translation workflow"""
    print("=" * 60)
    print("Translation Helper for Data Generation Pipeline")
    print("=" * 60)
    print()

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        print("OpenAI API key found. Automated translation available.")
        mode = input("Use automated translation? (y/n): ").lower().strip()
        use_auto = mode == "y"
    else:
        print("No OpenAI API key found.")
        print("To use automated translation, set OPENAI_API_KEY environment variable:")
        print("  export OPENAI_API_KEY='your-api-key-here'  # Linux/Mac")
        print("  set OPENAI_API_KEY=your-api-key-here      # Windows")
        use_auto = False

    print()

    # Initialize helper
    helper = TranslationHelper(api_key if use_auto else None)

    # Choose what to translate
    print("What would you like to translate?")
    print("1. Word list (words.json)")
    print("2. Templates (templates.json)")
    print("3. Both")
    print("4. Create manual translation template")
    print("5. Create verification checklist")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1" or choice == "3":
        if os.path.exists("words.json"):
            output = "words_translated.json" if use_auto else "words.json"
            helper.translate_word_list("words.json", output)
        else:
            print("Error: words.json not found!")

    if choice == "2" or choice == "3":
        if os.path.exists("templates.json"):
            output = "templates_translated.json" if use_auto else "templates.json"
            helper.translate_templates("templates.json", output)
        else:
            print("Error: templates.json not found!")

    if choice == "4":
        file_to_translate = input(
            "Enter file to create template for (words.json/templates.json): "
        ).strip()
        if os.path.exists(file_to_translate):
            create_manual_translation_template(file_to_translate)
        else:
            print(f"Error: {file_to_translate} not found!")

    if choice == "5":
        file_to_verify = input(
            "Enter translated file to verify (e.g., words_translated.json): "
        ).strip()
        if os.path.exists(file_to_verify):
            helper.create_verification_checklist(file_to_verify)
        else:
            print(f"Error: {file_to_verify} not found!")

    print("\n" + "=" * 60)
    print("Translation workflow complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review all Chinese translations")
    print("2. Have translations verified by a native Chinese speaker")
    print("3. Update the final JSON files with verified translations")
    print("4. Run generate_dataset.py to create the full dataset")


if __name__ == "__main__":
    main()
