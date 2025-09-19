import re
import unicodedata
from typing import List, Tuple, Optional
import hazm
import nltk
from .common import Gender
import pandas as pd
from emoji import is_emoji
from rapidfuzz.distance import Levenshtein, JaroWinkler, DamerauLevenshtein
import logging
from functools import lru_cache

# Constants
TOP_K = 3
LOG_ENABLED = True
NATIVE_LANG = 'fa'

class DataManager:
    """
    Handles loading and providing access to datasets.
    """
    def __init__(self, names_file: str, surnames_file: str) -> None:
        self.names_file = names_file
        self.surnames_file = surnames_file

        self.names_df = pd.read_csv(self.names_file)
        self.surnames_df = pd.read_csv(self.surnames_file)

    def get_names(self, column: str) -> pd.Series:
        """
        Get the names column from the names dataframe.

        :param column: Column name ('name' or 'english_name').
        :return: Pandas Series of names.
        """
        return self.names_df[column]

    @lru_cache(maxsize=1024)
    def get_gender(self, name: str, column: str) -> Optional[str]:
        """
        Get the gender associated with a name.

        :param name: The name to lookup.
        :param column: The column to search in ('name' or 'english_name').
        :return: Gender ('m', 'f', etc.) or None if not found.
        """
        gender_series = self.names_df.loc[self.names_df[column] == name, 'gender']
        if not gender_series.empty:
            return gender_series.values[0]
        return None

    def get_surnames(self, column: str) -> set:
        """
        Get the set of surnames from the surnames dataframe.

        :param column: Column name ('surname' or 'english_surname').
        :return: Set of surnames.
        """
        return set(self.surnames_df[column].str.lower())


class TextNormalizer:
    """
    Handles normalization of text.
    """

    def __init__(self, native_normalizer = None, lang: str = 'en') -> None:
        self.lang = lang
        self.normalizer = native_normalizer if lang == NATIVE_LANG else None
    @staticmethod
    def remove_emojis(text: str) -> str:
        """Remove emojis from text."""
        return ''.join(char for char in text if not is_emoji(char))
    @staticmethod
    def pre_normalize(text: str) -> str:
        """
        Pre-normalize text by removing emojis, normalizing unicode, and trimming spaces.

        :param text: The text to prenormalize.
        :return: Prenormalized text.
        """
        text = TextNormalizer.remove_emojis(text)
        text = unicodedata.normalize('NFKD', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def normalize(self, text: str) -> str:
        """
        Normalize text based on language.

        :param text: The text to normalize.
        :return: Normalized text.
        """
        if self.lang == NATIVE_LANG and self.normalizer:
            text = self.normalizer.normalize(text)
        elif self.lang == 'en':
            text = text.lower()
        return text


class Tokenizer:
    """
    Handles tokenization of text.
    """

    def __init__(self, native_tokenizer = None, lang: str = 'en') -> None:
        self.lang = lang
        if lang == NATIVE_LANG:
            self.tokenizer = native_tokenizer
        elif lang == 'en':
            self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    def split_display_name(self, display_name: str) -> List[str]:
        """
        Split display name into tokens.

        :param display_name: The display name to split.
        :return: List of tokens.
        """
        display_name = display_name.replace(".", " ").replace("_", " ").replace("@", "")
        tokens = self.tokenizer.tokenize(display_name)
        return [token for token in tokens if len(token) > 2]


class SurnameChecker:
    """
    Checks if a given name is a surname based on provided datasets.
    """

    def __init__(self, data_manager: DataManager) -> None:
        self.data_manager = data_manager
        self.top_persian_surnames = self.data_manager.get_surnames("surname")
        self.top_persian_surnames_en = self.data_manager.get_surnames("english_surname")

    @lru_cache(maxsize=1024)
    def is_surname(self, name: str, lang: str) -> bool:
        """
        Determine if a given name is among the top Persian surnames.

        :param name: The name to check.
        :param lang: Language of the name ('en' or NATIVE_LANG).
        :return: True if is surname, False otherwise.
        """
        if lang == 'en':
            if name in self.top_persian_surnames_en or any(
                    name.endswith(surname) for surname in self.top_persian_surnames_en):
                return True
        elif lang == NATIVE_LANG:
            if name in self.top_persian_surnames or any(
                    name.endswith(surname) for surname in self.top_persian_surnames):
                return True
        return False


class NameMatcher:
    """
    Handles matching names using various similarity metrics.
    """

    def __init__(self, data_manager: DataManager, top_k: int = TOP_K, debug: bool = LOG_ENABLED) -> None:
        self.data_manager = data_manager
        self.top_k = top_k
        self.debug = debug

    @lru_cache(maxsize=1024)
    def get_top_matches(
            self,
            name: str,
            lang: str,
            method: str = 'levenshtein'
    ) -> List[Tuple[str, float, str, int]]:
        """
        Get top N matches based on the specified similarity method.

        :param name: The name to match.
        :param lang: Language of the name ('en' or NATIVE_LANG).
        :param method: The similarity method ('levenshtein', 'damerau', 'jaro_winkler').
        :return: List of tuples containing matched name, score, gender, and index.
        """
        column = 'name' if lang == NATIVE_LANG else 'english_name'
        if method == 'levenshtein':
            return self._get_top_matches_levenshtein(name, column)
        elif method == 'damerau':
            return self._get_top_matches_damerau_levenshtein(name, column)
        elif method == 'jaro_winkler':
            return self._get_top_matches_jaro_winkler(name, column)
        else:
            raise ValueError(f"Unsupported method: {method}")

    def _get_top_matches_levenshtein(
            self,
            name: str,
            column: str
    ) -> List[Tuple[str, float, str, int]]:
        scores = self.data_manager.get_names(column).apply(
            lambda x: Levenshtein.normalized_similarity(name, x) * 100
        )
        top_indices = scores.nlargest(self.top_k).index
        matches = [
            (
                self.data_manager.names_df.at[idx, column],
                scores.at[idx],
                self.data_manager.names_df.at[idx, 'gender'],
                idx
            )
            for idx in top_indices
        ]
        if self.debug:
            logging.info(f"[Levenshtein] Name: {name}, Matches: {matches}")
        return matches

    def _get_top_matches_damerau_levenshtein(
            self,
            name: str,
            column: str
    ) -> List[Tuple[str, float, str, int]]:
        scores = self.data_manager.get_names(column).apply(
            lambda x: DamerauLevenshtein.normalized_similarity(name, x) * 100
        )
        top_indices = scores.nlargest(self.top_k).index
        matches = [
            (
                self.data_manager.names_df.at[idx, column],
                scores.at[idx],
                self.data_manager.names_df.at[idx, 'gender'],
                idx
            )
            for idx in top_indices
        ]
        if self.debug:
            logging.info(f"[Damerau-Levenshtein] Name: {name}, Matches: {matches}")
        return matches

    def _get_top_matches_jaro_winkler(
            self,
            name: str,
            column: str
    ) -> List[Tuple[str, float, str, int]]:
        scores = self.data_manager.get_names(column).apply(
            lambda x: JaroWinkler.similarity(name, x) * 100
        )
        top_indices = scores.nlargest(self.top_k).index
        matches = [
            (
                self.data_manager.names_df.at[idx, column],
                scores.at[idx],
                self.data_manager.names_df.at[idx, 'gender'],
                idx
            )
            for idx in top_indices
        ]
        if self.debug:
            logging.info(f"[Jaro-Winkler] Name: {name}, Matches: {matches}")
        return matches


class GenderDetector:
    """
    Orchestrates the process to guess gender based on display names.
    Automatically detects language and uses suitable normalizer and tokenizer.
    """

    def __init__(
            self,
            data_manager: DataManager,
            matcher: NameMatcher,
            surname_checker: SurnameChecker,
            top_k: int = TOP_K,
            debug: bool = LOG_ENABLED
    ) -> None:
        self.data_manager = data_manager
        self.matcher = matcher
        self.surname_checker = surname_checker
        self.top_k = top_k
        self.debug = debug

        # Initialize normalizers and tokenizers for both languages
        self.normalizer_en = TextNormalizer(lang='en')
        self.normalizer_fa = TextNormalizer(native_normalizer = hazm.Normalizer(), lang=NATIVE_LANG)
        self.tokenizer_en = Tokenizer(lang='en')
        self.tokenizer_fa = Tokenizer(native_tokenizer = hazm.WordTokenizer(), lang=NATIVE_LANG)

    @staticmethod
    def is_ascii(s: str) -> bool:
        """Check if a string is ASCII (English)."""
        return all(ord(c) < 128 for c in s)

    @staticmethod
    def get_guessed_gender(matches: List[Tuple[str, float, str, int]]) -> float:
        """
        Guess gender based on the top matches.

        :param matches: List of tuples (matched_name, score, gender, index).
        :return: Weighted sum of gender probabilities.
        """
        if not matches:
            return 0.0  # Neutral if no matches

        total = 0.0
        for idx, (_, score, gender, _) in enumerate(matches):
            if idx == 0 and score == 100:
                if gender.lower() == Gender.MALE.value:
                    return 100
                elif gender.lower() == Gender.FEMALE.value:
                    return -100

            if gender.lower() == Gender.MALE.value:
                total += score
            elif gender.lower() == Gender.FEMALE.value:
                total -= score
        return total / len(matches)

    def guess_gender(self, display_name: str) -> Tuple[str, Optional[float]]:
        """
        Guess the gender based on the display name.

        :param display_name: The display name to analyze.
        :return: Tuple of guessed gender ('m', 'f', 'u') and probability.
        """
        pre_normalized_display_name = TextNormalizer.pre_normalize(display_name)
        is_english = GenderDetector.is_ascii(pre_normalized_display_name)

        # Select appropriate normalizer and tokenizer based on language
        if is_english:
            normalizer = self.normalizer_en
            tokenizer = self.tokenizer_en
            lang = 'en'
        else:
            normalizer = self.normalizer_fa
            tokenizer = self.tokenizer_fa
            lang = NATIVE_LANG

        
        # Normalize the display name
        normalized_name = normalizer.normalize(pre_normalized_display_name)

        # Tokenize the display name
        tokens = tokenizer.split_display_name(normalized_name)

        if self.debug:
            logging.info(f"Normalized Name: {normalized_name}, Tokens: {tokens}")

        if not tokens:
            return Gender.UNKNOWN.value, None

        # Find the first non-surname token to search
        to_search: Optional[str] = None
        for word in tokens:
            if not self.surname_checker.is_surname(word, lang=lang):
                to_search = word.lower()
                break
            elif self.debug:
                logging.info(f"'{word}' is identified as a surname.")
        
        if to_search is None:
            return Gender.UNKNOWN.value, None

        # Get top matches using Levenshtein similarity
        matches = self.matcher.get_top_matches(to_search, lang, method='levenshtein')

        # Calculate gender probability
        prob = self.get_guessed_gender(matches) / 100

        if self.debug:
            logging.info(f"Gender Probability for '{to_search}': {prob}")

        
        # Assign gender based on probability
        if prob > 0:
            return Gender.MALE.value, prob
        elif prob < 0:
            return Gender.FEMALE.value, -prob
        else:
            return Gender.UNKNOWN.value, prob