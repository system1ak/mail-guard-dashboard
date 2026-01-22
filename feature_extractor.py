"""
TextFeatureExtractor - Convert text to 57 numeric features for spam detection
Extracted from Mail Guard app.py
"""

import numpy as np
import re
from collections import Counter


class TextFeatureExtractor:
    """
    Converts raw text to 57 numeric features matching Spambase format.
    Accounts for word frequencies, special characters, whitespace, alphanumeric patterns.
    """
    
    def __init__(self):
        """57 features in order:
        0-48: Top 49 most common word frequencies
        49: Average capital letter run length
        50: Longest capital letter run length
        51: Total capital letters count
        52: Frequency of ;
        53: Frequency of (
        54: Frequency of [
        55: Frequency of !
        56: Average word length
        """
        self.word_frequency_map = {}
        self.special_char_map = {';': 0, '(': 0, '[': 0, '!': 0}
        self.top_words = None
        self.text_length = 0
    
    def extract_words(self, text):
        """Extract words from text (lowercased, alphanumeric only)"""
        text_clean = re.sub(r'[^a-zA-Z0-9 ]', ' ', text.lower())
        words = text_clean.split()
        return [w for w in words if len(w) > 0]
    
    def calculate_capital_runs(self, text):
        """Calculate capital letter run statistics"""
        runs = re.findall(r'[A-Z]+', text)
        if not runs:
            return 0, 0
        avg_run_length = np.mean([len(r) for r in runs])
        max_run_length = max([len(r) for r in runs])
        return avg_run_length, max_run_length
    
    def count_capital_letters(self, text):
        """Count total capital letters"""
        return sum(1 for c in text if c.isupper())
    
    def count_special_chars(self, text):
        """Count frequency of special characters ; ( [ !"""
        char_counts = {';': 0, '(': 0, '[': 0, '!': 0}
        for char in char_counts:
            char_counts[char] = text.count(char)
        return char_counts
    
    def calculate_whitespace_stats(self, text):
        """Calculate whitespace and free space statistics"""
        total_chars = len(text)
        whitespace_chars = sum(1 for c in text if c.isspace())
        if total_chars == 0:
            return 0
        whitespace_percentage = (whitespace_chars / total_chars) * 100
        return whitespace_percentage
    
    def calculate_alphanumeric_stats(self, text):
        """Calculate alphanumeric character statistics"""
        alphanumeric_chars = sum(1 for c in text if c.isalnum())
        total_chars = len(text)
        if total_chars == 0:
            return 0
        alphanumeric_percentage = (alphanumeric_chars / total_chars) * 100
        return alphanumeric_percentage
    
    def fit(self, texts_list):
        """Learn top 49 most common words from training texts
        texts_list: list of raw text strings
        """
        all_words = []
        for text in texts_list:
            words = self.extract_words(text)
            all_words.extend(words)
        
        # Remove punctuation, convert to lowercase, split into words
        word_counter = Counter(all_words)
        self.top_words = [word for word, _ in word_counter.most_common(49)]
        return self
    
    def transform(self, text):
        """Convert single text to 57 numeric features
        Returns: np.array of shape (57,) with feature values
        """
        features = np.zeros(57)
        
        # Get top 49 most common words
        self.text_length = len(text)
        words = self.extract_words(text)
        word_count = len(words)
        
        # Text length and word extraction
        if word_count > 0:
            word_freq_in_text = Counter(words)
            for idx, word in enumerate(self.top_words[:49]):  # 1. Word Frequencies (0-48)
                if word in word_freq_in_text:
                    features[idx] = (word_freq_in_text[word] / word_count) * 100
        
        # 2. Capital Letter Statistics (49-51)
        avg_cap_run, max_cap_run = self.calculate_capital_runs(text)
        features[49] = avg_cap_run  # Average capital run length
        features[50] = max_cap_run  # Longest capital run length
        features[51] = self.count_capital_letters(text)  # Total capital count
        
        # 3. Special Character Frequencies (52-55)
        special_char_counts = self.count_special_chars(text)
        if self.text_length > 0:
            features[52] = (special_char_counts[';'] / self.text_length) * 100  # semicolon
            features[53] = (special_char_counts['('] / self.text_length) * 100  # parenthesis
            features[54] = (special_char_counts['['] / self.text_length) * 100  # bracket
            features[55] = (special_char_counts['!'] / self.text_length) * 100  # exclamation
        
        # 4. Average Word Length (56)
        if word_count > 0:
            features[56] = np.mean([len(w) for w in words])
        
        return features
    
    def fit_transform(self, texts_list):
        """Fit and transform in one step"""
        self.fit(texts_list)
        return np.array([self.transform(text) for text in texts_list])
