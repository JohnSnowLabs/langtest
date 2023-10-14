from typing import Callable
import functools


class StringDistance:

    """
    A class for calculating various string distance metrics.
    """

    """
    A class for calculating various string distance metrics.
    """

    def validate_input(func: Callable[[str, str], float]):
        """
        A decorator function for validating input strings.

        :param func: The function to be decorated.
        :return: The decorated function.
        """

        @functools.wraps(func)
        def wrapper(str1: str, str2: str) -> float:
            if not isinstance(str1, str) or not isinstance(str2, str):
                raise ValueError(
                    f"Input strings must be of type 'str', but received types: {type(str1)} and {type(str2)}"
                )
            return func(str1, str2)

        return wrapper

    def __getitem__(self, name):
        if name in self.available_string_distance:
            return self.available_string_distance[name]
        else:
            raise KeyError(f"String distance function '{name}' not found")

    @staticmethod
    @validate_input
    def _normalized_jaro_distance(str1: str, str2: str) -> float:
        """
        Calculate the normalized Jaro distance between two strings.

        The Jaro distance is a measure of similarity between two strings. It counts the number
        of matching characters in the two strings and measures the proximity of common characters.
        A Jaro distance of 1.0 indicates a perfect match, while 0.0 indicates no similarity.

        :param str1: The first input string.
        :param str2: The second input string.

        :return: The normalized Jaro distance between the two strings. This is a value between 0.0 (perfect match)
                and 1.0 (no similarity).
        """
        # Check if the strings are identical, return 0.0 for a perfect match
        if str1 == str2:
            return 0.0

        # Calculate the lengths of the input strings
        len_str1 = len(str1)
        len_str2 = len(str2)

        # Calculate the maximum allowed character distance for a match
        max_dist = max(len_str1, len_str2) // 2 - 1

        # Initialize variables to keep track of matches and matched characters
        match_count = 0
        str1_matches = [0] * len_str1
        str2_matches = [0] * len_str2

        # Loop through the characters in str1 and str2 to find matches
        for i in range(len_str1):
            for j in range(max(0, i - max_dist), min(len_str2, i + max_dist + 1)):
                if str1[i] == str2[j] and str2_matches[j] == 0:
                    str1_matches[i] = 1
                    str2_matches[j] = 1
                    match_count += 1
                    break

        # If there are no matches, return a Jaro distance of 1.0
        if match_count == 0:
            return 1.0

        # Calculate the number of transpositions
        transpositions = 0
        str2_index = 0
        for i in range(len_str1):
            if str1_matches[i]:
                while str2_matches[str2_index] == 0:
                    str2_index += 1
                if str1[i] != str2[str2_index]:
                    transpositions += 1
                str2_index += 1
        transpositions = transpositions // 2

        # Calculate and return the Jaro distance
        jaro_similarity = (
            match_count / len_str1
            + match_count / len_str2
            + (match_count - transpositions) / match_count
        ) / 3.0
        return 1 - jaro_similarity

    @staticmethod
    @validate_input
    def _normalized_jaro_winkler_distance(str1: str, str2: str) -> float:
        """
        Calculate the normalized Jaro-Winkler distance between two strings.

        The Jaro-Winkler distance is a variant of the Jaro distance that considers the common prefix
        of two strings and gives it extra weight. It is used to measure similarity with an emphasis
        on initial characters.

        :param str1: The first input string.
        :param str2: The second input string.

        :return: The normalized Jaro-Winkler distance between the two strings. This is a value between 0.0 (perfect match)
                 and 1.0 (no similarity), with extra weight given to the common prefix.
        """

        jaro_dist = 1 - StringDistance._normalized_jaro_distance(str1, str2)

        # If the jaro Similarity is above a threshold
        if jaro_dist > 0.7:

            # Find the length of common prefix
            prefix = 0

            for i in range(min(len(str1), len(str2))):

                # If the characters match
                if str1[i] == str2[i]:
                    prefix += 1

                # Else break
                else:
                    break

            # Maximum of 4 characters are allowed in prefix
            prefix = min(4, prefix)

            # Calculate jaro winkler Similarity
            jaro_dist += 0.1 * prefix * (1 - jaro_dist)

        return 1 - jaro_dist

    @staticmethod
    @validate_input
    def _normalized_hamming_distance(str1, str2):
        len1 = len(str1)
        len2 = len(str2)

        min_len = min(len1, len2)
        hamming_distance = sum(c1 != c2 for c1, c2 in zip(str1[:min_len], str2[:min_len]))
        hamming_distance += abs(len1 - len2)
        normalized_distance = hamming_distance / max(len1, len2)

        return normalized_distance
