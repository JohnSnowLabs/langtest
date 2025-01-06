from typing import Callable
import functools
from ..errors import Errors


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
                raise ValueError(Errors.E035(type_a=type(str1), type_b=str))
            return func(str1, str2)

        return wrapper

    def __getitem__(self, name):
        if name in self.available_string_distance:
            return self.available_string_distance[name]
        else:
            raise KeyError(Errors.E036(name=name))

    @staticmethod
    @validate_input
    def _normalized_jaro_distance(str1: str, str2: str) -> float:
        """
        Calculate the normalized Jaro distance between two input strings.

        Jaro distance is a measure of string similarity, with values between 0.0 (perfect match)
        and 1.0 (no similarity). It quantifies the similarity between two strings based on
        the number of matching characters and the number of transpositions required to match them.

        :param str1: The first input string.
        :param str2: The second input string.

        :return: The normalized Jaro distance between the two input strings, with values between 0.0 and 1.0.
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

    @staticmethod
    @validate_input
    def _normalized_levenshtein_distance(str1: str, str2: str) -> float:
        """
        Calculate the normalized Levenshtein distance between two strings.

        The Levenshtein distance measures the minimum number of single-character edits required to
        change one string into the other. The normalized Levenshtein distance is the ratio of the
        edit distance to the length of the longer string.

        :param str1: The first input string.
        :param str2: The second input string.

        :return: The normalized Levenshtein distance between the two strings. This is a value between 0.0 (identical)
                 and 1.0 (completely different), normalized by the length of the longer string.
        """
        len1 = len(str1)
        len2 = len(str2)

        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if str1[i - 1] == str2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost
                )

        normalized_distance = dp[len1][len2] / max(len1, len2)

        return normalized_distance

    @staticmethod
    @validate_input
    def _normalized_damerau_levenshtein_distance(str1: str, str2: str) -> float:
        """
        Calculate the normalized Damerau-Levenshtein distance between two strings.

        The Damerau-Levenshtein distance is an extension of the Levenshtein distance that also allows
        for transpositions of adjacent characters. The normalized Damerau-Levenshtein distance is the
        ratio of the edit distance to the length of the longer string.

        :param str1: The first input string.
        :param str2: The second input string.

        :return: The normalized Damerau-Levenshtein distance between the two strings. This is a value between 0.0 (identical)
                 and 1.0 (completely different), normalized by the length of the longer string.
        """
        len1 = len(str1)
        len2 = len(str2)

        # Create a matrix to store the edit distances
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        # Initialize the first row and column of the matrix
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        # Fill in the matrix with edit distances
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if str1[i - 1] == str2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost
                )

                # Check for transpositions (if possible)
                if (
                    i > 1
                    and j > 1
                    and str1[i - 1] == str2[j - 2]
                    and str1[i - 2] == str2[j - 1]
                ):
                    dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + cost)

        # Calculate the normalized Damerau-Levenshtein distance
        normalized_distance = dp[len1][len2] / max(len1, len2)

        return normalized_distance

    @staticmethod
    @validate_input
    def _normalized_indel_distance(str1: str, str2: str) -> float:
        """
        Calculate the normalized Indel distance between two strings.

        The Indel distance measures the number of insertions and deletions required to make two strings
        identical. The normalized Indel distance is the ratio of the total edit operations to the maximum
        length of the input strings.

        :param str1: The first input string.
        :param str2: The second input string.

        :return: The normalized Indel distance between the two strings. This is a value between 0.0 (identical)
                 and 1.0 (completely different), normalized by the maximum length of the input strings.
        """
        # Initialize a matrix with dimensions (len(str1) + 1) x (len(str2) + 1)
        matrix = [[0 for _ in range(len(str2) + 1)] for _ in range(len(str1) + 1)]

        # Initialize the first row and first column of the matrix
        for i in range(len(str1) + 1):
            matrix[i][0] = i
        for j in range(len(str2) + 1):
            matrix[0][j] = j

        # Fill in the matrix using dynamic programming
        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                cost = 0 if str1[i - 1] == str2[j - 1] else 2
                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,  # Deletion
                    matrix[i][j - 1] + 1,  # Insertion
                    matrix[i - 1][j - 1] + cost,  # Substitution
                )

        # The minimum number of insertions and deletions is in the bottom-right cell of the matrix
        indel_distance = matrix[len(str1)][len(str2)]

        len1, len2 = len(str1), len(str2)
        indel_similarity = (len1 + len2) - indel_distance
        max_similarity = len(str1) + len(str2)
        normalized_similarity = 1 - (indel_similarity / max_similarity)

        return normalized_similarity

    @classmethod
    def available_string_distance(cls, distance: str = "jaro"):
        """Get the specified distance metric for string similarity calculations.

        Args:
            distance (str, optional): The desired distance metric. Defaults to "jaro".

        Returns:
            callable: The corresponding distance calculation method.

        Raises:
            ValueError: If the specified distance metric is not supported.
        """
        distance_mapping = {
            "jaro": cls._normalized_jaro_distance,
            "jaro_winkler": cls._normalized_jaro_winkler_distance,
            "hamming": cls._normalized_hamming_distance,
            "levenshtein": cls._normalized_levenshtein_distance,
            "damerau_levenshtein": cls._normalized_damerau_levenshtein_distance,
            "indel": cls._normalized_indel_distance,
        }

        if distance not in distance_mapping:
            raise ValueError(Errors.E076(metric="string", selected_metric=distance))
        return distance_mapping[distance]
