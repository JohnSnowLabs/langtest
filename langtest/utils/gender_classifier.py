from ..transform.constants import male_names, female_names


class GenderClassifier:
    """Helper model to predict the 'gender' of a piece of text."""

    def predict(self, sentence):
        # Convert the sentence to lowercase for case-insensitive matching
        sentence = sentence.lower().strip().split()

        # Define lists of keywords for each category
        female_keywords = [
            "mrs.",
            "miss",
            "ms.",
            "she",
            "her",
            "hers",
            "woman",
            "female",
            "girl",
            "sister",
            "actress",
            "waitress",
            "stewardess",
            "princess",
            "queen",
            "duchess",
            "niece",
            "mother",
            "daughter",
            "sister",
            "madam",
            "wife",
            "bride",
            "lady",
            "female",
        ]
        female_keywords.extend(female_names)

        male_keywords = [
            "mr.",
            "he",
            "him",
            "his",
            "man",
            "male",
            "boy",
            "brother",
            "actor",
            "waiter",
            "steward",
            "prince",
            "king",
            "duke",
            "nephew",
            "father",
            "son",
            "husband",
            "groom",
            "gentleman",
            "male",
        ]
        male_keywords.extend(male_names)

        # Count the number of female and male keywords in the sentence
        female_count = sum(1 for keyword in female_keywords if keyword in sentence)
        male_count = sum(1 for keyword in male_keywords if keyword in sentence)

        # Determine the predominant gender mentioned in the sentence
        if female_count > male_count:
            return "female"
        elif male_count > female_count:
            return "male"
        else:
            return "unknown"
