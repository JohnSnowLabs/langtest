from numbers import Number
from pydantic.v1 import Field
from pydantic.v1.typing import Annotated
from ..errors import Errors
from langtest.transform.constants import (
    ordinal,
    unit,
    teen,
    ten,
    mill,
    nth_suff,
    ordinal_suff,
)
from langtest.transform.constants import (
    NON_DIGIT,
    WHITESPACES,
    COMMA_WORD,
    WHITESPACES_COMMA,
    DIGIT_GROUP,
    TWO_DIGITS,
    THREE_DIGITS,
    THREE_DIGITS_WORD,
    TWO_DIGITS_WORD,
    ONE_DIGIT_WORD,
    FOUR_DIGIT_COMMA,
)
from typing import (
    Union,
    Optional,
    List,
    Match,
    Any,
)

Word = Annotated[str, Field(min_length=1)]
Falsish = Any

STDOUT_ON = False  # Flag indicating whether standard output is enabled or disabled


def print3(txt: str) -> None:
    """Prints the provided text.

    Args:
        txt (str): The text to be printed.

    Returns:
        None
    """
    if STDOUT_ON:
        print(txt)


class BadChunkingOptionError(Exception):
    """Exception raised when an invalid chunking option is encountered."""

    pass


class NumOutOfRangeError(Exception):
    """Exception raised when a number is out of the supported range."""

    pass


class ConvertNumberToWord:
    """Converts a number to its corresponding word representation."""

    def __init__(self) -> None:
        """Constructor method"""
        self.mill_count = 0

    def millfn(self, ind: int = 0) -> str:
        """Retrieves the word representation of a given mill index.

        Args:
            ind (int): The index of the mill.

        Returns:
            str: The word representation of the mill.

        Raises:
            NumOutOfRangeError: If the index is out of range.
        """
        if ind >= len(mill):
            raise NumOutOfRangeError(Errors.E072)
        return mill[ind]

    def tenfn(self, tens: int, units: int, mindex: int = 0) -> str:
        """Converts a two-digit number to its word representation.

        Args:
            tens (int): The tens digit.
            units (int): The units digit.
            mindex (int): The index of the mill.

        Returns:
            str: The word representation of the number.
        """
        if tens != 1:
            tens_part = ten[tens]
            hyphen = "-" if tens and units else ""
            unit_part = unit[units]
            mill_part = self.millfn(mindex)
            return f"{tens_part}{hyphen}{unit_part}{mill_part}"
        return f"{teen[units]}{mill[mindex]}"

    def group1sub(self, mo: Match) -> str:
        """Substitutes a matched pattern from group 1 with its word representation.

        Args:
            mo (Match): The matched pattern object.

        Returns:
            str: The word representation of the matched pattern.
        """
        units = int(mo.group(1))
        if units == 1:
            return f" {self.number_args['one']}, "
        elif units:
            return f"{unit[units]}, "
        else:
            return f" {self.number_args['zero']}, "

    def group1bsub(self, mo: Match) -> str:
        """Substitutes a matched pattern from group 1b with its word representation.

        Args:
            mo (Match): The matched pattern object.

        Returns:
            str: The word representation of the matched pattern.
        """
        units = int(mo.group(1))
        if units:
            return f"{unit[units]}, "
        else:
            return f" {self.number_args['zero']}, "

    def group2sub(self, mo: Match) -> str:
        """Substitutes a matched pattern from group 2 with its word representation.

        Args:
            mo (Match): The matched pattern object.

        Returns:
            str: The word representation of the matched pattern.
        """
        tens = int(mo.group(1))
        units = int(mo.group(2))
        if tens:
            return f"{self.tenfn(tens, units)}, "
        if units:
            return f" {self.number_args['zero']} {unit[units]}, "
        return f" {self.number_args['zero']} {self.number_args['zero']}, "

    def group3sub(self, mo: Match) -> str:
        """Substitutes a matched pattern from group 3 with its word representation.

        Args:
            mo (Match): The matched pattern object.

        Returns:
            str: The word representation of the matched pattern.
        """
        hundreds = int(mo.group(1))
        tens = int(mo.group(2))
        units = int(mo.group(3))
        number_args = self.number_args
        hunword = (
            f" {number_args['one']}"
            if hundreds == 1
            else str(unit[hundreds]) if hundreds else f" {number_args['zero']}"
        )
        tenword = (
            self.tenfn(tens, units)
            if tens
            else (
                f" {number_args['zero']} {unit[units]}"
                if units
                else f" {number_args['zero']} {number_args['zero']}"
            )
        )
        return f"{hunword} {tenword}, "

    def number_to_words(
        self,
        num: Union[Number, Word],
        wantlist: bool = False,
        group: int = 0,
        comma: Union[Falsish, str] = ",",
        andword: str = "and",
        zero: str = "zero",
        one: str = "one",
        decimal: Union[Falsish, str] = "point",
        threshold: Optional[int] = None,
    ) -> Union[str, List[str]]:
        """Converts a number or word to its corresponding word representation.

        Args:
            num (Union[Number, Word]): The number or word to be converted.
            wantlist (bool, optional): Flag to indicate whether to return the result as a list of words. Defaults to False.
            group (int, optional): The group number indicating the pattern to be matched. Defaults to 0.
            comma (Union[Falsish, str], optional): The string to be used as a comma separator. Defaults to ",".
            andword (str, optional): The word to be used as a connector between numbers. Defaults to "and".
            zero (str, optional): The word representation of zero. Defaults to "zero".
            one (str, optional): The word representation of one. Defaults to "one".
            decimal (Union[Falsish, str], optional): The string to be used as a decimal point. Defaults to "point".
            threshold (Optional[int], optional): The threshold value for stylistic conversions. Defaults to None.

        Returns:
            Union[str, List[str]]: The word representation of the given number or word.

        Raises:
            BadChunkingOptionError: If the given group is invalid.
        """
        self.number_args = {"andword": andword, "zero": zero, "one": one}
        num = str(num)

        # Handle "stylistic" conversions (up to a given threshold)...
        if threshold is not None and float(num) > threshold:
            spnum = num.split(".", 1)
            while comma:
                spnum[0], n = FOUR_DIGIT_COMMA.subn(r"\1,\2", spnum[0])
                if n == 0:
                    break
            try:
                return f"{spnum[0]}.{spnum[1]}"
            except IndexError:
                return str(spnum[0])

        if group < 0 or group > 3:
            raise BadChunkingOptionError

        nowhite = num.lstrip()
        sign = (
            "plus"
            if nowhite.startswith("+")
            else "minus" if nowhite.startswith("-") else ""
        )

        if num in nth_suff:
            num = zero

        myord = num.endswith(tuple(nth_suff))
        if myord:
            num = num[:-2]

        finalpoint = False

        if decimal:
            chunks = num.split(".", 1) if group == 0 else num.split(".")
            finalpoint = chunks[-1] == ""
            if finalpoint:
                chunks.pop()

        else:
            chunks = [num]

        first = 1 if chunks[0] else 0
        loopstart = 1 if not first and len(chunks) > 1 else 0

        for i in range(loopstart, len(chunks)):
            chunk = NON_DIGIT.sub("", chunks[i] or "0")

            if group == 0 and (not first or first == ""):
                chunk = self.enword(chunk, 1)
            else:
                chunk = self.enword(chunk, group)

            chunk = chunk.rstrip(", ")
            chunk = WHITESPACES_COMMA.sub(",", chunk)
            if group == 0 and first:
                chunk = COMMA_WORD.sub(f" {andword} \\1", chunk)
            chunk = WHITESPACES.sub(" ", chunk)
            chunk = chunk.strip()

            if first:
                first = ""

            chunks[i] = chunk

        numchunks = []
        if first != 0:
            numchunks = chunks[0].split(f"{comma} ")

            if myord and numchunks:
                last_chunk = numchunks[-1]
                mo = ordinal_suff.search(last_chunk)
                numchunks[-1] = (
                    ordinal_suff.sub(ordinal.get(mo.group(1), ""), last_chunk)
                    if mo
                    else last_chunk + "th"
                )

            for chunk in chunks[1:]:
                numchunks.append(decimal)
                numchunks.extend(chunk.split(f"{comma} "))

            if finalpoint:
                numchunks.append(decimal)

            if wantlist:
                numchunks = [sign] + numchunks if sign else numchunks
                return numchunks
            elif group:
                signout = f"{sign} " if sign else ""
                return f"{signout}{', '.join(numchunks)}"
            else:
                signout = f"{sign} " if sign else ""
                num = f"{signout}{numchunks.pop(0)}"
                first = True if decimal is None else not num.endswith(decimal)
                num += "".join(
                    [
                        (
                            f" {nc}"
                            if not first
                            else f"{comma} {nc}" if nc == decimal else f" {nc}"
                        )
                        for nc in numchunks
                    ]
                )
                return num

    def enword(self, num: str, group: int) -> str:
        """Converts a numeric string to its word representation based on the specified grouping.

        Args:
            num (str): The numeric string to convert.
            group (int): The grouping of the numeric string.

        Returns:
            str: The word representation of the numeric string.

        Raises:
            NumOutOfRangeError: If the numeric string is out of range.

        """
        if group == 1:
            num = DIGIT_GROUP.sub(self.group1sub, num)
        elif group == 2:
            num = TWO_DIGITS.sub(self.group2sub, num)
            num = DIGIT_GROUP.sub(self.group1bsub, num, 1)
        elif group == 3:
            num = THREE_DIGITS.sub(self.group3sub, num)
            num = TWO_DIGITS.sub(self.group2sub, num, 1)
            num = DIGIT_GROUP.sub(self.group1sub, num, 1)
        elif int(num) == 0:
            return self.number_args["zero"]
        elif int(num) == 1:
            return self.number_args["one"]

        num = num.lstrip().lstrip("0")
        self.mill_count = 0
        while True:
            mo = THREE_DIGITS_WORD.search(num)
            if not mo:
                break
            num = THREE_DIGITS_WORD.sub(self.hundsub, num, 1)

        num = TWO_DIGITS_WORD.sub(self.tensub, num, 1)
        num = ONE_DIGIT_WORD.sub(self.unitsub, num, 1)

        return num

    def hundsub(self, mo: Match) -> str:
        """Substitution function for matching and converting a three-digit numeric group.

        Args:
            mo (Match): The matched object containing the captured groups.

        Returns:
            str: The word representation of the matched three-digit numeric group.

        """
        ret = self.hundfn(
            int(mo.group(1)), int(mo.group(2)), int(mo.group(3)), self.mill_count
        )
        self.mill_count += 1
        return ret

    def unitfn(self, units: int, mindex: int = 0) -> str:
        """Prettify string"""
        return f"{unit[units]}{self.millfn(mindex)}"

    def hundfn(self, hundreds: int, tens: int, units: int, mindex: int) -> str:
        """Constructs the word representation of a three-digit number.

        Args:
            hundreds (int): The hundreds digit.
            tens (int): The tens digit.
            units (int): The units digit.
            mindex (int): The mill index.

        Returns:
            str: The word representation of the three-digit number.

        """
        if hundreds:
            andword = f" {self.number_args['andword']} " if tens or units else ""
            return (
                f"{unit[hundreds]} hundred{andword}"
                f"{self.tenfn(tens, units)}{self.millfn(mindex)}, "
            )
        if tens or units:
            return f"{self.tenfn(tens, units)}{self.millfn(mindex)}, "
        return ""

    def tensub(self, mo: Match) -> str:
        """Substitution function for matching and converting a two-digit numeric group.

        Args:
            mo (Match): The matched object containing the captured groups.

        Returns:
            str: The word representation of the matched two-digit numeric group.

        """
        return f"{self.tenfn(int(mo.group(1)), int(mo.group(2)), self.mill_count)}, "

    def unitsub(self, mo: Match) -> str:
        """Substitution function for matching and converting a one-digit numeric group.

        Args:
            mo (Match): The matched object containing the captured groups.

        Returns:
            str: The word representation of the matched one-digit numeric group.

        """
        return f"{self.unitfn(int(mo.group(1)), self.mill_count)}, "
