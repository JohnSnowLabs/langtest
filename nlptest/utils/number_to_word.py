import ast
import re
import functools
import collections
import contextlib
from numbers import Number
from pydantic import Field, validate_arguments
from pydantic.typing import Annotated
from nlptest.transform.utils import (nth, ordinal, unit, teen, ten, mill, nth_suff, ordinal_suff)
from nlptest.transform.utils import NON_DIGIT, WHITESPACES, COMMA_WORD, WHITESPACES_COMMA, DIGIT_GROUP, TWO_DIGITS, THREE_DIGITS, THREE_DIGITS_WORD, TWO_DIGITS_WORD, ONE_DIGIT_WORD, FOUR_DIGIT_COMMA

from typing import (
    Union,
    Optional,
    List,
    Match,
    Any,
)


Word = Annotated[str, Field(min_length=1)]
Falsish = Any  # ideally, falsish would only validate on bool(value) is False

class BadChunkingOptionError(Exception):
    pass

class NumOutOfRangeError(Exception):
    pass

STDOUT_ON = False
def print3(txt: str) -> None:
    if STDOUT_ON:
        print(txt)

class engine:
    def __init__(self) -> None:
        self.mill_count = 0
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
        """
        Return a number in words.

        group = 1, 2 or 3 to group numbers before turning into words
        comma: define comma

        andword:
            word for 'and'. Can be set to ''.
            e.g. "one hundred and one" vs "one hundred one"

        zero: word for '0'
        one: word for '1'
        decimal: word for decimal point
        threshold: numbers above threshold not turned into words

        parameters not remembered from last call. Departure from Perl version.
        """
        self._number_args = {"andword": andword, "zero": zero, "one": one}
        num = str(num)

        # Handle "stylistic" conversions (up to a given threshold)...
        if threshold is not None and float(num) > threshold:
            spnum = num.split(".", 1)
            while comma:
                (spnum[0], n) = FOUR_DIGIT_COMMA.subn(r"\1,\2", spnum[0])
                if n == 0:
                    break
            try:
                return f"{spnum[0]}.{spnum[1]}"
            except IndexError:
                return str(spnum[0])

        if group < 0 or group > 3:
            raise BadChunkingOptionError
        nowhite = num.lstrip()
        if nowhite[0] == "+":
            sign = "plus"
        elif nowhite[0] == "-":
            sign = "minus"
        else:
            sign = ""

        if num in nth_suff:
            num = zero

        myord = num[-2:] in nth_suff
        if myord:
            num = num[:-2]
        finalpoint = False
        if decimal:
            if group != 0:
                chunks = num.split(".")
            else:
                chunks = num.split(".", 1)
            if chunks[-1] == "":  # remove blank string if nothing after decimal
                chunks = chunks[:-1]
                finalpoint = True  # add 'point' to end of output
        else:
            chunks = [num]

        first: Union[int, str, bool] = 1
        loopstart = 0

        if chunks[0] == "":
            first = 0
            if len(chunks) > 1:
                loopstart = 1

        for i in range(loopstart, len(chunks)):
            chunk = chunks[i]
            # remove all non numeric \D
            chunk = NON_DIGIT.sub("", chunk)
            if chunk == "":
                chunk = "0"

            if group == 0 and (first == 0 or first == ""):
                chunk = self.enword(chunk, 1)
            else:
                chunk = self.enword(chunk, group)

            if chunk[-2:] == ", ":
                chunk = chunk[:-2]
            chunk = WHITESPACES_COMMA.sub(",", chunk)

            if group == 0 and first:
                chunk = COMMA_WORD.sub(f" {andword} \\1", chunk)
            chunk = WHITESPACES.sub(" ", chunk)
            # chunk = re.sub(r"(\A\s|\s\Z)", self.blankfn, chunk)
            chunk = chunk.strip()
            if first:
                first = ""
            chunks[i] = chunk

        numchunks = []
        if first != 0:
            numchunks = chunks[0].split(f"{comma} ")

        if myord and numchunks:
            # TODO: can this be just one re as it is in perl?
            mo = ordinal_suff.search(numchunks[-1])
            if mo:
                numchunks[-1] = ordinal_suff.sub(ordinal[mo.group(1)], numchunks[-1])
            else:
                numchunks[-1] += "th"

        for chunk in chunks[1:]:
            numchunks.append(decimal)
            numchunks.extend(chunk.split(f"{comma} "))

        if finalpoint:
            numchunks.append(decimal)

        # wantlist: Perl list context. can explicitly specify in Python
        if wantlist:
            if sign:
                numchunks = [sign] + numchunks
            return numchunks
        elif group:
            signout = f"{sign} " if sign else ""
            return f"{signout}{', '.join(numchunks)}"
        else:
            signout = f"{sign} " if sign else ""
            num = f"{signout}{numchunks.pop(0)}"
            if decimal is None:
                first = True
            else:
                first = not num.endswith(decimal)
            for nc in numchunks:
                if nc == decimal:
                    num += f" {nc}"
                    first = 0
                elif first:
                    num += f"{comma} {nc}"
                else:
                    num += f" {nc}"
            return num
    def enword(self, num: str, group: int) -> str:
            # import pdb
            # pdb.set_trace()

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
                num = self._number_args["zero"]
            elif int(num) == 1:
                num = self._number_args["one"]
            else:
                num = num.lstrip().lstrip("0")
                self.mill_count = 0
                # surely there's a better way to do the next bit
                mo = THREE_DIGITS_WORD.search(num)
                while mo:
                    num = THREE_DIGITS_WORD.sub(self.hundsub, num, 1)
                    mo = THREE_DIGITS_WORD.search(num)
                num = TWO_DIGITS_WORD.sub(self.tensub, num, 1)
                num = ONE_DIGIT_WORD.sub(self.unitsub, num, 1)
            return num
    def hundsub(self, mo: Match) -> str:
            ret = self.hundfn(
                int(mo.group(1)), int(mo.group(2)), int(mo.group(3)), self.mill_count
            )
            self.mill_count += 1
            return ret
    def millfn(self, ind: int = 0) -> str:
            if ind > len(mill) - 1:
                print3("number out of range")
                raise NumOutOfRangeError
            return mill[ind]

    def unitfn(self, units: int, mindex: int = 0) -> str:
        return f"{unit[units]}{self.millfn(mindex)}"

    def tenfn(self, tens, units, mindex=0) -> str:
        if tens != 1:
            tens_part = ten[tens]
            if tens and units:
                hyphen = "-"
            else:
                hyphen = ""
            unit_part = unit[units]
            mill_part = self.millfn(mindex)
            return f"{tens_part}{hyphen}{unit_part}{mill_part}"
        return f"{teen[units]}{mill[mindex]}"

    def hundfn(self, hundreds: int, tens: int, units: int, mindex: int) -> str:
        if hundreds:
            andword = f" {self._number_args['andword']} " if tens or units else ""
            # use unit not unitfn as simpler
            return (
                f"{unit[hundreds]} hundred{andword}"
                f"{self.tenfn(tens, units)}{self.millfn(mindex)}, "
            )
        if tens or units:
            return f"{self.tenfn(tens, units)}{self.millfn(mindex)}, "
        return ""
    def tensub(self, mo: Match) -> str:
        return f"{self.tenfn(int(mo.group(1)), int(mo.group(2)), self.mill_count)}, "
    def unitsub(self, mo: Match) -> str:
        return f"{self.unitfn(int(mo.group(1)), self.mill_count)}, "
