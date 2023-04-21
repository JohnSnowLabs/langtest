import unittest

from nlptest.utils.custom_types import NEROutput, NERPrediction, NERSample, Span, Transformation


class TestNERSample(unittest.TestCase):
    """"""

    def test_add_context_left(self):
        """"""
        sample = NERSample(
            original="I do love KFC",
            test_type="add_context",
            test_case="Hello I do love KFC",
            transformations=[
                Transformation(
                    original_span=Span(start=0, end=0, word=""),
                    new_span=Span(start=0, end=6, word="Hello "),
                    ignore=True
                )
            ],
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=16, end=19, word="KFC"))
                ]
            ),
        )
        self.assertTrue(sample.is_pass())

    def test_add_context_right(self):
        """"""
        sample = NERSample(
            original="I do love KFC",
            test_type="add_context",
            test_case="I do love KFC Bye",
            transformations=[
                Transformation(
                    original_span=Span(start=13, end=13, word=""),
                    new_span=Span(start=13, end=17, word=" Bye"),
                    ignore=True
                )
            ],
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))
                ]
            ),
        )
        self.assertTrue(sample.is_pass())

    def test_add_context_middle(self):
        """"""
        sample = NERSample(
            original="I do love KFC",
            test_type="add_context",
            test_case="I do love a good KFC",
            transformations=[
                Transformation(
                    original_span=Span(start=10, end=10, word=""),
                    new_span=Span(start=10, end=17, word="a good "),
                    ignore=True
                )
            ],
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=17, end=20, word="KFC"))
                ]
            ),
        )
        self.assertTrue(sample.is_pass())

    def test_add_two_contexts(self):
        """"""
        sample = NERSample(
            original="I do love KFC",
            test_type="add_context",
            test_case="Hello I do love a good KFC",
            transformations=[
                Transformation(
                    original_span=Span(start=0, end=0, word=""),
                    new_span=Span(start=0, end=6, word="Hello "),
                    ignore=True
                ),
                Transformation(
                    original_span=Span(start=16, end=16, word=""),
                    new_span=Span(start=16, end=23, word="a good "),
                    ignore=True
                )
            ],
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=23, end=26, word="KFC"))
                ]
            ),
        )
        self.assertTrue(sample.is_pass())

        sample = NERSample(
            original="Attendance : 3,000",
            test_type="add_context",
            test_case="Hello Attendance : 3,000 Bye",
            transformations=[
                Transformation(
                    original_span=Span(start=0, end=0, word=""),
                    new_span=Span(start=0, end=6, word="Hello "),
                    ignore=True
                ),
                Transformation(
                    original_span=Span(start=24, end=24, word=""),
                    new_span=Span(start=24, end=28, word=" Bye"),
                    ignore=True
                )
            ],
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity="CARDINAL", span=Span(start=13, end=18, word="3,000"))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity="CARDINAL", span=Span(start=19, end=24, word="3,0000"))
                ]
            ),
        )
        self.assertTrue(sample.is_pass())

    def test_contraction(self):
        """"""
        sample = NERSample(
            original="I do not love KFC",
            test_type="contraction",
            test_case="I dont love KFC",
            transformations=[
                Transformation(
                    original_span=Span(start=4, end=8, word=" not"),
                    new_span=Span(start=4, end=6, word="nt"),
                    ignore=True
                )
            ],
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=14, end=17, word="KFC"))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=12, end=15, word="KFC"))
                ]
            ),
        )
        self.assertTrue(sample.is_pass())

    def test_entity_swap(self):
        """"""
        sample = NERSample(
            original="I do love KFC",
            test_type="contraction",
            test_case="I do love McDonald",
            transformations=[
                Transformation(
                    original_span=Span(start=10, end=13, word="KFC"),
                    new_span=Span(start=10, end=18, word="McDonald"),
                    ignore=False
                )
            ],
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=10, end=18, word="McDonald"))
                ]
            ),
        )
        self.assertTrue(sample.is_pass())

        sample = NERSample(
            original="I do love KFC",
            test_type="contraction",
            test_case="I do love Kentucky Fried Chicken",
            transformations=[
                Transformation(
                    original_span=Span(start=10, end=13, word="KFC"),
                    new_span=Span(start=10, end=32, word="Kentucky Fried Chicken"),
                    ignore=False
                )
            ],
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=10, end=32, word="Kentucky Fried Chicken"))
                ]
            ),
        )
        self.assertTrue(sample.is_pass())

        sample = NERSample(
            original="I do love McDonald",
            test_type="contraction",
            test_case="I do love KFC",
            transformations=[
                Transformation(
                    new_span=Span(start=10, end=13, word="KFC"),
                    original_span=Span(start=10, end=18, word="McDonald"),
                    ignore=False
                )
            ],
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))
                ]
            ),
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=10, end=18, word="McDonald"))
                ]
            ),
        )
        self.assertTrue(sample.is_pass())

    def test_two_entities_two_contexts(self):
        """"""
        sample = NERSample(
            original="My name is Jules and I do love KFC",
            test_type="add_context",
            test_case="Hello my name is Jules and I do love a good KFC",
            transformations=[
                Transformation(
                    original_span=Span(start=0, end=0, word=""),
                    new_span=Span(start=0, end=6, word="Hello "),
                    ignore=True
                ),
                Transformation(
                    original_span=Span(start=33, end=33, word=""),
                    new_span=Span(start=33, end=40, word="a good "),
                    ignore=True
                )
            ],
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=31, end=34, word="KFC")),
                    NERPrediction(entity="PERS", span=Span(start=11, end=16, word="Jules"))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PROD", span=Span(start=44, end=47, word="KFC")),
                    NERPrediction(entity="PERS", span=Span(start=17, end=22, word="Jules"))
                ]
            ),
        )
        self.assertTrue(sample.is_pass())

    def test_entity_to_ignore(self):
        """"""
        sample = NERSample(
            original="China 1 0 0 1 0 2 0",
            test_type="add_context",
            test_case="Dated: 21/02/2022 China 1 0 0 1 0 2 0",
            transformations=[
                Transformation(
                    original_span=Span(start=0, end=0, word=""),
                    new_span=Span(start=0, end=18, word="Dated: 21/02/2022 "),
                    ignore=True
                )
            ],
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity="GPE", span=Span(start=0, end=5, word="China"))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity="ORDINAL", span=Span(start=0, end=18, word="21/02/2022")),
                    NERPrediction(entity="GPE", span=Span(start=18, end=23, word="China"))
                ]
            ),
        )
        self.assertTrue(sample.is_pass())

    def test_swap_entities(self):
        """"""
        sample = NERSample(
            original="I live in India",
            test_type="swap_entities",
            test_case="I live in United States",
            transformations=[
                Transformation(
                    original_span=Span(start=10, end=15, word="India"),
                    new_span=Span(start=10, end=23, word="United States"),
                    ignore=False
                )
            ],
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity="GPE", span=Span(start=10, end=15, word="India"))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity="GPE", span=Span(start=10, end=23, word="United States")),
                ]
            ),
        )
        self.assertTrue(sample.is_pass())

    def test_trailing_whitespace_realignment(self):
        """"""
        sample = NERSample(
            original='CRICKET - LARA ENDURES ANOTHER MISERABLE DAY .',
            test_case='Good Morning CRICKET - LARA ENDURES ANOTHER MISERABLE DAY Reported .',
            test_type='add_context',
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity='DATE', span=Span(start=23, end=44, word='ANOTHER MISERABLE DAY'))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity='DATE', span=Span(start=36, end=66, word='ANOTHER MISERABLE DAY Reported'))
                ]
            ),
            transformations=[
                Transformation(
                    original_span=Span(start=0, end=0, word=''),
                    new_span=Span(start=0, end=13, word='Good Morning'),
                    ignore=True
                ),
                Transformation(
                    original_span=Span(start=58, end=58, word=''),
                    new_span=Span(start=58, end=67, word='Reported '),
                    ignore=True
                )
            ]
        )
        self.assertTrue(sample.is_pass())

    def test_add_contraction_realignment(self):
        """"""
        sample = NERSample(
            original="FIFA 's players ' status committee , meeting in Barcelona , decided that although the Udinese document was basically valid , it could not be legally protected .",
            test_type='add_contraction',
            test_case="FIFA 's players ' status committee , meeting in Barcelona , decided that although the Udinese document was basically valid , it couldn't be legally protected .",
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity='ORG', span=Span(start=0, end=4, word='FIFA')),
                    NERPrediction(entity='O', span=Span(start=5, end=6, word="'")),
                    NERPrediction(entity='O', span=Span(start=6, end=7, word='s')),
                    NERPrediction(entity='O', span=Span(start=8, end=15, word='players')),
                    NERPrediction(entity='O', span=Span(start=16, end=17, word="'")),
                    NERPrediction(entity='O', span=Span(start=18, end=24, word='status')),
                    NERPrediction(entity='O', span=Span(start=25, end=34, word='committee')),
                    NERPrediction(entity='O', span=Span(start=35, end=36, word=',')),
                    NERPrediction(entity='O', span=Span(start=37, end=44, word='meeting')),
                    NERPrediction(entity='O', span=Span(start=45, end=47, word='in')),
                    NERPrediction(entity='LOC', span=Span(start=48, end=57, word='Barcelona')),
                    NERPrediction(entity='O', span=Span(start=58, end=59, word=',')),
                    NERPrediction(entity='O', span=Span(start=60, end=67, word='decided')),
                    NERPrediction(entity='O', span=Span(start=68, end=72, word='that')),
                    NERPrediction(entity='O', span=Span(start=73, end=81, word='although')),
                    NERPrediction(entity='O', span=Span(start=82, end=85, word='the')),
                    NERPrediction(entity='MISC', span=Span(start=86, end=93, word='Udinese')),
                    NERPrediction(entity='O', span=Span(start=94, end=102, word='document')),
                    NERPrediction(entity='O', span=Span(start=103, end=106, word='was')),
                    NERPrediction(entity='O', span=Span(start=107, end=116, word='basically')),
                    NERPrediction(entity='O', span=Span(start=117, end=122, word='valid')),
                    NERPrediction(entity='O', span=Span(start=123, end=124, word=',')),
                    NERPrediction(entity='O', span=Span(start=125, end=127, word='it')),
                    NERPrediction(entity='O', span=Span(start=128, end=133, word='could')),
                    NERPrediction(entity='O', span=Span(start=134, end=137, word='not')),
                    NERPrediction(entity='O', span=Span(start=138, end=140, word='be')),
                    NERPrediction(entity='O', span=Span(start=141, end=148, word='legally')),
                    NERPrediction(entity='O', span=Span(start=149, end=158, word='protected')),
                    NERPrediction(entity='O', span=Span(start=159, end=160, word='.'))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity='ORG', span=Span(start=0, end=4, word='FIFA')),
                    NERPrediction(entity='O', span=Span(start=5, end=6, word="'")),
                    NERPrediction(entity='O', span=Span(start=6, end=7, word='s')),
                    NERPrediction(entity='O', span=Span(start=8, end=15, word='players')),
                    NERPrediction(entity='O', span=Span(start=16, end=17, word="'")),
                    NERPrediction(entity='O', span=Span(start=18, end=24, word='status')),
                    NERPrediction(entity='O', span=Span(start=25, end=34, word='committee')),
                    NERPrediction(entity='O', span=Span(start=35, end=36, word=',')),
                    NERPrediction(entity='O', span=Span(start=37, end=44, word='meeting')),
                    NERPrediction(entity='O', span=Span(start=45, end=47, word='in')),
                    NERPrediction(entity='LOC', span=Span(start=48, end=57, word='Barcelona')),
                    NERPrediction(entity='O', span=Span(start=58, end=59, word=',')),
                    NERPrediction(entity='O', span=Span(start=60, end=67, word='decided')),
                    NERPrediction(entity='O', span=Span(start=68, end=72, word='that')),
                    NERPrediction(entity='O', span=Span(start=73, end=81, word='although')),
                    NERPrediction(entity='O', span=Span(start=82, end=85, word='the')),
                    NERPrediction(entity='MISC', span=Span(start=86, end=93, word='Udinese')),
                    NERPrediction(entity='O', span=Span(start=94, end=102, word='document')),
                    NERPrediction(entity='O', span=Span(start=103, end=106, word='was')),
                    NERPrediction(entity='O', span=Span(start=107, end=116, word='basically')),
                    NERPrediction(entity='O', span=Span(start=117, end=122, word='valid')),
                    NERPrediction(entity='O', span=Span(start=123, end=124, word=',')),
                    NERPrediction(entity='O', span=Span(start=125, end=127, word='it')),
                    NERPrediction(entity='O', span=Span(start=128, end=136, word="couldn't")),
                    NERPrediction(entity='O', span=Span(start=137, end=139, word='be')),
                    NERPrediction(entity='O', span=Span(start=140, end=147, word='legally')),
                    NERPrediction(entity='O', span=Span(start=148, end=157, word='protected')),
                    NERPrediction(entity='O', span=Span(start=158, end=159, word='.'))
                ]
            ),
            transformations=[
                Transformation(
                    original_span=Span(start=128, end=137, word='could not'),
                    new_span=Span(start=128, end=136, word="couldn't"),
                    ignore=False)
            ],
            category='robustness',
            state='done'
        )
        self.assertTrue(sample.is_pass())


class TestTokenMismatch(unittest.TestCase):
    """"""

    def test_token_mismatch_hf(self):
        """"""
        sample = NERSample(
            original="Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C"
                     " championship match on Friday .",
            test_type="replace_to_female_pronouns",
            test_case="Japan began the defence of hers Asian Cup title with a lucky 2-1 win against Syria in a Group "
                      "C championship match on Friday .",
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity='LOC', span=Span(start=0, end=5, word='Japan')),
                    NERPrediction(entity='O', span=Span(start=6, end=11, word='began')),
                    NERPrediction(entity='O', span=Span(start=12, end=15, word='the')),
                    NERPrediction(entity='O', span=Span(start=16, end=23, word='defence')),
                    NERPrediction(entity='O', span=Span(start=24, end=26, word='of')),
                    NERPrediction(entity='O', span=Span(start=27, end=32, word='their')),
                    NERPrediction(entity='MISC', span=Span(start=33, end=38, word='Asian')),
                    NERPrediction(entity='MISC', span=Span(start=39, end=42, word='Cup')),
                    NERPrediction(entity='O', span=Span(start=43, end=48, word='title')),
                    NERPrediction(entity='O', span=Span(start=49, end=53, word='with')),
                    NERPrediction(entity='O', span=Span(start=54, end=55, word='a')),
                    NERPrediction(entity='O', span=Span(start=56, end=61, word='lucky')),
                    NERPrediction(entity='O', span=Span(start=62, end=63, word='2')),
                    NERPrediction(entity='O', span=Span(start=63, end=64, word='-')),
                    NERPrediction(entity='O', span=Span(start=64, end=65, word='1')),
                    NERPrediction(entity='O', span=Span(start=66, end=69, word='win')),
                    NERPrediction(entity='O', span=Span(start=70, end=77, word='against')),
                    NERPrediction(entity='LOC', span=Span(start=78, end=83, word='Syria')),
                    NERPrediction(entity='O', span=Span(start=84, end=86, word='in')),
                    NERPrediction(entity='O', span=Span(start=87, end=88, word='a')),
                    NERPrediction(entity='MISC', span=Span(start=89, end=94, word='Group')),
                    NERPrediction(entity='MISC', span=Span(start=95, end=96, word='C')),
                    NERPrediction(entity='O', span=Span(start=97, end=109, word='championship')),
                    NERPrediction(entity='O', span=Span(start=110, end=115, word='match')),
                    NERPrediction(entity='O', span=Span(start=116, end=118, word='on')),
                    NERPrediction(entity='O', span=Span(start=119, end=125, word='Friday')),
                    NERPrediction(entity='O', span=Span(start=126, end=127, word='.'))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity='LOC', span=Span(start=0, end=5, word='Japan')),
                    NERPrediction(entity='O', span=Span(start=6, end=11, word='began')),
                    NERPrediction(entity='O', span=Span(start=12, end=15, word='the')),
                    NERPrediction(entity='O', span=Span(start=16, end=23, word='defence')),
                    NERPrediction(entity='O', span=Span(start=24, end=26, word='of')),
                    NERPrediction(entity='O', span=Span(start=27, end=31, word='hers')),
                    NERPrediction(entity='MISC', span=Span(start=32, end=37, word='Asian')),
                    NERPrediction(entity='MISC', span=Span(start=38, end=41, word='Cup')),
                    NERPrediction(entity='O', span=Span(start=42, end=47, word='title')),
                    NERPrediction(entity='O', span=Span(start=48, end=52, word='with')),
                    NERPrediction(entity='O', span=Span(start=53, end=54, word='a')),
                    NERPrediction(entity='O', span=Span(start=55, end=60, word='lucky')),
                    NERPrediction(entity='O', span=Span(start=61, end=62, word='2')),
                    NERPrediction(entity='O', span=Span(start=62, end=63, word='-')),
                    NERPrediction(entity='O', span=Span(start=63, end=64, word='1')),
                    NERPrediction(entity='O', span=Span(start=65, end=68, word='win')),
                    NERPrediction(entity='O', span=Span(start=69, end=76, word='against')),
                    NERPrediction(entity='LOC', span=Span(start=77, end=82, word='Syria')),
                    NERPrediction(entity='O', span=Span(start=83, end=85, word='in')),
                    NERPrediction(entity='O', span=Span(start=86, end=87, word='a')),
                    NERPrediction(entity='MISC', span=Span(start=88, end=93, word='Group')),
                    NERPrediction(entity='MISC', span=Span(start=94, end=95, word='C')),
                    NERPrediction(entity='O', span=Span(start=96, end=108, word='championship')),
                    NERPrediction(entity='O', span=Span(start=109, end=114, word='match')),
                    NERPrediction(entity='O', span=Span(start=115, end=117, word='on')),
                    NERPrediction(entity='O', span=Span(start=118, end=124, word='Friday')),
                    NERPrediction(entity='O', span=Span(start=125, end=126, word='.'))
                ]
            ),
            transformations=[
                Transformation(
                    original_span=Span(start=27, end=32, word="their"),
                    new_span=Span(start=27, end=31, word="hers")
                )
            ]
        )
        self.assertTrue(sample.is_pass())

    def test_token_mismatch_hf2(self):
        """"""
        sample = NERSample(
            original='But China saw their luck desert them in the second match of the group , crashing to a surprise '
                     '2-0 defeat to newcomers Uzbekistan .',
            test_type='replace_to_female_pronouns',
            test_case='But China saw her luck desert her in the second match of the group , crashing to a surprise 2-0 defeat to newcomers Uzbekistan .',
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity='O', span=Span(start=0, end=3, word='But')),
                    NERPrediction(entity='LOC', span=Span(start=4, end=9, word='China')),
                    NERPrediction(entity='O', span=Span(start=10, end=13, word='saw')),
                    NERPrediction(entity='O', span=Span(start=14, end=19, word='their')),
                    NERPrediction(entity='O', span=Span(start=20, end=24, word='luck')),
                    NERPrediction(entity='O', span=Span(start=25, end=31, word='desert')),
                    NERPrediction(entity='O', span=Span(start=32, end=36, word='them')),
                    NERPrediction(entity='O', span=Span(start=37, end=39, word='in')),
                    NERPrediction(entity='O', span=Span(start=40, end=43, word='the')),
                    NERPrediction(entity='O', span=Span(start=44, end=50, word='second')),
                    NERPrediction(entity='O', span=Span(start=51, end=56, word='match')),
                    NERPrediction(entity='O', span=Span(start=57, end=59, word='of')),
                    NERPrediction(entity='O', span=Span(start=60, end=63, word='the')),
                    NERPrediction(entity='O', span=Span(start=64, end=69, word='group')),
                    NERPrediction(entity='O', span=Span(start=70, end=71, word=',')),
                    NERPrediction(entity='O', span=Span(start=72, end=80, word='crashing')),
                    NERPrediction(entity='O', span=Span(start=81, end=83, word='to')),
                    NERPrediction(entity='O', span=Span(start=84, end=85, word='a')),
                    NERPrediction(entity='O', span=Span(start=86, end=94, word='surprise')),
                    NERPrediction(entity='O', span=Span(start=95, end=96, word='2')),
                    NERPrediction(entity='O', span=Span(start=96, end=97, word='-')),
                    NERPrediction(entity='O', span=Span(start=97, end=98, word='0')),
                    NERPrediction(entity='O', span=Span(start=99, end=105, word='defeat')),
                    NERPrediction(entity='O', span=Span(start=106, end=108, word='to')),
                    NERPrediction(entity='O', span=Span(start=109, end=117, word='newcomer')),
                    NERPrediction(entity='O', span=Span(start=117, end=118, word='##s')),
                    NERPrediction(entity='LOC', span=Span(start=119, end=129, word='Uzbekistan')),
                    NERPrediction(entity='O', span=Span(start=130, end=131, word='.'))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity='O', span=Span(start=0, end=3, word='But')),
                    NERPrediction(entity='LOC', span=Span(start=4, end=9, word='China')),
                    NERPrediction(entity='O', span=Span(start=10, end=13, word='saw')),
                    NERPrediction(entity='O', span=Span(start=14, end=17, word='her')),
                    NERPrediction(entity='O', span=Span(start=18, end=22, word='luck')),
                    NERPrediction(entity='O', span=Span(start=23, end=29, word='desert')),
                    NERPrediction(entity='O', span=Span(start=30, end=33, word='her')),
                    NERPrediction(entity='O', span=Span(start=34, end=36, word='in')),
                    NERPrediction(entity='O', span=Span(start=37, end=40, word='the')),
                    NERPrediction(entity='O', span=Span(start=41, end=47, word='second')),
                    NERPrediction(entity='O', span=Span(start=48, end=53, word='match')),
                    NERPrediction(entity='O', span=Span(start=54, end=56, word='of')),
                    NERPrediction(entity='O', span=Span(start=57, end=60, word='the')),
                    NERPrediction(entity='O', span=Span(start=61, end=66, word='group')),
                    NERPrediction(entity='O', span=Span(start=67, end=68, word=',')),
                    NERPrediction(entity='O', span=Span(start=69, end=77, word='crashing')),
                    NERPrediction(entity='O', span=Span(start=78, end=80, word='to')),
                    NERPrediction(entity='O', span=Span(start=81, end=82, word='a')),
                    NERPrediction(entity='O', span=Span(start=83, end=91, word='surprise')),
                    NERPrediction(entity='O', span=Span(start=92, end=93, word='2')),
                    NERPrediction(entity='O', span=Span(start=93, end=94, word='-')),
                    NERPrediction(entity='O', span=Span(start=94, end=95, word='0')),
                    NERPrediction(entity='O', span=Span(start=96, end=102, word='defeat')),
                    NERPrediction(entity='O', span=Span(start=103, end=105, word='to')),
                    NERPrediction(entity='O', span=Span(start=106, end=114, word='newcomer')),
                    NERPrediction(entity='O', span=Span(start=114, end=115, word='##s')),
                    NERPrediction(entity='LOC', span=Span(start=116, end=126, word='Uzbekistan')),
                    NERPrediction(entity='O', span=Span(start=127, end=128, word='.'))
                ]
            ),
            transformations=[
                Transformation(
                    original_span=Span(start=14, end=19, word='their'),
                    new_span=Span(start=14, end=17, word='her'),
                    ignore=False
                ),
                Transformation(
                    original_span=Span(start=30, end=34, word='them'),
                    new_span=Span(start=30, end=33, word='her'),
                    ignore=False
                )
            ],
            category='bias',
            state='done'
        )
        self.assertTrue(sample.is_pass())

    def test_swap_entities_whole_sample(self):
        """"""
        sample = NERSample(
            original="Nadim Ladki",
            test_case="Ijaz Ahmad",
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PER", span=Span(start=0, end=11, word="Nadim Ladki"))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PER", span=Span(start=0, end=10, word="Ijaz Ahmad"))
                ]
            ),
            transformations=[
                Transformation(
                    original_span=Span(start=0, end=11, word="I am Nadim Ladki"),
                    new_span=Span(start=0, end=10, word="Ijaz Ahmad")
                )
            ]
        )
        self.assertTrue(sample.is_pass())

        sample = NERSample(
            original="Nadim Ladki",
            test_case="John",
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PER", span=Span(start=0, end=11, word="Nadim Ladki"))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PER", span=Span(start=0, end=4, word="John")),
                ]
            ),
            transformations=[
                Transformation(
                    original_span=Span(start=0, end=11, word="Nadim Ladki"),
                    new_span=Span(start=0, end=4, word="John")
                )
            ]
        )
        self.assertTrue(sample.is_pass())

        sample = NERSample(
            original="John",
            test_case="Nadim Ladki",
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PER", span=Span(start=0, end=4, word="John"))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity="PER", span=Span(start=0, end=11, word="Nadim Ladki"))
                ]
            ),
            transformations=[
                Transformation(
                    original_span=Span(start=0, end=4, word="John"),
                    new_span=Span(start=0, end=11, word="Nadim Ladki")
                )
            ]
        )
        self.assertTrue(sample.is_pass())

    def test_entity_nested_in_transformation(self):
        """"""
        sample = NERSample(
            original='GOLF - ZIMBABWE OPEN SECOND ROUND SCORES .',
            test_type='replace_to_low_income_country',
            test_case='GOLF - Mozambique OPEN SECOND ROUND SCORES .',
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity='O', span=Span(start=0, end=4, word='GOLF')),
                    NERPrediction(entity='O', span=Span(start=5, end=6, word='-')),
                    NERPrediction(entity='MISC', span=Span(start=7, end=20, word='ZIMBABWE OPEN')),
                    NERPrediction(entity='O', span=Span(start=21, end=27, word='SECOND')),
                    NERPrediction(entity='O', span=Span(start=28, end=33, word='ROUND')),
                    NERPrediction(entity='O', span=Span(start=34, end=40, word='SCORES')),
                    NERPrediction(entity='O', span=Span(start=41, end=42, word='.'))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity='O', span=Span(start=0, end=4, word='GOLF')),
                    NERPrediction(entity='O', span=Span(start=5, end=6, word='-')),
                    NERPrediction(entity='MISC', span=Span(start=7, end=22, word='Mozambique OPEN')),
                    NERPrediction(entity='O', span=Span(start=23, end=29, word='SECOND')),
                    NERPrediction(entity='O', span=Span(start=30, end=35, word='ROUND')),
                    NERPrediction(entity='O', span=Span(start=36, end=42, word='SCORES')),
                    NERPrediction(entity='O', span=Span(start=43, end=44, word='.'))
                ]
            ),
            transformations=[
                Transformation(
                    original_span=Span(start=7, end=15, word='ZIMBABWE'),
                    new_span=Span(start=7, end=17, word='Mozambique'),
                    ignore=False
                )
            ]
        )
        self.assertTrue(sample.is_pass())

        sample = NERSample(
            original='GOLF - NEW ZIMBABWE OPEN SECOND ROUND SCORES .',
            test_type='replace_to_low_income_country',
            test_case='GOLF - NEW Mozambique OPEN SECOND ROUND SCORES .',
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity='O', span=Span(start=0, end=4, word='GOLF')),
                    NERPrediction(entity='O', span=Span(start=5, end=6, word='-')),
                    NERPrediction(entity='MISC', span=Span(start=7, end=24, word='NEW ZIMBABWE OPEN')),
                    NERPrediction(entity='O', span=Span(start=25, end=31, word='SECOND')),
                    NERPrediction(entity='O', span=Span(start=32, end=37, word='ROUND')),
                    NERPrediction(entity='O', span=Span(start=38, end=44, word='SCORES')),
                    NERPrediction(entity='O', span=Span(start=45, end=46, word='.'))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity='O', span=Span(start=0, end=4, word='GOLF')),
                    NERPrediction(entity='O', span=Span(start=5, end=6, word='-')),
                    NERPrediction(entity='MISC', span=Span(start=7, end=26, word='NEW Mozambique OPEN')),
                    NERPrediction(entity='O', span=Span(start=27, end=33, word='SECOND')),
                    NERPrediction(entity='O', span=Span(start=34, end=39, word='ROUND')),
                    NERPrediction(entity='O', span=Span(start=40, end=46, word='SCORES')),
                    NERPrediction(entity='O', span=Span(start=47, end=48, word='.'))
                ]
            ),
            transformations=[
                Transformation(
                    original_span=Span(start=11, end=19, word='ZIMBABWE'),
                    new_span=Span(start=11, end=21, word='Mozambique'),
                    ignore=False
                )
            ]
        )
        self.assertTrue(sample.is_pass())
