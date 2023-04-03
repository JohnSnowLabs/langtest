from nlptest.utils.custom_types import NEROutput, NERPrediction, Sample, Span, Transformation


class TestSample:
    def test_add_context_left(self):
        """"""
        sample = Sample(
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
                predictions=[NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))]
            ),
            actual_results=NEROutput(
                predictions=[NERPrediction(entity="PROD", span=Span(start=16, end=19, word="KFC"))]
            ),
        )
        assert sample.is_pass()

    def test_add_context_right(self):
        """"""
        sample = Sample(
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
                predictions=[NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))]
            ),
            actual_results=NEROutput(
                predictions=[NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))]
            ),
        )
        assert sample.is_pass()

    def test_add_context_middle(self):
        """"""
        sample = Sample(
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
                predictions=[NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))]
            ),
            actual_results=NEROutput(
                predictions=[NERPrediction(entity="PROD", span=Span(start=17, end=20, word="KFC"))]
            ),
        )
        assert sample.is_pass()

    def test_add_two_contexts(self):
        """"""
        sample = Sample(
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
                predictions=[NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))]
            ),
            actual_results=NEROutput(
                predictions=[NERPrediction(entity="PROD", span=Span(start=23, end=26, word="KFC"))]
            ),
        )
        assert sample.is_pass()

        sample = Sample(
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
                    original_span=Span(start=18, end=18, word=""),
                    new_span=Span(start=18, end=22, word=" Bye"),
                    ignore=True
                )
            ],
            expected_results=NEROutput(
                predictions=[NERPrediction(entity="CARDINAL", span=Span(start=13, end=18, word="3,000"))]
            ),
            actual_results=NEROutput(
                predictions=[NERPrediction(entity="CARDINAL", span=Span(start=19, end=24, word="KFC"))]
            ),
        )
        assert sample.is_pass()

    def test_contraction(self):
        """"""
        sample = Sample(
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
                predictions=[NERPrediction(entity="PROD", span=Span(start=14, end=17, word="KFC"))]
            ),
            actual_results=NEROutput(
                predictions=[NERPrediction(entity="PROD", span=Span(start=12, end=15, word="KFC"))]
            ),
        )
        assert sample.is_pass()

    def test_entity_swap(self):
        """"""
        sample = Sample(
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
                predictions=[NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))]
            ),
            actual_results=NEROutput(
                predictions=[NERPrediction(entity="PROD", span=Span(start=10, end=18, word="McDonald"))]
            ),
        )
        assert sample.is_pass()

        sample = Sample(
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
                predictions=[NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))]
            ),
            actual_results=NEROutput(
                predictions=[NERPrediction(entity="PROD", span=Span(start=10, end=32, word="Kentucky Fried Chicken"))]
            ),
        )
        assert sample.is_pass()

        sample = Sample(
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
                predictions=[NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))]
            ),
            expected_results=NEROutput(
                predictions=[NERPrediction(entity="PROD", span=Span(start=10, end=18, word="McDonald"))]
            ),
        )
        assert sample.is_pass()

    def test_two_entities_two_contexts(self):
        """"""
        sample = Sample(
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
        assert sample.is_pass()

    def test_entity_to_ignore(self):
        """"""
        sample = Sample(
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
        assert sample.is_pass()

    def test_swap_entities(self):
        """"""
        sample = Sample(
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
        assert sample.is_pass()

    def test_token_mismatch_hf(self):
        """"""
        sample = Sample(
            original="Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday .",
            test_type="replace_to_female_pronouns",
            test_case="Japan began the defence of hers Asian Cup title with a lucky 2-1 win against Syria in a Group "
                      "C championship match on Friday .",
            expected_results=NEROutput(
                predictions=[
                    NERPrediction(entity='B-LOC', span=Span(start=0, end=5, word='Japan')),
                    NERPrediction(entity='O', span=Span(start=6, end=11, word='began')),
                    NERPrediction(entity='O', span=Span(start=12, end=15, word='the')),
                    NERPrediction(entity='O', span=Span(start=16, end=23, word='defence')),
                    NERPrediction(entity='O', span=Span(start=24, end=26, word='of')),
                    NERPrediction(entity='O', span=Span(start=27, end=32, word='their')),
                    NERPrediction(entity='B-MISC', span=Span(start=33, end=38, word='Asian')),
                    NERPrediction(entity='I-MISC', span=Span(start=39, end=42, word='Cup')),
                    NERPrediction(entity='O', span=Span(start=43, end=48, word='title')),
                    NERPrediction(entity='O', span=Span(start=49, end=53, word='with')),
                    NERPrediction(entity='O', span=Span(start=54, end=55, word='a')),
                    NERPrediction(entity='O', span=Span(start=56, end=61, word='lucky')),
                    NERPrediction(entity='O', span=Span(start=62, end=63, word='2')),
                    NERPrediction(entity='O', span=Span(start=63, end=64, word='-')),
                    NERPrediction(entity='O', span=Span(start=64, end=65, word='1')),
                    NERPrediction(entity='O', span=Span(start=66, end=69, word='win')),
                    NERPrediction(entity='O', span=Span(start=70, end=77, word='against')),
                    NERPrediction(entity='B-LOC', span=Span(start=78, end=83, word='Syria')),
                    NERPrediction(entity='O', span=Span(start=84, end=86, word='in')),
                    NERPrediction(entity='O', span=Span(start=87, end=88, word='a')),
                    NERPrediction(entity='B-MISC', span=Span(start=89, end=94, word='Group')),
                    NERPrediction(entity='I-MISC', span=Span(start=95, end=96, word='C')),
                    NERPrediction(entity='O', span=Span(start=97, end=109, word='championship')),
                    NERPrediction(entity='O', span=Span(start=110, end=115, word='match')),
                    NERPrediction(entity='O', span=Span(start=116, end=118, word='on')),
                    NERPrediction(entity='O', span=Span(start=119, end=125, word='Friday')),
                    NERPrediction(entity='O', span=Span(start=126, end=127, word='.'))
                ]
            ),
            actual_results=NEROutput(
                predictions=[
                    NERPrediction(entity='B-LOC', span=Span(start=0, end=5, word='Japan')),
                    NERPrediction(entity='O', span=Span(start=6, end=11, word='began')),
                    NERPrediction(entity='O', span=Span(start=12, end=15, word='the')),
                    NERPrediction(entity='O', span=Span(start=16, end=23, word='defence')),
                    NERPrediction(entity='O', span=Span(start=24, end=26, word='of')),
                    NERPrediction(entity='O', span=Span(start=27, end=31, word='hers')),
                    NERPrediction(entity='B-MISC', span=Span(start=32, end=37, word='Asian')),
                    NERPrediction(entity='I-MISC', span=Span(start=38, end=41, word='Cup')),
                    NERPrediction(entity='O', span=Span(start=42, end=47, word='title')),
                    NERPrediction(entity='O', span=Span(start=48, end=52, word='with')),
                    NERPrediction(entity='O', span=Span(start=53, end=54, word='a')),
                    NERPrediction(entity='O', span=Span(start=55, end=60, word='lucky')),
                    NERPrediction(entity='O', span=Span(start=61, end=62, word='2')),
                    NERPrediction(entity='O', span=Span(start=62, end=63, word='-')),
                    NERPrediction(entity='O', span=Span(start=63, end=64, word='1')),
                    NERPrediction(entity='O', span=Span(start=65, end=68, word='win')),
                    NERPrediction(entity='O', span=Span(start=69, end=76, word='against')),
                    NERPrediction(entity='B-LOC', span=Span(start=77, end=82, word='Syria')),
                    NERPrediction(entity='O', span=Span(start=83, end=85, word='in')),
                    NERPrediction(entity='O', span=Span(start=86, end=87, word='a')),
                    NERPrediction(entity='B-MISC', span=Span(start=88, end=93, word='Group')),
                    NERPrediction(entity='I-MISC', span=Span(start=94, end=95, word='C')),
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
        assert sample.is_pass()
