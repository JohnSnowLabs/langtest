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
        realigned_actual_results = sample._get_realigned_spans()
        assert realigned_actual_results == sample.expected_results.predictions

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
        realigned_actual_results = sample._get_realigned_spans()
        assert realigned_actual_results == sample.expected_results.predictions

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
        realigned_actual_results = sample._get_realigned_spans()
        assert realigned_actual_results == sample.expected_results.predictions

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
        realigned_actual_results = sample._get_realigned_spans()
        assert realigned_actual_results == sample.expected_results.predictions

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
        realigned_actual_results = sample._get_realigned_spans()
        assert realigned_actual_results == sample.expected_results.predictions

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
                    ignore=True
                )
            ],
            expected_results=NEROutput(
                predictions=[NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))]
            ),
            actual_results=NEROutput(
                predictions=[NERPrediction(entity="PROD", span=Span(start=10, end=18, word="McDonald"))]
            ),
        )
        realigned_actual_results = sample._get_realigned_spans()
        assert realigned_actual_results == sample.expected_results.predictions

        sample = Sample(
            original="I do love KFC",
            test_type="contraction",
            test_case="I do love Kentucky Fried Chicken",
            transformations=[
                Transformation(
                    original_span=Span(start=10, end=13, word="KFC"),
                    new_span=Span(start=10, end=32, word="Kentucky Fried Chicken"),
                    ignore=True
                )
            ],
            expected_results=NEROutput(
                predictions=[NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))]
            ),
            actual_results=NEROutput(
                predictions=[NERPrediction(entity="PROD", span=Span(start=10, end=32, word="Kentucky Fried Chicken"))]
            ),
        )
        realigned_actual_results = sample._get_realigned_spans()
        assert realigned_actual_results == sample.expected_results.predictions

        sample = Sample(
            original="I do love McDonald",
            test_type="contraction",
            test_case="I do love KFC",
            transformations=[
                Transformation(
                    new_span=Span(start=10, end=13, word="KFC"),
                    original_span=Span(start=10, end=18, word="McDonald"),
                    ignore=True
                )
            ],
            actual_results=NEROutput(
                predictions=[NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))]
            ),
            expected_results=NEROutput(
                predictions=[NERPrediction(entity="PROD", span=Span(start=10, end=18, word="McDonald"))]
            ),
        )
        realigned_actual_results = sample._get_realigned_spans()
        assert realigned_actual_results == sample.expected_results.predictions

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
        realigned_actual_results = sample._get_realigned_spans()
        assert realigned_actual_results == sample.expected_results.predictions
