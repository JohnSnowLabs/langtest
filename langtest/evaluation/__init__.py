import asyncio
from typing import Any, Dict, List, Optional, Sequence
from llama_index.bridge.pydantic import Field
from pydantic import BaseModel
from llama_index import ServiceContext
from llama_index.finetuning.embeddings.common import EmbeddingQAFinetuneDataset
from llama_index.prompts.mixin import PromptDictType
from llama_index.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.evaluation import RetrieverEvaluator, BaseRetrievalEvaluator
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.evaluation.retrieval.base import (
    BaseRetrievalEvaluator,
    RetrievalEvalMode,
    BaseRetrievalMetric,
    RetrievalEvalResult,
)
import pandas as pd
from collections import defaultdict

TESTS = {
    "uppercase": lambda x: x.upper(),
    "lowercase": lambda x: x.lower(),
    "titlecase": lambda x: x.title(),
    "capitalize": lambda x: x.capitalize(),
}


class Evaluator(BaseEvaluator):
    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        tests_config: Optional[dict] = None,
    ) -> None:
        self._service_context = service_context or ServiceContext.from_defaults()
        self._tests_config = tests_config or {}

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        pass

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        llm = self._service_context.llm_predictor

        print("Evaluating...")
        print("Query: ", query)
        print("Response: ", llm(query))
        print("Contexts: ", contexts)
        print("Reference: ", reference)
        print("kwargs: ", kwargs)
        return EvaluationResult(0.0, 0.0, 0.0, 0.0, 0.0)


class LangtestRetrieverEvaluator(RetrieverEvaluator):
    """ """

    eval_results = defaultdict(list)
    _service_context = ServiceContext.from_defaults()

    def __init__(
        self,
        metrics: Sequence[RetrieverEvaluator],
        retriever: BaseRetriever,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(metrics=metrics, retriever=retriever, **kwargs)
        self.retriever = retriever
        # self._config = config or TESTS.keys()

    def configure(self, config: dict):
        self._config = config

    async def _aget_retrieved_ids(
        self, query: str, mode: RetrievalEvalMode = RetrievalEvalMode.TEXT
    ) -> Sequence[str]:
        """Get retrieved ids."""
        retrieved_nodes = await self.retriever.aretrieve(query)
        return [node.node.node_id for node in retrieved_nodes]

    async def aevaluate_dataset(
        self,
        dataset: EmbeddingQAFinetuneDataset,
        workers: int = 2,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> Dict[str, List[RetrievalEvalResult]]:
        """Evaluate dataset."""
        response_jobs = defaultdict(list)
        mode = RetrievalEvalMode.from_str(dataset.mode)
        for query_id, query in dataset.queries.items():
            expected_ids = dataset.relevant_docs[query_id]

            # original test case
            original_response = self.eval_worker(
                query=query, expected_ids=expected_ids, mode=mode, workers=workers
            )
            response_jobs["original"].append(original_response)
            for test_type in list(TESTS.keys()):
                # test case
                test_case = TESTS[test_type](query)
                test_case_response = self.eval_worker(
                    query=test_case, expected_ids=expected_ids, mode=mode, workers=workers
                )
                response_jobs[test_type].append(test_case_response)

        for test_type, responses in response_jobs.items():
            responses = await asyncio.gather(*responses)
            self.eval_results[test_type] = responses

        return self.eval_results

    async def eval_worker(
        self,
        query: str,
        expected_ids: List[str],
        mode: RetrievalEvalMode,
        workers: int = 2,
    ) -> RetrievalEvalResult:
        """"""
        semaphore = asyncio.Semaphore(workers)
        async with semaphore:
            return await self.aevaluate(query, expected_ids=expected_ids, mode=mode)

    def display_results(
        self,
    ):
        metric_df = []
        for test_name, results in self.eval_results.items():
            metric_dicts = []
            for eval_result in results:
                metric_dict = eval_result.metric_vals_dict
                metric_dicts.append(metric_dict)

            full_df = pd.DataFrame(metric_dicts)

            hit_rate = full_df["hit_rate"].mean()
            mrr = full_df["mrr"].mean()

            metric_df.append(
                {
                    "Embeddings Model": self._service_context.embed_model.model_name,
                    "Test Name": test_name,
                    "Hit Rate": hit_rate,
                    "MRR": mrr,
                }
            )

        final_df = pd.DataFrame(metric_df)

        return final_df
