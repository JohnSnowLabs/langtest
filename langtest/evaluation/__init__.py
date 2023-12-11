from typing import Any, Optional
from llama_index import ServiceContext
from llama_index.prompts.mixin import PromptDictType
from llama_index.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.evaluation import RetrieverEvaluator


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