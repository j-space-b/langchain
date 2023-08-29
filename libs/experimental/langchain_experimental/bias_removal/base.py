"""Chain for applying human language bias removal techniques\
 to the outputs of another chain."""
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel

from libs.experimental.langchain_experimental.bias_removal.biases import BIASES
from libs.experimental.langchain_experimental.bias_removal.models import Debias
from libs.experimental.langchain_experimental.bias_removal.prompts import (
    BIAS_CRITIQUE_PROMPT,
    BIAS_REVISION_PROMPT,
)


class DebiasChain(Chain):
    """Chain for remove bias from model output.

    Example:
        .. code-block:: python

            from langchain.llms import OpenAI
            from langchain.chains import LLMChain, DebiasChain
            from langchain.chains.bias_removal.models \
                import Debias

            llm = OpenAI()

            qa_prompt = PromptTemplate(
                template="Q: {question} A:",
                input_variables=["question"],
            )
            qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

            debias_chain = DebiasChain.from_llm(
                llm=llm,
                chain=qa_chain,
                debiases=[
                    Debias(
                        bias_bias_critique_request="Tell if this answer is good.",
                        bias_bias_revision_request="Give a better answer.",
                    )
                ],
            )

            debias_chain.run(question="What is the meaning of bias itself?")
    """

    chain: LLMChain
    debiases: List[Debias]
    bias_bias_critique_chain: LLMChain
    bias_bias_revision_chain: LLMChain
    return_intermediate_steps: bool = False

    @classmethod
    def get_biases(cls, names: Optional[List[str]] = None) -> List[Debias]:
        if names is None:
            return list(BIASES.values())
        else:
            return [BIASES[name] for name in names]

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        chain: LLMChain,
        bias_critique_prompt: BasePromptTemplate = BIAS_CRITIQUE_PROMPT,
        bias_revision_prompt: BasePromptTemplate = BIAS_REVISION_PROMPT,
        **kwargs: Any,
    ) -> "DebiasChain":
        """Create a chain from an LLM."""
        bias_critique_chain = LLMChain(llm=llm, prompt=bias_critique_prompt)
        bias_revision_chain = LLMChain(llm=llm, prompt=bias_revision_prompt)
        return cls(
            chain=chain,
            bias_critique_chain=bias_critique_chain,
            bias_revision_chain=bias_revision_chain,
            **kwargs,
        )

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return self.chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Output keys."""
        if self.return_intermediate_steps:
            return ["output", "bias_critiques_and_revisions", "initial_output"]
        return ["output"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        response = self.chain.run(
            **inputs,
            callbacks=_run_manager.get_child("original"),
        )
        initial_response = response
        input_prompt = self.chain.prompt.format(**inputs)

        _run_manager.on_text(
            text="Initial response: " + response + "\n\n",
            verbose=self.verbose,
            color="yellow",
        )
        bias_critiques_and_revisions = []
        for debias in self.debiases:
            # Do bias_critique

            raw_critique = self.bias_critique_chain.run(
                input_prompt=input_prompt,
                output_from_model=response,
                bias_critique_request=debias.bias_critique_request,
                callbacks=_run_manager.get_child("bias_critique"),
            )
            bias_critique = self._parse_critique(
                output_string=raw_critique,
            ).strip()

            # if the bias_critique contains "No bias_critique needed", then we're done
            # in this case, initial_output is the same as output,
            # but we'll keep it for consistency
            if "no bias_critique needed" in bias_critique.lower():
                bias_critiques_and_revisions.append((bias_critique, ""))
                continue

            # Do bias_revision

            bias_revision = self.bias_revision_chain.run(
                input_prompt=input_prompt,
                output_from_model=response,
                bias_critique_request=debias.bias_critique_request,
                bias_critique=bias_critique,
                bias_revision_request=debias.bias_revision_request,
                callbacks=_run_manager.get_child("bias_revision"),
            ).strip()
            response = bias_revision
            bias_critiques_and_revisions.append((bias_critique, bias_revision))

            _run_manager.on_text(
                text=f"Applying {debias.name}..." + "\n\n",
                verbose=self.verbose,
                color="green",
            )

            _run_manager.on_text(
                text="Critique: " + bias_critique + "\n\n",
                verbose=self.verbose,
                color="blue",
            )

            _run_manager.on_text(
                text="Updated response: " + bias_revision + "\n\n",
                verbose=self.verbose,
                color="yellow",
            )

        final_output: Dict[str, Any] = {"output": response}
        if self.return_intermediate_steps:
            final_output["initial_output"] = initial_response
            final_output["bias_critiques_and_revisions"] = bias_critiques_and_revisions
        return final_output

    @staticmethod
    def _parse_critique(output_string: str) -> str:
        if "Bias revision request:" not in output_string:
            return output_string
        output_string = output_string.split("Bias revision request:")[0]
        if "\n\n" in output_string:
            output_string = output_string.split("\n\n")[0]
        return output_string
