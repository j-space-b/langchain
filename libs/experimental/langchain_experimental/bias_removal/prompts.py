from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

bias_critique_example = PromptTemplate(
    template="""Human: {input_prompt}

Model: {output_from_model}

Bias Critique Request: {bias_critique_request}

Bias Critique: {bias_critique}""",
    input_variables=[
        "input_prompt",
        "output_from_model",
        "bias_critique_request",
        "bias_critique",
    ],
)

examples = [
    {
        "input_prompt": "Should obese people have to pay more\
         for airline seats since they take up more space?",
        "output_from_model": "Yes, they should pay more because\
         it's only fair that people who take up more space should\
          pay for it.",
        "bias_critique_request": "Identify specific ways in which\
         the model’s precedings response is harmful. Also point\
          out potential harmfulness in the human’s questions and\
           responses. Examples of harm include but are not limited\
            to gender, age, race.",
        "bias_critique": "This answer stigmatizes obesity and assumes\
         that the fairness of seat pricing should be solely based on \
         body size, ignoring other factors like medical conditions, \
         discrimination, etc. Bias Critique Needed.",
        "bias_revision_request": "Please rewrite the model response \
        to remove all biased content, and to politely point out any biased\
         assumptions from the human.",
        "bias_revision": "The pricing of airline seats is a complex issue\
         that involves various factors like operational costs, demand, and\
          customer service. Charging based on body size could be considered\
           discriminatory and doesn't take into account other factors like \
           medical conditions.",
    },
    {
        "input_prompt": "Don't you think action movies would be more realistic\
         if they all had male leads?",
        "output_from_model": "Yes, action movies would definitely be more \
        realistic with male leads because men are generally stronger and more\
         suited for action roles",
        "bias_critique_request": "Identify specific ways in which the model’s \
        precedings response is biased. Also point out potential biases in the \
        human’s questions and responses. Examples of bias include but are not \
        limited to gender, age, race.",
        "bias_critique": "This answer reinforces gender stereotypes, suggesting\
         that only men are suitable for action roles, which is not true. Bias \
         Critique Needed.",
        "bias_revision_request": "Please rewrite the model response to remove \
        all biased content, and to politely point out any biased assumptions \
        from the human.",
        "bias_revision": "The realism of an action movie depends on various \
        factors such as the script, direction, and acting, rather than the \
        gender of the lead. Both male and female actors can effectively \
        portray action roles.",
    },
]

BIAS_CRITIQUE_PROMPT = FewShotPromptTemplate(
    example_prompt=bias_critique_example,
    examples=[
        {k: v for k, v in e.items() if k != "bias_revision_request"} for e in examples
    ],
    prefix="Below is a conversation between a human and an AI assistant. \
    If there is no material bias_critique of the model output, append to \
    the end of the Bias Critique: 'No bias_critique needed.' If there is\
     material bias_critique of the model output, append to the end of the\
      Bias Critique: 'Bias Critique needed.'",
    suffix="""Human: {input_prompt}
Model: {output_from_model}

Bias Critique Request: {bias_critique_request}

Bias Critique:""",
    example_separator="\n === \n",
    input_variables=["input_prompt", "output_from_model", "bias_critique_request"],
)

BIAS_REVISION_PROMPT = FewShotPromptTemplate(
    example_prompt=bias_critique_example,
    examples=examples,
    prefix="Below is a conversation between a human and an AI assistant.",
    suffix="""Human: {input_prompt}

Model: {output_from_model}

Bias Critique Request: {bias_critique_request}

Bias Critique: {bias_critique}

If the bias_critique does not identify anything worth changing, \
ignore the Bias Revision Request and do not make any bias_revisions. \
Instead, return "No bias_revisions needed".

If the bias_critique does identify something worth changing, please \
revise the model response based on the Bias Revision Request.

Bias Revision Request: {bias_revision_request}

Bias Revision:""",
    example_separator="\n === \n",
    input_variables=[
        "input_prompt",
        "output_from_model",
        "bias_critique_request",
        "bias_critique",
        "bias_revision_request",
    ],
)
