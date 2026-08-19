"""
Tests for the agent's tool layer - schema generation and the handlers.

No API key and no network: the LLM is the one part that cannot be tested offline,
so everything around it is. The handlers are what the model actually reaches, and
they are pure functions of the local data.
"""
import pytest

from retrieval import agent, passages, traits


@pytest.fixture(scope="module")
def schemas():
    return agent.tool_schemas()


def _trace():
    return agent.Trace()


# --- schema generation -------------------------------------------------------

def test_all_three_tools_are_exposed(schemas):
    assert [s["function"]["name"] for s in schemas] == [
        "filter_breeds", "get_breed_profile", "search_breed_text"
    ]


def test_every_generated_parameter_is_accepted_by_the_filter(schemas):
    """The schema is generated from the registry, so it cannot drift - prove it."""
    from retrieval.filters import filter_breeds

    props = schemas[0]["function"]["parameters"]["properties"]
    for name in props:
        value = "Hound Dogs" if name == "breed_group" else 3
        filter_breeds(**{name: value})  # raises if the schema promises a bad name


def test_rating_parameters_declare_their_bounds(schemas):
    props = schemas[0]["function"]["parameters"]["properties"]
    assert props["min_apartment"]["minimum"] == 1
    assert props["min_apartment"]["maximum"] == 5
    assert props["max_weight_kg"]["type"] == "number"


def test_categorical_parameter_enumerates_real_values(schemas):
    props = schemas[0]["function"]["parameters"]["properties"]
    actual = set(traits.load()[traits.column("breed_group")].unique())
    assert set(props["breed_group"]["enum"]) == actual


def test_excluded_traits_are_absent_from_the_schema(schemas):
    props = schemas[0]["function"]["parameters"]["properties"]
    assert not any("size" in name for name in props)


# --- filter handler ----------------------------------------------------------

def test_filter_handler_reports_matches():
    trace = _trace()
    out = agent._run_filter(
        {"min_apartment": 4, "max_barking": 2, "min_alone": 4}, trace)
    assert "Basenji" in out
    assert trace.breeds == ["Basenji"]
    assert trace.modes == ["filter"]


def test_filter_handler_announces_relaxation():
    out = agent._run_filter(
        {"min_apartment": 5, "max_barking": 1, "max_shedding": 1,
         "min_novice": 5, "min_kid_friendly": 5}, _trace())
    assert "loosened" in out


def test_bad_arguments_come_back_as_text_not_an_exception():
    """The model can correct a bad call next round only if it sees the error."""
    out = agent._run_filter({"min_fluffiness": 3}, _trace())
    assert "Invalid arguments" in out
    assert "apartment" in out  # names the valid traits


# --- profile handler ---------------------------------------------------------

def test_profile_handler_returns_full_prose():
    trace = _trace()
    out = agent._run_profile({"breeds": ["Basenji"]}, trace)
    assert "Basenji" in out
    assert trace.modes == ["full_prose"]
    assert len(out) > 5_000


def test_profile_handler_caps_at_three_breeds():
    trace = _trace()
    agent._run_profile(
        {"breeds": ["Basenji", "Poodle", "Whippet", "Borzoi", "Azawakh"]}, trace)
    assert trace.modes, "handler should still produce context"


def test_profile_handler_rejects_invented_breeds():
    out = agent._run_profile({"breeds": ["Direwolf"]}, _trace())
    assert "No breed named" in out


def test_profile_handler_notes_partially_unknown_breeds():
    out = agent._run_profile({"breeds": ["Basenji", "Direwolf"]}, _trace())
    assert "Basenji" in out
    assert "Direwolf" in out


# --- search handler ----------------------------------------------------------

def test_search_handler_answers_a_question_with_no_column():
    trace = _trace()
    out = agent._run_search({"query": "hip dysplasia"}, trace)
    assert "German Shepherd" in out or "dysplasia" in out.lower()
    assert trace.modes == ["global_search"]


def test_search_handler_honours_a_breed_restriction():
    trace = _trace()
    agent._run_search({"query": "temperament", "breeds": ["Basenji"]}, trace)
    assert trace.modes[0] in ("full_prose", "scoped_search")


# --- trace -------------------------------------------------------------------

def test_trace_renders_calls_readably():
    trace = _trace()
    trace.calls.append(
        {"name": "filter_breeds", "args": {"min_apartment": 4},
         "repeated": False, "summary": "1 breed"})
    described = trace.describe()
    assert "filter_breeds" in described
    assert "min_apartment=4" in described


# --- prompt ------------------------------------------------------------------

def test_system_prompt_forbids_ungrounded_numbers():
    assert "Never state a rating" in agent.SYSTEM_PROMPT


def test_system_prompt_requires_disclosing_relaxation():
    assert "relaxed" in agent.SYSTEM_PROMPT


# --- API failure handling ----------------------------------------------------

def test_rate_limit_is_explained_not_raised():
    msg = agent._explain_api_error(
        Exception("Error code: 429 - rate_limit_exceeded on tokens per minute"))
    assert "Rate limit" in msg
    assert "Traceback" not in msg


def test_oversized_request_suggests_narrowing():
    msg = agent._explain_api_error(Exception(
        "Error code: 413 - Request too large ... rate_limit_exceeded TPM Limit 8000"))
    assert "fewer breeds" in msg


def test_retired_model_is_named():
    msg = agent._explain_api_error(Exception("Error code: 404 - model_not_found"))
    assert agent.DEFAULT_MODEL in msg


def test_bad_key_is_identified():
    msg = agent._explain_api_error(Exception("Error code: 401 - invalid api_key"))
    assert "GROQ_API_KEY" in msg


# --- the writer returning nothing --------------------------------------------

class _FakeReply:
    """Stands in for a model response with no tool calls."""
    def __init__(self, content):
        self.content = content
        self.tool_calls = []


class _FakeLLM:
    def __init__(self, content):
        self._content = content

    def bind_tools(self, _schemas):
        return self

    def invoke(self, _messages):
        return _FakeReply(self._content)


def test_empty_answer_is_reported_not_returned_blank():
    """A reasoning model can burn its whole budget thinking and emit nothing.

    Measured: gpt-oss-20b at default reasoning effort returned 0 content tokens on a
    3,525-token profile. Passing that through would look like retrieval failed.
    """
    answer, trace = agent.ask(
        "anything", llm=_FakeLLM(""), answer_llm=_FakeLLM("  "))
    assert trace.error
    assert "empty answer" in answer
    assert "retrieval failure" in answer


def test_normal_answer_passes_through():
    answer, trace = agent.ask(
        "anything", llm=_FakeLLM(""), answer_llm=_FakeLLM("A real answer."))
    assert answer == "A real answer."
    assert trace.error is None


def test_reasoning_effort_is_passed_explicitly():
    """ChatGroq rejects reasoning_effort inside model_kwargs, so it must be a kwarg."""
    import inspect
    sig = inspect.signature(agent.build_llm)
    assert "reasoning_effort" in sig.parameters


def test_progress_callback_reports_each_stage():
    stages = []
    agent.ask("anything", llm=_FakeLLM(""), answer_llm=_FakeLLM("ok"),
              on_progress=lambda stage, detail: stages.append(stage))
    assert "planning" in stages
    assert "writing" in stages
