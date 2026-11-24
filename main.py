import os
import streamlit as st
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END


# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Candidate Screening",
    page_icon="🧑‍💼",
    layout="wide",
)

st.title("🧑‍💼 AI Candidate Screening")
st.caption(
    "A LangGraph-powered screening workflow that evaluates experience level "
    "and Python Developer skill fit."
)

# -----------------------------
# API key configuration
# -----------------------------
def get_openai_api_key():
    """Read the key from Streamlit secrets first, then environment variables."""
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        # Streamlit secrets may not be configured during local development.
        pass
    return os.getenv("OPENAI_API_KEY")


api_key = get_openai_api_key()

if not api_key:
    st.warning(
        "OpenAI API key not found. Add OPENAI_API_KEY to your environment "
        "or Streamlit secrets before running a screening."
    )
    with st.expander("How to configure the API key"):
        st.markdown(
            """
            **Local run:** create a `.streamlit/secrets.toml` file:
            ```toml
            OPENAI_API_KEY = "your-openai-api-key"
            ```

            **Streamlit Community Cloud:** open your app's **Settings → Secrets**
            and add the same line. Do not commit your API key to GitHub.
            """
        )


# -----------------------------
# LangGraph workflow
# -----------------------------
class ScreeningState(TypedDict, total=False):
    application: str
    experience_level: str
    skill_match: str
    response: str


@st.cache_resource
def build_screening_graph(openai_api_key: str):
    """Build and cache the LangGraph screening workflow."""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=openai_api_key,
        temperature=0,
    )

    def categorize_experience(state: ScreeningState) -> ScreeningState:
        prompt = ChatPromptTemplate.from_template(
            """
            Classify the candidate's overall professional experience into exactly
            one of these labels: Entry-level, Mid-level, or Senior-level.

            Return only the label, with no explanation.

            Candidate application:
            {application}
            """
        )
        result = (prompt | llm).invoke(
            {"application": state["application"]}
        ).content
        return {"experience_level": str(result).strip()}

    def assess_skillset(state: ScreeningState) -> ScreeningState:
        prompt = ChatPromptTemplate.from_template(
            """
            Assess whether this candidate is a skills match for a Python Developer
            role based on the application text.

            Return exactly one of these labels: Match or No Match.
            Return no explanation.

            Candidate application:
            {application}
            """
        )
        result = (prompt | llm).invoke(
            {"application": state["application"]}
        ).content
        return {"skill_match": str(result).strip()}

    def schedule_hr_interview(state: ScreeningState) -> ScreeningState:
        return {
            "response": "Candidate has been shortlisted for an HR interview."
        }

    def escalate_to_recruiter(state: ScreeningState) -> ScreeningState:
        return {
            "response": (
                "Candidate has senior-level experience but doesn't match "
                "job skills. Escalate to a recruiter for review."
            )
        }

    def reject_application(state: ScreeningState) -> ScreeningState:
        return {
            "response": "Candidate doesn't meet the job requirements and has been rejected."
        }

    def route_app(state: ScreeningState) -> str:
        skill_match = state.get("skill_match", "").strip().lower()
        experience = state.get("experience_level", "").strip().lower()

        if skill_match.startswith("match") and not skill_match.startswith("no match"):
            return "schedule_hr_interview"
        if "senior" in experience:
            return "escalate_to_recruiter"
        return "reject_application"

    workflow = StateGraph(ScreeningState)

    workflow.add_node("categorize_experience", categorize_experience)
    workflow.add_node("assess_skillset", assess_skillset)
    workflow.add_node("schedule_hr_interview", schedule_hr_interview)
    workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
    workflow.add_node("reject_application", reject_application)

    workflow.add_edge(START, "categorize_experience")
    workflow.add_edge("categorize_experience", "assess_skillset")
    workflow.add_conditional_edges("assess_skillset", route_app)

    workflow.add_edge("schedule_hr_interview", END)
    workflow.add_edge("escalate_to_recruiter", END)
    workflow.add_edge("reject_application", END)

    return workflow.compile()


def run_candidate_screening(application: str, graph):
    result = graph.invoke({"application": application})
    return {
        "experience_level": result.get("experience_level", "Not available"),
        "skill_match": result.get("skill_match", "Not available"),
        "response": result.get("response", "No recommendation returned."),
    }


# -----------------------------
# User interface
# -----------------------------
st.subheader("Candidate application")
st.write(
    "Paste the candidate's resume summary or application below. "
    "The workflow will classify experience, assess Python skills, and return a recommendation."
)

with st.form("candidate_screening_form"):
    application_text = st.text_area(
        "Application / Resume summary",
        height=220,
        placeholder=(
            "Example: I have 5 years of experience in software engineering "
            "with expertise in Python, Django, REST APIs, and SQL."
        ),
    )
    submitted = st.form_submit_button(
        "Run candidate screening",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not application_text.strip():
        st.error("Please enter the candidate's application before screening.")
    elif not api_key:
        st.error("Configure your OpenAI API key before running the workflow.")
    else:
        try:
            with st.spinner("Analyzing the candidate with LangGraph..."):
                screening_graph = build_screening_graph(api_key)
                results = run_candidate_screening(
                    application_text.strip(),
                    screening_graph,
                )

            st.success("Screening completed.")

            st.subheader("Screening results")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Experience level", results["experience_level"])

            with col2:
                st.metric("Python skill match", results["skill_match"])

            st.subheader("Final recommendation")
            response = results["response"]

            if "shortlisted" in response.lower():
                st.success(response, icon="✅")
            elif "recruiter" in response.lower():
                st.warning(response, icon="⚠️")
            else:
                st.error(response, icon="❌")

            with st.expander("View submitted application"):
                st.write(application_text.strip())

        except Exception as exc:
            st.error("The screening workflow could not be completed.")
            st.exception(exc)

st.divider()
st.caption(
    "Note: This is an AI-assisted screening demo. Review outputs before making "
    "employment decisions; model results may be inaccurate or biased."
)
