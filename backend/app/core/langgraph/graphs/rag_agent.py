from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from app.core.llm.provider import get_chat_model
from app.core.vector_store.client import VectorStoreManager


class RAGState(TypedDict):
    """State for the Agentic RAG subgraph."""

    question: str
    documents: list[Document]
    generation: str
    query_rewrite_count: int
    route: str  # "vectorstore", "web_search", "direct"


GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.""",
        ),
        (
            "human",
            "Retrieved document:\n{document}\n\nUser question: {question}\n\nRelevant (yes/no):",
        ),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a question rewriter. Improve the question to get better search results.",
        ),
        ("human", "Original question: {question}\n\nImproved question:"),
    ]
)

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a grader assessing whether an LLM generation is grounded in a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'yes' means the answer is grounded in the facts.""",
        ),
        ("human", "Facts:\n{documents}\n\nLLM Generation: {generation}\n\nGrounded (yes/no):"),
    ]
)

GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant answering questions based on provided context.
Use the following retrieved documents to answer the question.
If you don't know the answer from the context, say so clearly.

Context:
{context}""",
        ),
        ("human", "{question}"),
    ]
)

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a router that decides how to handle a user question.
Choose one of:
- 'vectorstore': The question can be answered from the knowledge base
- 'direct': The question is a greeting or simple question that doesn't need retrieval
Respond with just the word.""",
        ),
        ("human", "{question}"),
    ]
)


def build_rag_agent(
    collection_names: list[str],
    model_name: str | None = None,
    k: int = 5,
) -> Any:
    """Build a Corrective/Adaptive RAG agent as a LangGraph subgraph.

    Flow:
    1. CLASSIFY - Decide if retrieval is needed
    2. RETRIEVE - Search vector store
    3. GRADE - Evaluate document relevance
    4. DECIDE - Generate, rewrite query, or fallback
    5. GENERATE - Produce answer
    6. CHECK - Hallucination check
    """
    model = get_chat_model(model=model_name)
    vector_store = VectorStoreManager()

    def classify(state: RAGState) -> RAGState:
        """Route the question to vectorstore or direct answer."""
        chain = ROUTER_PROMPT | model | StrOutputParser()
        route = chain.invoke({"question": state["question"]}).strip().lower()
        if "vectorstore" in route:
            state["route"] = "vectorstore"
        else:
            state["route"] = "direct"
        return state

    def retrieve(state: RAGState) -> RAGState:
        """Retrieve documents from vector store."""
        docs = vector_store.cross_collection_search(
            query=state["question"],
            collections=collection_names,
            k=k,
        )
        state["documents"] = docs
        return state

    def grade_documents(state: RAGState) -> RAGState:
        """Grade retrieved documents for relevance."""
        grader = GRADER_PROMPT | model | StrOutputParser()
        relevant_docs = []

        for doc in state.get("documents", []):
            result = grader.invoke(
                {
                    "document": doc.page_content,
                    "question": state["question"],
                }
            )
            if "yes" in result.lower():
                relevant_docs.append(doc)

        state["documents"] = relevant_docs
        return state

    def decide_next(state: RAGState) -> Literal["generate", "rewrite"]:
        """Decide whether to generate or rewrite the query."""
        if state.get("documents"):
            return "generate"
        if state.get("query_rewrite_count", 0) >= 2:
            return "generate"  # Give up rewriting, generate with what we have
        return "rewrite"

    def rewrite_query(state: RAGState) -> RAGState:
        """Rewrite the query for better retrieval."""
        chain = REWRITE_PROMPT | model | StrOutputParser()
        new_question = chain.invoke({"question": state["question"]})
        state["question"] = new_question.strip()
        state["query_rewrite_count"] = state.get("query_rewrite_count", 0) + 1
        return state

    def generate(state: RAGState) -> RAGState:
        """Generate answer from retrieved documents."""
        context = "\n\n".join(doc.page_content for doc in state.get("documents", []))
        chain = GENERATE_PROMPT | model | StrOutputParser()
        generation = chain.invoke(
            {
                "context": context or "No relevant documents found.",
                "question": state["question"],
            }
        )
        state["generation"] = generation
        return state

    def direct_answer(state: RAGState) -> RAGState:
        """Generate a direct answer without retrieval."""
        result = model.invoke([HumanMessage(content=state["question"])])
        state["generation"] = result.content
        return state

    def hallucination_check(state: RAGState) -> Literal["pass", "fail"]:
        """Check if the generation is grounded in documents."""
        if not state.get("documents"):
            return "pass"  # No docs to check against

        checker = HALLUCINATION_PROMPT | model | StrOutputParser()
        docs_text = "\n\n".join(doc.page_content for doc in state["documents"])
        result = checker.invoke(
            {
                "documents": docs_text,
                "generation": state["generation"],
            }
        )
        return "pass" if "yes" in result.lower() else "fail"

    def route_question(state: RAGState) -> Literal["retrieve", "direct"]:
        """Route based on classification."""
        return "retrieve" if state.get("route") == "vectorstore" else "direct"

    # Build the graph
    workflow = StateGraph(RAGState)

    workflow.add_node("classify", classify)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_node("direct_answer", direct_answer)

    workflow.set_entry_point("classify")
    workflow.add_conditional_edges(
        "classify",
        route_question,
        {
            "retrieve": "retrieve",
            "direct": "direct_answer",
        },
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_next,
        {
            "generate": "generate",
            "rewrite": "rewrite_query",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        hallucination_check,
        {
            "pass": END,
            "fail": "generate",  # Re-generate
        },
    )
    workflow.add_edge("direct_answer", END)

    return workflow.compile()
