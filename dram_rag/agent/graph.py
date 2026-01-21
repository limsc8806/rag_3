from __future__ import annotations

from dataclasses import asdict

from ..config import Settings
from ..generation.answer_grader import grade_generation
from ..generation.llm_clients import make_llm_client
from ..generation.rag_chain import generate_answer
from ..index.stores import IndexBundle
from ..retrieval.graders import filter_relevant
from ..retrieval.query_rewrite import rewrite_query
from ..retrieval.retrievers import retrieve
from ..types import AgentState


class AdaptiveRAGAgent:
    """A deterministic approximation of the Adaptive RAG agent.

    - retrieve -> grade docs -> (rewrite loop) -> generate -> grade answer -> (rewrite loop)

    Notes:
    - The provided notebook uses LLM-based graders and LangGraph.
    - This implementation keeps the same control flow, but uses deterministic
      graders by default so it can run without external dependencies.
    """

    def __init__(self, settings: Settings, index_bundle: IndexBundle):
        self.settings = settings
        self.bundle = index_bundle
        self.llm = make_llm_client(settings.llm)

    def run(self, question: str) -> AgentState:
        st = AgentState(question=question, query=question)

        max_loops = max(0, int(self.settings.agent.max_loops))
        for loop in range(max_loops + 1):
            st.loop_count = loop

            # Retrieve
            retrieved = retrieve(
                self.bundle,
                st.query,
                top_k_text=self.settings.retrieval.top_k_text,
                top_k_img=self.settings.retrieval.top_k_img,
                merge_top_k=self.settings.retrieval.merge_top_k,
            )
            st.trace.append({"node": "retrieve", "loop": loop, "query": st.query, "n": len(retrieved)})

            # Grade documents
            filtered = filter_relevant(retrieved, min_relevance_score=self.settings.retrieval.min_relevance_score)
            st.retrieved = filtered
            st.trace.append({"node": "grade_documents", "loop": loop, "kept": len(filtered)})

            if not filtered:
                # Query rewrite
                new_query = rewrite_query(st.question, prev_query=st.query)
                st.trace.append({"node": "transform_query", "loop": loop, "new_query": new_query})
                st.query = new_query
                continue

            # Generate
            answer = generate_answer(st.question, filtered, self.llm)
            st.generation = answer
            st.trace.append({"node": "generate", "loop": loop, "answer_chars": len(answer)})

            # Grade generation
            g = grade_generation(st.question, answer)
            st.trace.append({"node": "grade_generation", "loop": loop, **asdict(g)})

            if g.supported and g.useful:
                break

            # Otherwise rewrite and loop
            st.query = rewrite_query(st.question, prev_query=st.query)
            st.trace.append({"node": "transform_query", "loop": loop, "new_query": st.query, "reason": g.reason})

        return st
