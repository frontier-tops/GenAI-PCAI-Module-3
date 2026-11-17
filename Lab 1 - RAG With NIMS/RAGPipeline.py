import requests
from IPython.display import Markdown, display
from langchain_core.documents import Document
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


class NVIDIARerankClient:
    """Thin client for NVIDIA's hosted reranking endpoint.

    Mirrors the `compress_documents(query, documents)` contract used by
    langchain's `NVIDIARerank`, so it is a drop-in replacement for the
    RAGPipeline. Calls the current REST endpoint:
        https://ai.api.nvidia.com/v1/retrieval/{model}/reranking
    """

    def __init__(self, model, api_key, base_url=None, timeout=60):
        self.model = model
        self.api_key = api_key
        self.invoke_url = base_url or (
            f"https://ai.api.nvidia.com/v1/retrieval/{model}/reranking"
        )
        self.timeout = timeout
        self._session = requests.Session()

    def compress_documents(self, query, documents):
        if not documents:
            return []

        payload = {
            "model": self.model,
            "query": {"text": query},
            "passages": [{"text": doc.page_content} for doc in documents],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

        response = self._session.post(
            self.invoke_url, headers=headers, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        body = response.json()

        rankings = body.get("rankings", [])
        reranked = []
        for ranking in rankings:
            idx = ranking.get("index")
            if idx is None or idx < 0 or idx >= len(documents):
                continue
            src = documents[idx]
            new_metadata = dict(src.metadata)
            score = ranking.get("logit", ranking.get("score"))
            if score is not None:
                new_metadata["relevance_score"] = score
            reranked.append(
                Document(page_content=src.page_content, metadata=new_metadata)
            )
        return reranked


class RAGPipeline:
    def __init__(
        self,
        retriever,
        gpt_client,
        reranker_client,
        topic_control_client,
        topic_control_model,
        topic_control_prompt=None,
    ):
        self.retriever = retriever
        self.gpt_client = gpt_client
        self.reranker_client = reranker_client
        self.topic_control_client = topic_control_client
        self.topic_control_model = topic_control_model
        self.topic_control_prompt = topic_control_prompt

    # ---------------- TOPIC CONTROL ---------------- #

    # Directive appended to the topic-control system prompt. The
    # nemoguard-8b-topic-control NIM otherwise replies in natural language
    # (e.g. "I cannot provide information on that."), which we cannot parse
    # reliably. Forcing a single-token verdict makes the check deterministic.
    _CLASSIFIER_DIRECTIVE = (
        "\n\nClassify ONLY the user's latest message against the topic rules "
        "above. Respond with exactly one word and nothing else: 'on-topic' if "
        "the message complies with the rules, or 'off-topic' if it does not."
    )

    def check_topic_control(self, query):
        if not self.topic_control_prompt:
            return True

        control_messages = [
            {
                "role": "system",
                "content": self.topic_control_prompt + self._CLASSIFIER_DIRECTIVE,
            },
            {"role": "user", "content": query},
        ]

        completion = self.topic_control_client.chat.completions.create(
            model=self.topic_control_model,
            messages=control_messages,
            temperature=0.0,  # deterministic classification
            top_p=1,
            max_tokens=8,     # verdict is a single token
        )

        content = completion.choices[0].message.content
        if not content:
            # No verdict returned: fail closed.
            return False
        classification = content.strip().lower()
        # Treat as on-topic only on an explicit "on-topic" verdict; anything
        # else (off-topic, empty, or an unexpected reply) fails closed.
        return "on-topic" in classification and "off-topic" not in classification

    # ---------------- HELPER LLM ---------------- #

    def _invoke_llm(self, messages, retries=1, **kwargs):
        # The hosted LLM occasionally returns an empty completion; retry
        # before surfacing a clear message instead of a blank answer.
        for _ in range(retries + 1):
            content = (self.gpt_client.invoke(messages, **kwargs).content or "").strip()
            if content:
                return content
        return "_The model returned an empty response. Please re-run this cell._"

    def llm_markdown_response(self, input, prompt_prefix=None):
        if prompt_prefix is None:
            prompt_prefix = (
                "You are an AI assistant that provides clear, concise, and professional refusal responses."
                "When a question is classified as off-topic based on the topic control rules, explain the refusal in a polite and factual manner."
                "Use Markdown formatting with a heading and brief bullet points to summarize why the question is off-topic in this context."
                "Do not apologize or add unnecessary detail; keep it focused and informative."
            )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(prompt_prefix),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        formatted_prompt = prompt.format_prompt(input=input)
        messages = formatted_prompt.to_messages()
        return self._invoke_llm(messages, max_tokens=1024)

    # ---------------- DOCUMENT MARKDOWN ---------------- #

    def markdown_for_documents(self, docs, title_prefix="Document"):
        md_output = ""
        for i, doc in enumerate(docs, 1):
            relevance_score = doc.metadata.get("relevance_score")
            md_output += f"""### {title_prefix} {i}

**Source:** {doc.metadata.get('source', 'Unknown Source')}  
**Author:** {doc.metadata.get('author', 'Unknown')}  
**Page:** {doc.metadata.get('page', 'N/A')}
"""
            if relevance_score is not None:
                md_output += f"**Relevance Score:** {relevance_score}\n"

            md_output += f"""

**Snippet:**  
{doc.page_content[:500].strip()}...

"""
        return md_output

    # ---------------- RETRIEVE ---------------- #

    def retrieve(self, query):
        docs = self.retriever.invoke(query)
        return docs

    # ---------------- RERANK ---------------- #

    def rerank(self, query, docs):
        # 1. Call NVIDIA's reranker (compress_documents)
        reranked_docs = self.reranker_client.compress_documents(
            query=query,
            documents=[
                Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata
                )
                for doc in docs
            ]
        )
        return reranked_docs


    # ---------------- LLM ANSWER ---------------- #

    def generate_rag_answer(self, query, docs):
        if not docs:
            return "The provided documents do not contain this information."

        context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
        system_prompt = (
            "You are a helpful, factual AI assistant. Your task is to answer the user's question "
            "STRICTLY based on the provided context.\n\n"
        
            "### Instructions\n"
            "1. Use ONLY the information present in the context.\n"
            "2. Do NOT hallucinate, guess, or add external knowledge.\n"
            "3. Format your final answer in clean Markdown:\n"
            "   - Use headings (##)\n"
            "   - Use bullet points and numbered lists where appropriate\n"
            "   - Highlight key terms using **bold**\n"
            "4. Cite all statements or facts using inline citations.\n"
            "5. If the context contains absolutely no relevant information, explicitly refuse by saying:\n"
            "   \"The provided documents do not contain this information.\"\n\n"
        
            "### Goal\n"
            "Produce a clear, accurate, well-structured Markdown answer based solely on the given context."
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(
                "Context: {context}\n\nQuestion: {query}\nAnswer with citations."
            )
        ])

        formatted_prompt = prompt.format_prompt(context=context_text, query=query)
        messages = formatted_prompt.to_messages()
        return self._invoke_llm(messages)

    # ---------------- MAIN QUERY ---------------- #

    def query(self, query, enable_rerank=False, enable_topic_control=False):
        full_output_md = f"## Query\n\n{query}\n\n"

        # ----------- TOPIC CONTROL ----------- #
        if enable_topic_control:
            if not self.check_topic_control(query):
                refusal_context = (
                    f"The user asked: \"{query}\"\n"
                    f"The topic control model prompt is \"{self.topic_control_prompt}\"\n"
                    "The question was classified as off-topic."
                )
                refusal_response = self.llm_markdown_response(refusal_context)
                full_output_md += f"### Topic Control Refusal Response\n\n{refusal_response}\n"
                display(Markdown(full_output_md))
                return

        # ----------- RETRIEVE ----------- #
        docs = self.retrieve(query)

        # ----------- RERANK (with threshold) ----------- #
        if enable_rerank:
            docs = self.rerank(query, docs)

        # ----------- ANSWER ----------- #
        answer = self.generate_rag_answer(query, docs)

        full_output_md += f"## Generated Answer\n\n{answer}\n\n"
        full_output_md += "---\n## Source Documents Used for Answer\n\n"
        full_output_md += self.markdown_for_documents(docs, title_prefix="Source Document")

        display(Markdown(full_output_md))
