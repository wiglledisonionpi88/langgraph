from syrupy import SnapshotAssertion


def test_chain(snapshot: SnapshotAssertion) -> None:
    from typing import Annotated, TypedDict

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    # useful to generate SQL query
    model_low_temp = ChatOpenAI(temperature=0.1)
    # useful to generate natural language outputs
    model_high_temp = ChatOpenAI(temperature=0.7)

    class State(TypedDict):
        # to track conversation history
        messages: Annotated[list, add_messages]
        # input
        user_query: str
        # output
        sql_query: str
        sql_explanation: str

    class Input(TypedDict):
        user_query: str

    class Output(TypedDict):
        sql_query: str
        sql_explanation: str

    generate_prompt = SystemMessage(
        "You are a helpful data analyst, who generates SQL queries for users based on their questions. Output only the SQL query."
    )

    def generate_sql(state: State) -> State:
        user_message = HumanMessage(state["user_query"])
        messages = [generate_prompt, *state["messages"], user_message]
        res = model_low_temp.invoke(messages)
        return {
            "sql_query": res.content,
            # update conversation history
            "messages": [user_message, res],
        }

    explain_prompt = SystemMessage(
        "You are a helpful data analyst, who explains SQL queries to users."
    )

    def explain_sql(state: State) -> State:
        messages = [
            explain_prompt,
            # contains the user's query and the SQL query from the previous step
            *state["messages"],
        ]
        res = model_high_temp.invoke(messages)
        return {
            "sql_explanation": res.content,
            # update conversation history
            "messages": res,
        }

    builder = StateGraph(State, input=Input, output=Output)
    builder.add_node("generate_sql", generate_sql)
    builder.add_node("explain_sql", explain_sql)
    builder.add_edge(START, "generate_sql")
    builder.add_edge("generate_sql", "explain_sql")
    builder.add_edge("explain_sql", END)

    graph = builder.compile()

    assert graph.get_graph().draw_mermaid() == snapshot

    assert graph.invoke(
        {"user_query": "What is the total sales for each product?"}
    ) == {
        "sql_query": "SELECT product_name, SUM(sales_amount) AS total_sales\nFROM sales\nGROUP BY product_name;",
        "sql_explanation": "This query will retrieve the total sales for each product by summing up the sales_amount column for each product and grouping the results by product_name.",
    }


def test_router(snapshot: SnapshotAssertion) -> None:
    from typing import Annotated, Literal, TypedDict

    from langchain_core.documents import Document
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.vectorstores.in_memory import InMemoryVectorStore
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    embeddings = OpenAIEmbeddings()
    # useful to generate SQL query
    model_low_temp = ChatOpenAI(temperature=0.1)
    # useful to generate natural language outputs
    model_high_temp = ChatOpenAI(temperature=0.7)

    class State(TypedDict):
        # to track conversation history
        messages: Annotated[list, add_messages]
        # input
        user_query: str
        # output
        domain: Literal["records", "insurance"]
        documents: list[Document]
        answer: str

    class Input(TypedDict):
        user_query: str

    class Output(TypedDict):
        documents: str
        answer: str

    # refer to chapter 2 on how to fill a vector store with documents
    medical_records_store = InMemoryVectorStore.from_documents([], embeddings)
    medical_records_retriever = medical_records_store.as_retriever()

    insurance_faqs_store = InMemoryVectorStore.from_documents([], embeddings)
    insurance_faqs_retriever = insurance_faqs_store.as_retriever()

    router_prompt = SystemMessage(
        """You need to decide which domain to route the user query to. You have two domains to choose from:
- records: contains medical records of the patient, such as diagnosis, treatment, and prescriptions.
- insurance: contains frequently asked questions about insurance policies, claims, and coverage.

Output only the domain name."""
    )

    def router_node(state: State) -> State:
        user_message = HumanMessage(state["user_query"])
        messages = [router_prompt, *state["messages"], user_message]
        res = model_low_temp.invoke(messages)
        return {
            "domain": res.content,
            # update conversation history
            "messages": [user_message, res],
        }

    def pick_retriever(
        state: State,
    ) -> Literal["retrieve_medical_records", "retrieve_insurance_faqs"]:
        if state["domain"] == "records":
            return "retrieve_medical_records"
        else:
            return "retrieve_insurance_faqs"

    def retrieve_medical_records(state: State) -> State:
        documents = medical_records_retriever.invoke(state["user_query"])
        return {
            "documents": documents,
        }

    def retrieve_insurance_faqs(state: State) -> State:
        documents = insurance_faqs_retriever.invoke(state["user_query"])
        return {
            "documents": documents,
        }

    medical_records_prompt = SystemMessage(
        "You are a helpful medical chatbot, who answers questions based on the patient's medical records, such as diagnosis, treatment, and prescriptions."
    )

    insurance_faqs_prompt = SystemMessage(
        "You are a helpful medical insurance chatbot, who answers frequently asked questions about insurance policies, claims, and coverage."
    )

    def generate_answer(state: State) -> State:
        if state["domain"] == "records":
            prompt = medical_records_prompt
        else:
            prompt = insurance_faqs_prompt
        messages = [
            prompt,
            *state["messages"],
            HumanMessage(f"Documents: {state["documents"]}"),
        ]
        res = model_high_temp.invoke(messages)
        return {
            "answer": res.content,
            # update conversation history
            "messages": res,
        }

    builder = StateGraph(State, input=Input, output=Output)
    builder.add_node("router", router_node)
    builder.add_node("retrieve_medical_records", retrieve_medical_records)
    builder.add_node("retrieve_insurance_faqs", retrieve_insurance_faqs)
    builder.add_node("generate_answer", generate_answer)
    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", pick_retriever)
    builder.add_edge("retrieve_medical_records", "generate_answer")
    builder.add_edge("retrieve_insurance_faqs", "generate_answer")
    builder.add_edge("generate_answer", END)

    graph = builder.compile()

    assert graph.get_graph().draw_mermaid() == snapshot

    print(
        [
            c
            for c in graph.stream(
                {"user_query": "Am I covered for COVID-19 treatment?"}
            )
        ]
    )


def test_agent(snapshot: SnapshotAssertion) -> None:
    import ast
    from typing import Annotated, TypedDict

    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    from langgraph.graph import START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition

    @tool
    def calculator(query: str) -> str:
        """A simple calculator tool. Input should be a mathematical expression."""
        return ast.literal_eval(query)

    search = DuckDuckGoSearchRun()
    tools = [search, calculator]
    model = ChatOpenAI(temperature=0.1).bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def model_node(state: State) -> State:
        res = model.invoke(state["messages"])
        return {"messages": res}

    builder = StateGraph(State)
    builder.add_node("model", model_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")

    graph = builder.compile()

    assert graph.get_graph().draw_mermaid() == snapshot

    assert [
        c
        for c in graph.stream(
            {
                "messages": [
                    HumanMessage(
                        "How old was the 30th president of the United States when he died?"
                    )
                ]
            }
        )
    ] == []


def test_agent_always_tool(snapshot: SnapshotAssertion) -> None:
    import ast
    from typing import Annotated, TypedDict
    from uuid import uuid4

    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_core.messages import AIMessage, HumanMessage, ToolCall
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    from langgraph.graph import START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition

    @tool
    def calculator(query: str) -> str:
        """A simple calculator tool. Input should be a mathematical expression."""
        return ast.literal_eval(query)

    search = DuckDuckGoSearchRun()
    tools = [search, calculator]
    model = ChatOpenAI(temperature=0.1).bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def model_node(state: State) -> State:
        res = model.invoke(state["messages"])
        return {"messages": res}

    def first_model(state: State) -> State:
        query = state["messages"][-1].content
        search_tool_call = ToolCall(
            name="duckduckgo_search", args={"query": query}, id=uuid4().hex
        )
        return {"messages": AIMessage(content="", tool_calls=[search_tool_call])}

    builder = StateGraph(State)
    builder.add_node("first_model", first_model)
    builder.add_node("model", model_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "first_model")
    builder.add_edge("first_model", "tools")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")

    graph = builder.compile()

    assert graph.get_graph().draw_mermaid() == snapshot

    assert [
        c
        for c in graph.stream(
            {
                "messages": [
                    HumanMessage(
                        "How old was the 30th president of the United States when he died?"
                    )
                ]
            }
        )
    ] == []


def test_agent_select_tools(snapshot: SnapshotAssertion) -> None:
    import ast
    from typing import Annotated, TypedDict

    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_core.documents import Document
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    from langchain_core.vectorstores.in_memory import InMemoryVectorStore
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    from langgraph.graph import START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition

    @tool
    def calculator(query: str) -> str:
        """A simple calculator tool. Input should be a mathematical expression."""
        return ast.literal_eval(query)

    search = DuckDuckGoSearchRun()
    tools = [search, calculator]
    embeddings = OpenAIEmbeddings()
    model = ChatOpenAI(temperature=0.1)
    tools_retriever = InMemoryVectorStore.from_documents(
        [Document(tool.description, metadata={"name": tool.name}) for tool in tools],
        embeddings,
    ).as_retriever()

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        selected_tools: list[str]

    def model_node(state: State) -> State:
        selected_tools = [
            tool for tool in tools if tool.name in state["selected_tools"]
        ]
        res = model.bind_tools(selected_tools).invoke(state["messages"])
        return {"messages": res}

    def select_tools(state: State) -> State:
        query = state["messages"][-1].content
        tool_docs = tools_retriever.invoke(query)
        return {"selected_tools": [doc.metadata["name"] for doc in tool_docs]}

    builder = StateGraph(State)
    builder.add_node("select_tools", select_tools)
    builder.add_node("model", model_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "select_tools")
    builder.add_edge("select_tools", "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")

    graph = builder.compile()

    assert graph.get_graph().draw_mermaid() == snapshot

    assert [
        c
        for c in graph.stream(
            {
                "messages": [
                    HumanMessage(
                        "How old was the 30th president of the United States when he died?"
                    )
                ]
            }
        )
    ] == []


def test_reflection(snapshot: SnapshotAssertion):
    from typing import Annotated, TypedDict

    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )
    from langchain_openai import ChatOpenAI

    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    model = ChatOpenAI()

    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    generate_prompt = SystemMessage(
        "You are an essay assistant tasked with writing excellent 5-paragraph essays."
        " Generate the best essay possible for the user's request."
        " If the user provides critique, respond with a revised version of your previous attempts."
    )

    def generate(state: State) -> State:
        answer = model.invoke([generate_prompt] + state["messages"])
        return {"messages": [answer]}

    reflection_prompt = SystemMessage(
        "You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission."
        " Provide detailed recommendations, including requests for length, depth, style, etc."
    )

    def reflect(state: State) -> State:
        # Invert the messages to get the LLM to reflect on its own output
        cls_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
        # First message is the original user request. We hold it the same for all nodes
        translated = [reflection_prompt, state["messages"][0]] + [
            cls_map[msg.__class__](content=msg.content) for msg in state["messages"][1:]
        ]
        answer = model.invoke(translated)
        # We treat the output of this as human feedback for the generator
        return {"messages": [HumanMessage(content=answer.content)]}

    def should_continue(state: State):
        if len(state["messages"]) > 6:
            # End after 3 iterations, each with 2 messages
            return END
        else:
            return "reflect"

    builder = StateGraph(State)
    builder.add_node("generate", generate)
    builder.add_node("reflect", reflect)
    builder.add_edge(START, "generate")
    builder.add_conditional_edges("generate", should_continue)
    builder.add_edge("reflect", "generate")

    graph = builder.compile()

    assert graph.get_graph().draw_mermaid() == snapshot
