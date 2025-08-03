from typing import TypedDict, List, Union, Annotated, Sequence
import pandas as pd
import torch
from langchain_core.messages import (HumanMessage, BaseMessage, SystemMessage, ToolMessage)
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

from predictor import predict_glucose
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from rag import get_retriever


memory = MemorySaver()

load_dotenv()


# --- Define state structure ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    patient_id: str
    input_tensor: torch.Tensor
    raw_patient_data: pd.DataFrame
    predicted_glucose: float
    glucose_level: str
    trend_note: str
    advice: str
    low_range: float
    high_range: float
    emergency: bool
    user_input: str
    rag_complete: bool
    emergency_response: str
    age: int
    gender: str
    diabetes_proficiency: str
    emergency_contact_number: str
    emergency_email: str


# --- LangGraph Nodes ---
def predict_node(state: AgentState):
    input_tensor = torch.tensor(state["input_tensor"])
    patient_data = pd.DataFrame(state["raw_patient_data"])

    predicted = predict_glucose(input_tensor, patient_data)

    return {
        "predicted_glucose": float(predicted)
    }


def classify_risk(state: AgentState):
    glucose = state["predicted_glucose"]
    emergency = False

    if glucose < state["low_range"]:
        level = "Low"
        if glucose < 54:
            emergency = True  # severe hypoglycemia (source: NIH)
        else:
            emergency = False

    elif glucose > state["high_range"]:
        level = "High"
        if glucose > 250:
            emergency = True  # severe hypoglycemia (source: NIH)
        else:
            emergency = False

    else:
        level = "Normal"

    return {"glucose_level": level, "emergency": emergency}


def emergency_escalation_node(state: AgentState):
    messages = [
        SystemMessage(content=f"First call the `get_user_input` tool to get the user input. Then, based on "
                              f"the user input from the last tool call, see if user and replies positively (like yes, please etc.) and wants to "
                              f"notify the healthcare provider, ONLY then call the `notify_healthcare_provider` tool with "
                              f"appropriate the user summary using the data: Predicted: {state["predicted_glucose"]},"
                              f" Glucose Level: {state["glucose_level"]} and the emergency email: {state["emergency_email"]}. "
                              f"If the user does not wants to inform the healthcare provider then DONNOT call the"
                              f" `notify_healthcare_provider` tool and "
                              f"reply with the user intent and condition summary."
                              f"Current message history: {state['messages']}"),
    ]

    user_text = state.get("user_input", "You are a helpful agent.")
    messages += [HumanMessage(content=user_text)]

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1).bind_tools(tools)
    response = llm.invoke(messages)

    if not hasattr(messages[-1], "tool_calls"):
        return {
            "messages": [response],
            "emergency_response": response.content
        }


    return {"messages": [response]}


def trend_node(state: AgentState):
    patient_data = pd.DataFrame(state["raw_patient_data"].copy())
    patient_data = patient_data.sort_values("time").reset_index(drop=True)

    recent_glucose = patient_data["glucose"].ffill().tail(6).values
    current_avg = recent_glucose.mean()
    predicted = state["predicted_glucose"]

    if predicted > current_avg + 10:
        trend = "rising"
    elif predicted < current_avg - 10:
        trend = "falling"
    else:
        trend = "stable"

    return {"trend_note": trend}


def coach_node(state: AgentState):
    if state.get("rag_complete", False):
        rag_results = ""
        for message in reversed(state["messages"]):
            if isinstance(message, ToolMessage):
                rag_results = message.content
                break

        # print(f"RAG:\n {rag_results}")

        prompt = f"""
        You are a helpful Diabetes Management Assistant. Based on the retrieved information and patient data, 
        provide concise yet comprehensive clinical advice for glucose management. Mention the sources of the
        recommendations and provide text with newlines.

        Patient Information:
        - Age: {state["age"]}
        - Gender: {state["gender"]}
        - Diabetes Proficiency: {state["diabetes_proficiency"]}
        - Predicted glucose: {state["predicted_glucose"]:.1f} mg/dL
        - Emergency: {state["emergency"]} 
        - Glucose level: {state["glucose_level"]}
        - Trend: {state.get("trend_note", "stable")}

        Retrieved Information:
        {rag_results}

        Please provide specific, actionable advice based on this information to manage the current glucose 
        level. Make sure to give correct advice if emergency hyper or hypoglycemia. Do not ask follow-up questions. 
        Also, the information provided should be understandable to the user based on their diabetes proficiency level. 
        """

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)
        response = llm.invoke([HumanMessage(content=prompt)])

        return {"advice": response.content, "rag_complete": False}

    else:
        # First time in coach node, initiate RAG
        prompt = f"""
        You are a helpful Diabetes Management Assistant. The patient has a predicted glucose level of 
        {state["predicted_glucose"]:.1f} mg/dL which is {state["glucose_level"]}.

        Please use the retriever tool to find relevant information about managing {state["glucose_level"].lower()} 
        glucose levels. If the patient has a low glucose level, include in the query the term hypoglycemia and 
        managing low blood sugar. if the level is high, include the term hyperglycemia and managing high blood glucose. 
        Make the query appropriate for retrieval from a vector store.
        """

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2).bind_tools(rag_tools)
        response = llm.invoke([HumanMessage(content=prompt)])

        return {"messages": [response], "rag_complete": True}



# --- Router Nodes ---
def router_emergency(state: AgentState):
    return state["emergency"]


def router_info(state: AgentState):
    if not state["messages"]:
        return False

    last_message = state["messages"][-1]

    # Check if the last message has tool calls and they're not empty
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # print("DEBUG: Routing to Tools")
        return True
    # print("DEBUG: Routing to Trend")
    return False


def router_coaching_rag(state: AgentState):
    """Check if the last message contains tool calls."""
    if state.get("rag_complete", False):
        return True

    if state.get('messages') and len(state['messages']) > 0:
        last_message = state['messages'][-1]
        if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
            return True

    return False


# --- Tool Nodes ---
@tool
def notify_healthcare_provider(user_data_summary: str, emergency_email: str):
    """
    Notifies the healthcare provider about the medical situation of the person
    Args:
        user_data_summary: the summary of user glucose data and risk level
        emergency_email: the email of a known person
    """

    return f"The healthcare provider has been informed! Meanwhile perform some precautionary measures."


@tool
def get_user_input(predicted_glucose: float, glucose_level: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    Gets the user input to decide if user wants to notify the healthcare provider or not
    Args:
        predicted_glucose: the predicted glucose number
        tool_call_id: tool call ID
        glucose_level: the predicted glucose level of the patient
    """
    # note that the interrupt node causes the entire node to be re-executed so stand-alone node or side effect stuff
    # placed after the interrupt call
    user_input = interrupt({"question": f"Your glucose levels are critical ({predicted_glucose:.2f} mg/dL -"
                                        f" {glucose_level})."
                                        f" Would you like to notify your healthcare provider?"})
    state_update = {
        "user_input": user_input,
        "messages": [ToolMessage("User response: " + user_input, tool_call_id=tool_call_id)]
    }
    return Command(update=state_update)


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Vector Store documents
    Args:
        query: the query searched for within the vector store
    """
    retriever = get_retriever()
    documents = retriever.invoke(query)

    if not documents:
        return "I found no relevant information in the documents."

    results = []
    for i, doc in enumerate(documents):
        results.append(f"Document {i + 1}:\n{doc.page_content}")

    return "\n\n".join(results)


tools = [notify_healthcare_provider, get_user_input]
rag_tools = [retriever_tool]


# --- Build LangGraph ---
graph = StateGraph(AgentState)


# Add Nodes
graph.add_node("Predict", predict_node)
graph.add_node("Classify", classify_risk)
graph.add_node("Trend", trend_node)
graph.add_node("Coach", coach_node)
graph.add_node("Emergency", emergency_escalation_node)
graph.add_node("Tools", ToolNode(tools=tools))
graph.add_node("RagTools", ToolNode(tools=rag_tools))


# Set Flow
graph.add_edge(START, "Predict")
graph.add_edge("Predict", "Classify")
graph.add_conditional_edges(
    "Classify",
    router_emergency,
    {
        True: "Emergency",
        False: "Trend"
    }
)
graph.add_edge("Trend", "Coach")
graph.add_conditional_edges(
    "Coach",
    router_coaching_rag,
    {
        True: "RagTools",
        False: END
    }
)
graph.add_edge("RagTools", "Coach")
graph.add_conditional_edges(
    "Emergency",
    router_info,
    {
        True: "Tools",
        False: "Trend"
    }
)
graph.add_edge("Tools", "Emergency")

# Compile the graph
graph = graph.compile(checkpointer=memory)

# Visualization
with open("mermaid_graph.txt", 'w') as f:
    f.write(graph.get_graph().draw_mermaid())