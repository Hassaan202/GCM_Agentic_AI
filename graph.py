from typing import TypedDict, List, Union, Annotated, Sequence
import pandas as pd
import torch
from langchain_core.messages import (HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage,
                                     messages_to_dict,
                                     messages_from_dict)
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
import json
from predictor import predict_glucose


load_dotenv()
os.environ["GOOGLE_API_KEY"] = ""


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
    healthcare_provider_informed: bool


# --- LangGraph Nodes ---
def predict_node(state: AgentState):
    input_tensor = state["input_tensor"]
    patient_data = state["raw_patient_data"]

    predicted = predict_glucose(input_tensor, patient_data)

    return {"predicted_glucose": predicted}


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

    # print(f"Emergency: {emergency}")

    return {"glucose_level": level, "emergency": emergency}


def emergency_escalation_node(state: AgentState):
    user_input = input(f"Your glucose levels are critical ({state['predicted_glucose']} mg/dL) . Would you like to notify "
                       f"your healthcare provider? (yes/no): ")

    messages = [
        SystemMessage(content=f"If the user wants to notify, ONLY then call the `notify_healthcare_provider` tool with "
                              f"appropriate the user summary using the data: Predicted: {state["predicted_glucose"]},"
                              f" Glucose Level: {state["glucose_level"]}. Otherwise DONNOT call the tool."),
        HumanMessage(content=user_input)
    ]

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17", temperature=0.2).bind_tools(tools)
    response = llm.invoke(messages)

    return {"messages": messages + [response]}


def trend_node(state: AgentState):
    patient_data = state["raw_patient_data"].copy()
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
    prompt = f"""
        You are a helpful Diabetes Management Assistant. You will help the patient manage their glucose levels based on 
        CGM data. Be helpful and supportive. 
        Predicted glucose is {state["predicted_glucose"]:.1f} mg/dL which is {state["glucose_level"]}.
        Give clinical advice only and ask no follow up questions.
        """
    systemPrompt = SystemMessage(content=prompt)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17", temperature=0.4)
    response = llm.invoke([systemPrompt] + ["Glucose level: " + state["glucose_level"]])

    return {"advice": response.content}



# --- Router Nodes ---
def router_emergency(state: AgentState):
    return state["emergency"]


def router_info(state: AgentState):
    if not state["messages"]:
        return False

    last_message = state["messages"][-1]
    # print(f"DEBUG: Last message type: {type(last_message)}")
    # print(f"DEBUG: Has tool_calls attr: {hasattr(last_message, 'tool_calls')}")
    # print(f"DEBUG: tool_calls value: {getattr(last_message, 'tool_calls', 'NO ATTR')}")

    # Check if the last message has tool calls and they're not empty
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # print("DEBUG: Routing to Tools")
        return True
    # print("DEBUG: Routing to Trend")
    return False


# --- Tool Nodes ---
@tool
def notify_healthcare_provider(user_data_summary: str):
    """
    Notifies the healthcare provider about the medical situation of the person
    Args:
        user_data_summary: the summary of user glucose data and risk level
    """
    # TODO: Add some mechanism to inform the healthcare provider about the situation
    return f"The healthcare provider has been informed! Meanwhile perform some precautionary measures."



tools = [notify_healthcare_provider]


# --- Build LangGraph ---
graph = StateGraph(AgentState)


# Add Nodes
graph.add_node("Predict", predict_node)
graph.add_node("Classify", classify_risk)
graph.add_node("Trend", trend_node)
graph.add_node("Coach", coach_node)
graph.add_node("Emergency", emergency_escalation_node)
toolNode = ToolNode(tools=tools)
graph.add_node("Tools", toolNode)


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
graph.add_edge("Coach", END)
graph.add_conditional_edges(
    "Emergency",
    router_info,
    {
        True: "Tools",
        False: "Trend"
    }
)
graph.add_edge("Tools", "Trend")

# Compile the graph
graph = graph.compile()
