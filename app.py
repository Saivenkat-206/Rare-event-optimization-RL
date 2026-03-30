import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

# Assuming your files are named:
# dqn.py (contains QNetwork)
# env_risk.py (contains Risk-Aware GridEnv)
# env_naive.py (contains Naive GridEnv)
from dqn import QNetwork
from env import GridEnv as RiskEnv
from env2 import GridEnv as NaiveEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="RL Tail-Risk Demo", layout="wide")
st.title("⚡ Power Grid Maintenance: Naive vs. Risk-Aware DQN")

# ---- Helper Functions ----
@st.cache_resource
def load_model(agent_type):
    """Loads and caches the model to prevent reloading on every click."""
    model = QNetwork(3, 2).to(device)
    path = "NaiveDQN.pth" if agent_type == "Naive" else "FixedDQN.pth"
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file {path} not found. Please ensure it exists.")
        return None

def run_simulation(env, model):
    state, _ = env.reset()
    history = {
        "prep": [],
        "budget": [],
        "failures": [],
        "actions": [],
        "total_reward": 0
    }

    for t in range(200):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = torch.argmax(model(s)).item()

        next_state, reward, terminated, truncated, info = env.step(action)

        history["prep"].append(state[0])
        history["budget"].append(state[2])
        history["actions"].append(action)
        history["total_reward"] += reward

        if info.get("failure"):
            history["failures"].append(t)

        state = next_state
        if terminated or truncated:
            break

    return history

def plot_behavior(history, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["prep"], label="Preparedness (Health)", color="#2ecc71", linewidth=2)
    ax.plot(history["budget"], label="Budget (Resources)", color="#3498db", linewidth=2)
    
    # Mark failures with red dashed lines
    for i, f in enumerate(history["failures"]):
        label = "Failure Event" if i == 0 else ""
        ax.axvline(f, color="#e74c3c", linestyle="--", alpha=0.7, label=label)

    ax.set_ylim(-0.1, 1.1)
    ax.set_title(title)
    ax.set_xlabel("Time Steps")
    ax.legend(loc="lower left")
    return fig

# ---- Sidebar Controls ----
st.sidebar.header("Simulation Settings")
run_button = st.sidebar.button("Run Comparison", use_container_width=True)
st.sidebar.info("The Naive agent only learns from actual crashes. The Risk-Aware agent receives continuous 'anxiety' penalties for low preparedness.")

# ---- Main Dashboard ----
if run_button:
    # 1. Load Models
    naive_model = load_model("Naive")
    risk_model = load_model("Risk-Aware")

    if naive_model and risk_model:
        col1, col2 = st.columns(2)

        # 2. Run Simulations
        naive_data = run_simulation(NaiveEnv(), naive_model)
        risk_data = run_simulation(RiskEnv(), risk_model)

        # 3. Display Results
        with col1:
            st.subheader("Naive Agent")
            st.pyplot(plot_behavior(naive_data, "Naive Policy: Reactive"))
            
            st.metric("Total Reward", f"{naive_data['total_reward']:.2f}")
            st.metric("Crashes/Failures", len(naive_data['failures']))
            st.metric("Maintenance Count", sum(naive_data['actions']))

        with col2:
            st.subheader("Risk-Aware Agent")
            st.pyplot(plot_behavior(risk_data, "Risk-Aware Policy: Proactive"))
            
            st.metric("Total Reward", f"{risk_data['total_reward']:.2f}")
            st.metric("Crashes/Failures", len(risk_data['failures']))
            st.metric("Maintenance Count", sum(risk_data['actions']))
            
        st.divider()
        st.caption("Note: The Risk-Aware agent's total reward might be lower numerically due to the continuous penalties, but its survival rate is typically much higher.")
else:
    st.write("Click the **Run Comparison** button in the sidebar to start the simulation.")
