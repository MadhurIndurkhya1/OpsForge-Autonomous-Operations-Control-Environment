# 🚀 OpsForge: Autonomous Operations Control Environment

## 🧠 Overview

OpsForge is a realistic simulation environment designed to model
**real-world operational decision-making under constraints**.

Unlike traditional AI tasks that focus on isolated predictions, OpsForge
evaluates how an agent: - prioritizes multiple tasks\
- manages limited resources\
- handles time pressure and uncertainty

------------------------------------------------------------------------

## 🎯 Problem Statement

Modern systems (SaaS, fintech, logistics) constantly deal with: - bug
reports\
- fraud alerts\
- customer complaints\
- service requests

These must be handled with: - limited engineers\
- limited budget\
- strict deadlines

👉 OpsForge simulates this complexity in a controlled environment.

------------------------------------------------------------------------

## ⚙️ Core Working

At each step:

1.  **Observation**
    -   Active tickets (severity, deadline, value)
    -   Available resources (engineers, budget, time)
2.  **Decision (Action)**
    -   Select a ticket
    -   Assign engineers
    -   Allocate budget
    -   Set priority
3.  **Environment Response**
    -   Ticket resolution (success/failure)
    -   Deadline updates
    -   Reward calculation
4.  **Loop continues**
    -   New tickets may appear
    -   System evolves dynamically

------------------------------------------------------------------------

## 🔥 Key Features

-   ✅ Real-world inspired environment\
-   ✅ Resource constraints (engineers, budget, time)\
-   ✅ Deadline-based urgency\
-   ✅ Stochastic outcomes (uncertainty)\
-   ✅ Multi-step decision-making\
-   ✅ Reward with partial scoring

------------------------------------------------------------------------

## 🧩 Task Levels

  Level    Description
  -------- --------------------------
  Easy     Basic prioritization
  Medium   Resource allocation
  Hard     Full multi-step strategy

------------------------------------------------------------------------

## 🏗️ Project Structure

    OpsForge/
    │── environment.py
    │── models.py
    │── reward.py
    │── tasks.py
    │── inference.py
    │── test_env.py
    │── requirements.txt
    │── Dockerfile
    │── openenv.yaml
    │── README.md

------------------------------------------------------------------------

## 🧪 Installation

``` bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Running the Simulation

### 🔹 Heuristic Mode (Recommended)

``` bash
python inference.py --no-llm
```

------------------------------------------------------------------------

### 🔹 Different Tasks

``` bash
python inference.py --task easy --no-llm
python inference.py --task medium --no-llm
python inference.py --task hard --no-llm
```

------------------------------------------------------------------------

## 🤖 Agent Design

### Heuristic Agent

-   prioritizes urgent tasks (deadline-aware)
-   balances resource usage
-   avoids early exhaustion

### LLM Agent (Optional)

-   uses external AI models
-   requires API setup

------------------------------------------------------------------------

## 📊 Evaluation Metrics

-   Total Reward\
-   Tickets Resolved\
-   Missed Deadlines\
-   Resolve Rate\
-   Final Grade (PASS / FAIL)

------------------------------------------------------------------------

## 🌍 Real-World Applications

OpsForge models real-world systems such as:

-   SaaS incident management\
-   Fraud detection pipelines\
-   Customer support prioritization\
-   Logistics and delivery optimization

👉 It can act as a **decision-support simulation system**.

------------------------------------------------------------------------

## 🧠 Key Insights

-   Greedy strategies fail in dynamic environments\
-   Urgency (deadlines) must be prioritized\
-   Resource conservation is critical\
-   Long-term planning improves performance

------------------------------------------------------------------------

## 🏁 Conclusion

OpsForge demonstrates how AI can operate as a **decision-making system
in dynamic environments**, rather than solving isolated problems.

------------------------------------------------------------------------

## 👨‍💻 Author

Developed as an AI environment project focused on real-world decision
systems.
