# test_env.py
from environment import OpsForgeEnv

env = OpsForgeEnv()
obs = env.reset()

print("Initial Observation:", obs)

from models import Action   # add this import

action = Action(
    ticket_id=obs.tickets[0].id,
    assign_engineers=1,
    spend_budget=10.0,
    priority=3
)

obs, reward, done, _ = env.step(action)

print("Next Observation:", obs)
print("Reward:", reward)