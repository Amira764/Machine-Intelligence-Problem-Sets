# This file contains the options you need to modify to solve Question 2

# IMPORTANT NOTE:
# Make sure to comment your code explaining why you picked these values.
# Any uncommented code will lose points.

def question2_1():
    # Pick options that make the agent go for the nearby +1 reward using the risky path.
    # noise = 0.0: Keeps the agent on the risky path without distractions
    # discount_factor = 0.8: Focuses more on short-term rewards so it goes for +1 quickly
    # living_reward = -3.0: Big step penalty encourages the agent to act fast
    return {
        "noise": 0.0,
        "discount_factor": 0.8,
        "living_reward": -3.0
    }

def question2_2():
    # Pick options to make the agent take the long, safe path to the +1 reward.
    # noise = 0.3: Adds a little unpredictability but still avoids risks
    # discount_factor = 0.6: Looks for short-term rewards
    # living_reward = -2.0: Big step penalty encourages the agent to act fast
    return {
        "noise": 0.3,
        "discount_factor": 0.6,
        "living_reward": -2.0
    }

def question2_3():
    # Make the agent go for the faraway +10 reward using the risky path.
    # noise = 0.0: Keeps the agent on the risky path without distractions
    # discount_factor = 0.9: Focuses more on bigger rewards further away
    # living_reward = -2: Big step penalty encourages the agent to act fast
    return {
        "noise": 0.0,
        "discount_factor": 0.9,
        "living_reward": -2
    }

def question2_4():
    # Make the agent take the long, safe path to the faraway +10 reward.
    # noise = 0.2: Adds a little unpredictability but still avoids risks
    # discount_factor = 0.99: Focuses more on bigger rewards further away
    # living_reward = -0.05: A small penalty to discourage taking too long but doesn’t punish the safe route
    return {
        "noise": 0.2,
        "discount_factor": 0.99,
        "living_reward": -0.05
    }

def question2_5():
    # Set options so the agent avoids all terminal states and keeps playing forever.
    # noise = 0.2: Adds a little unpredictability but still avoids risks
    # discount_factor = 0.9: Focuses more on bigger rewards further away
    # living_reward = 2: A positive reward every step makes the agent keep going
    return {
        "noise": 0.2,
        "discount_factor": 0.9,
        "living_reward": 2
    }

def question2_6():
    # Make the agent end the episode as quickly as possible, even if penalties are involved.
    # noise = 0.4: Adds a lot of randomness to speed up decision-making.
    # discount_factor = 0.1: Focuses only on immediate rewards, pushing for quick termination
    # living_reward = -10: Big penalty for each step forces the agent to terminate fast
    return {
        "noise": 0.4,
        "discount_factor": 0.1,
        "living_reward": -10
    }
