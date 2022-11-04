def decayed_learning_rate(step, learning_rate, decay_steps=1000, decay_rate=0.75):
    # careful here! step includes batch steps in the tf framework
    return learning_rate * decay_rate ** (step / decay_steps)
