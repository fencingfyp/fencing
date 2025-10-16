class StaticTracker:
    def __init__(self):
        self.targets = {}

    def add_target(self, name, _, initial_positions):
        self.targets[name] = initial_positions

    def update_all(self, _):
        return self.targets.copy()