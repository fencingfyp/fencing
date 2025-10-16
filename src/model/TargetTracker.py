import abc

class TargetTracker(abc.ABC):
    @abc.abstractmethod
    def add_target(self, name, frame, initial_positions):
        pass
    
    @abc.abstractmethod
    def update_all(self, frame):
        pass