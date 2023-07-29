from abc import ABC, abstractmethod
class Attack(ABC):
    def __init__(self, name):
        self.name = name
        self.poisoned_model = None
    def get_attack_name(self):
        if self.name is not None:
            return self.name 
    def get_poisoned_model(self):
        if self.poisoned_model is not None:
            return self.poisoned_model 
        
    @abstractmethod
    def attack():
        pass
