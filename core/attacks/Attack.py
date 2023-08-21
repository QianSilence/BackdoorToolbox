from abc import ABC, abstractmethod
class Attack(ABC):
    def __init__(self, name = None):
        self.name = name
        
    def get_attack_name(self):
        if self.name is not None:
            return self.name 
    
    @abstractmethod
    def attack():
        raise NotImplementedError
