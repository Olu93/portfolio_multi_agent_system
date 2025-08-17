from abc import ABC, abstractmethod

class Destination(ABC):
    def __init__(self, name:str, url:str):
        self.name = name
        self.url = url

    @abstractmethod
    def store(self, data:dict):
        pass

    @abstractmethod
    def get(self, query:str):
        pass

    def build(self):
        return self

    def __str__(self):
        return f"{self.name} - {self.url}"
    
    def __repr__(self):
        return f"{self.name} - {self.url}"
    

class KafkaDestination(Destination):
    def __init__(self, name:str, url:str):
        super().__init__(name, url)

    def store(self, data:dict):
        pass
    
    def get(self, query:str):
        pass