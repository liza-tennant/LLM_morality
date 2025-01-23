#player.py
import random 

class Opponent: 
    def __init__(self, strategy, C_symbol, D_symbol, RN_stream):
        self.strategy = strategy
        self.C = C_symbol 
        self.D = D_symbol 
        self.RN = RN_stream

    def make_move(self, prev_move_opponent):
        if self.strategy == 'Random':
            #ßreturn random.choice([self.C, self.D])
            return self.RN.choice([self.C, self.D], 1).item()
            
        elif self.strategy == 'TFT':
            return prev_move_opponent

        elif self.strategy == 'AC':
            return self.C
        
        elif self.strategy == 'AD':
            return self.D

        elif self.strategy == 'LLM':
            pass #will generate opponent move outside of this class 