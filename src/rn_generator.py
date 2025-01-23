from numpy.random import SeedSequence, default_rng

class my_RN_generator: 
    """ this class takes in a single master_seed (defined before each set of runs) and creates 5 (independent) streams of random numbers. 
    for players 1&2; then, for each player, 11 streams of numbers to be used in the experiments
    
    It allows us to generate 10 random numbers for every player: 
    [0] 'ppo_rn': random seed to be used in TRL's PPOConfig, 
    [1] 'intiial_state_rn': to be used in the generate_inital_state() function, 
    [2] 'playerM_rn': stream to be used by player M in player.make_move(), 
    [3] 'playerO_rn': to be used by player O in player.make_move(), specificallly when player O is using the Random strategy, 
    """

    def __init__(self, master_seed):
        self.master_seed = master_seed
        self.CD_symbols_rn = None
        self.ppo_rn = None
        self.initial_state_rn = None
        self.IPD_pompt_rn1 = None
        self.IPD_pompt_rn2 = None
        self.playerO_rn = None
        #self.playerM_rn = None



    def generate(self):
        ss = SeedSequence(self.master_seed)
        child_seeds = ss.spawn(6)

        self.CD_symbols_rn = default_rng(child_seeds[0])
        self.ppo_rn = default_rng(child_seeds[1])
        self.initial_state_rn = default_rng(child_seeds[2])
        self.IPD_pompt_rn1 = default_rng(child_seeds[3]) 
        self.IPD_pompt_rn2 = default_rng(child_seeds[4]) 
        self.playerO_rn = default_rng(child_seeds[5])
        #self.playerM_rn = default_rng(child_seeds[3])