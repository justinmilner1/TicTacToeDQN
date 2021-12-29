import copy
import math
import random
import logging
import numpy as np
import tensorflow as tf
import game
from abc import abstractmethod
from dqn import DeepQNetworkModel
from memory_buffers import ExperienceReplayMemory
from math import inf as infinity


class Player:
    """
    Base class for all player types
    """
    name = None
    player_id = None

    def __init__(self):
        pass

    def shutdown(self):
        pass

    def add_to_memory(self, add_this):
        pass

    def save(self, filename):
        pass

    @abstractmethod
    def select_cell(self, board, **kwargs):
        pass

    @abstractmethod
    def learn(self, **kwargs):
        pass


class Human(Player):
    """
    This player type allow a human player to play the game
    """
    def select_cell(self, board, **kwargs):
        cell = input("Select cell to fill:\n012\n345\n678\ncell number: ")
        return cell

    def learn(self, **kwargs):
        pass


class Drunk(Player):
    """
    Drunk player always selects a random valid move
    """
    def select_cell(self, board, **kwargs):
        available_cells = np.where(board == 0)[0]
        choice = random.choice(available_cells)
        return choice

    def learn(self, **kwargs):
        pass


class Novice(Player):
    """
    A more sophisticated bot, which follows the following strategy:
    1) If it already has 2-in-a-row, capture the required cell for 3
    2) If not, and if the opponent has 2-in-a-row, capture the required cell to prevent him, from winning
    3) Else, select a random vacant cell
    """
    def find_two_of_three(self, board, which_player_id):
        cell = None
        winning_options = [[0, 1, 2], [3, 4, 5], [6, 7, 8],
                           [0, 3, 6], [1, 4, 7], [2, 5, 8],
                           [0, 4, 8], [2, 4, 6]]
        random.shuffle(winning_options)
        for seq in winning_options:
            s = board[seq[0]] + board[seq[1]] + board[seq[2]]
            if s == 2 * which_player_id:
                a = np.array([board[seq[0]], board[seq[1]], board[seq[2]]])
                c = np.where(a == 0)[0][0]
                cell = seq[c]
                break
        return cell

    def select_cell(self, board, **kwargs):
        cell = self.find_two_of_three(board,self.player_id)
        if cell is None:
            cell = self.find_two_of_three(board,-self.player_id)
        if cell is None:
            available_cells = np.where(board == 0)[0]
            cell = random.choice(available_cells)
        return cell

    def learn(self, **kwargs):
        pass

class Minimax(Player):
    HUMAN = None
    COMP = None
    def select_cell(self, board, **kwargs):

        self.COMP = 1#self.player_id          #comp is the player calling the minimax function
        self.HUMAN = -1#self.player_id * -1    #human is the opponent
        #print("Playing as: ", self.player_id)
        board = np.reshape(board, (3, 3))   #convert board from 1d to 2d
        #print(board)
        best = []
        if len(self.empty_cells(board)) == 9:
            best.append(random.choice([0, 1, 2]))
            best.append(random.choice([0, 1, 2]))
        else:
            best = self.minimax(board, len(self.empty_cells(board)), self.player_id)
        best_move = (best[0] * 3) + best[1]
        # print("best: ", best)
        # print("best_move: ", best_move)
        # print("------------------------")
        return best_move


    def minimax(self, state, depth, player):
        '''
        COMP = +1 -->
        HUMAN = -1

        '''
        # print("minimax called. State: ", state, ", depth: ", depth, ", player: ", player)
        if player == self.COMP:
            best = [-1, -1, -infinity]  #minimizing
        else:
            best = [-1, -1, +infinity]  #maximizing

        if depth == 0 or self.game_over(state):
            score = self.evaluate(state, player)
            return [-1, -1, score]

        for cell in self.empty_cells(state):
            # print("went into loop, cell: ", cell)
            x, y = cell[0], cell[1]
            state[x][y] = player
            score = self.minimax(state, depth - 1, player*-1)
            state[x][y] = 0
            score[0], score[1] = x, y

            if player == self.COMP:  #in case when the move does not improve the outcome (such as ties), no move gets assigned, causing error
                if score[2] > best[2]: #i changed these from '>' to '>='
                    #print("Best updated from: ", best, " to: ", score)
                    best = score  # min value
            else:
                if score[2] < best[2]:
                    best = score  # max value


        return best

    def empty_cells(self, state):
        """
        Each empty cell will be added into cells' list
        :param state: the state of the current board
        :return: a list of empty cells
        """
        cells = []

        for x, row in enumerate(state):
            for y, cell in enumerate(row):
                if cell == 0:
                    cells.append([x, y])

        return cells

    def wins(self, state, player):
        """
        This function tests if a specific player wins. Possibilities:
        * Three rows    [X X X] or [O O O]
        * Three cols    [X X X] or [O O O]
        * Two diagonals [X X X] or [O O O]
        :param state: the state of the current board
        :param player: a human or a computer
        :return: True if the player wins
        """
        win_state = [
            [state[0][0], state[0][1], state[0][2]],
            [state[1][0], state[1][1], state[1][2]],
            [state[2][0], state[2][1], state[2][2]],
            [state[0][0], state[1][0], state[2][0]],
            [state[0][1], state[1][1], state[2][1]],
            [state[0][2], state[1][2], state[2][2]],
            [state[0][0], state[1][1], state[2][2]],
            [state[2][0], state[1][1], state[0][2]],
        ]
        if [player, player, player] in win_state:
            return True
        else:
            return False

    def game_over(self, state):
        """
        This function test if the human or computer wins
        :param state: the state of the current board
        :return: True if the human or computer wins
        """
        return self.wins(state, -1) or self.wins(state, 1)


    def evaluate(self, state, player):
        """
        Function to heuristic evaluation of state.
        :param state: the state of the current board
        :return: +1 if the computer wins; -1 if the human wins; 0 draw
        """
        if self.wins(state, self.COMP):
            score = self.COMP
        elif self.wins(state, self.HUMAN):
            score = self.HUMAN
        else:
            score = 0

        return score

    def learn(self, **kwargs):
        pass


class QPlayer(Player):
    """
    A reinforcement learning agent, based on Double Deep Q Network model
    This class holds two Q-Networks: `qnn` is the learning network, `q_target` is the semi-constant network
    """
    def __init__(self, session, hidden_layers_size, gamma, learning_batch_size, batches_to_q_target_switch, tau, memory_size,
                 maximize_entropy=False, var_scope_name=None):
        """
        :param session: a tf.Session instance
        :param hidden_layers_size: an array of integers, specifying the number of layers of the network and their size
        :param gamma: the Q-Learning discount factor
        :param learning_batch_size: training batch size
        :param batches_to_q_target_switch: after how many batches (trainings) should the Q-network be copied to Q-Target
        :param tau: a number between 0 and 1, determining how to combine the network and Q-Target when copying is performed
        :param memory_size: size of the memory buffer used to keep the training set
        :param maximize_entropy: boolean, should the network try to maximize entropy over direct future rewards
        :param var_scope_name: the variable scope to use for the player
        """
        layers_size = [item for sublist in [[9],hidden_layers_size,[9]] for item in sublist]
        self.session = session
        self.model = DeepQNetworkModel(session=self.session,
                                       layers_size=layers_size,
                                       memory=ExperienceReplayMemory(memory_size),
                                       default_batch_size=learning_batch_size,
                                       gamma=gamma,
                                       double_dqn=True,
                                       learning_procedures_to_q_target_switch=batches_to_q_target_switch,
                                       tau=tau,
                                       maximize_entropy=maximize_entropy,
                                       var_scope_name=var_scope_name)
        self.session.run(tf.compat.v1.global_variables_initializer())
        super(QPlayer, self).__init__()

    def select_cell(self, board, **kwargs):
        return self.model.act(board, epsilon=kwargs['epsilon'])

    def learn(self, **kwargs):
        return self.model.learn(learning_rate=kwargs['learning_rate'])

    def add_to_memory(self, add_this):
        state = self.player_id * add_this['state']
        next_state = self.player_id * add_this['next_state']
        self.model.add_to_memory(state=state, action=add_this['action'], reward=add_this['reward'],
                                 next_state=next_state, is_terminal_state=add_this['game_over'])

    def save(self, filename):
        saver = tf.compat.v1.train.Saver()
        saver.save(self.session, filename)

    def restore(self, filename):
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.session, filename)

    def shutdown(self):
        try:
            self.session.close()
        except Exception as e:
            logging.warning('Failed to close session', e)
