import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import transacc2pythomata

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

class LSTMAutoma(nn.Module):

    def __init__(self, hidden_dim, vocab_size, tagset_size):
        super(LSTMAutoma, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(vocab_size, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):

        lstm_out, _ = self.lstm(sentence.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        return tag_space

    def predict(self, sentence):
        tag_space = self.forward(sentence)
        out = F.softmax(tag_space, dim=1)[-1]
        return out

sftmx = torch.nn.Softmax(dim=-1)

def sftmx_with_temp(x, temp):
    return sftmx(x/temp)

class ProbabilisticAutoma(nn.Module):
    def __init__(self, numb_of_actions, numb_of_states, numb_of_output):
        super(ProbabilisticAutoma, self).__init__()
        self.numb_of_actions = numb_of_actions
        self.alphabet = [str(i) for i in range(numb_of_actions)]
        self.numb_of_states = numb_of_states
        self.numb_of_outputs = numb_of_output
        self.output_values = torch.Tensor(list(range(numb_of_output)))
        self.activation = sftmx_with_temp
        self.trans_prob = torch.normal(0, 0.1, size=( numb_of_actions, numb_of_states, numb_of_states), requires_grad=True, device=device)
        self.fin_matrix = torch.normal(0, 0.1, size=( numb_of_states, numb_of_output), requires_grad=True, device=device)

    #input: sequence of actions (batch, length_seq)
    def forward(self, action_seq, temp, current_state= None):
        batch_size = action_seq.size()[0]
        length_seq = action_seq.size()[1]

        if current_state == None:
            s = torch.zeros((batch_size,self.numb_of_states)).to(device)
            #initial state is 0 for construction
            s[:,0] = 1.0
        else:
            s = current_state
        for i in range(length_seq):
            a = action_seq[:,i]
            s, r = self.step(s, a, temp)
        return r

    def step(self,state, action, temp):
        if type(action) == int:
            action= torch.IntTensor([action])
        trans_prob = self.activation(self.trans_prob, temp)
        fin_matrix = self.activation(self.fin_matrix, temp)

        selected_prob = torch.index_select(trans_prob, 0, action)
        next_state = torch.matmul(state.unsqueeze(dim=1), selected_prob)
        next_output = torch.matmul(next_state, fin_matrix)
        next_state = next_state.squeeze()
        next_output = next_output.squeeze()
        return next_state, next_output

    def net2dfa(self, min_temp):

        trans_prob = self.activation(self.trans_prob, min_temp)
        fin_matrix = self.activation(self.fin_matrix, min_temp)

        trans_prob = torch.argmax(trans_prob, dim= 2)
        fin_matrix = torch.argmax(fin_matrix, dim=1)

        #2transacc
        trans = {}
        for s in range(self.numb_of_states):
            trans[s] = {}
        acc = []
        for i, rew in enumerate(fin_matrix):
                if rew == 1:
                    acc.append(True)
                else:
                    acc.append(False)
        for a in range(trans_prob.size()[0]):
            for s, s_prime in enumerate(trans_prob[a]):
                    trans[s][str(a)] = s_prime.item()

        #print("###################à trans acc DFA")
        #print(trans)
        #print(acc)
        #2pythomata

        pyautomaton = transacc2pythomata(trans, acc, self.alphabet)
        #print("############ pythomata dfa:")
        #print(pyautomaton.__dict__)

        pyautomaton = pyautomaton.reachable()
        #print("############ pythomata REACHABLE DFA")
        #print(pyautomaton.__dict__)

        pyautomaton = pyautomaton.minimize()
        print("############ pythomata MINIMUM DFA")
        print(pyautomaton.__dict__)

        return pyautomaton


    def initFromDfa(self, reduced_dfa, final_states):
        with torch.no_grad():
            #zeroing transition probabilities
            for a in range(self.numb_of_actions):
                for s1 in range(self.numb_of_states):
                    for s2 in range(self.numb_of_states):
                        self.trans_prob[a, s1, s2] = 0.0

            #zeroing output matrix
            for s in range(self.numb_of_states):
                for r in range(self.numb_of_outputs):
                    self.fin_matrix[s,r] = 0.0


        #set the transition probabilities as the one in the dfa
        for s in reduced_dfa:
            for a in reduced_dfa[s]:
                with torch.no_grad():
                    self.trans_prob[a, s, reduced_dfa[s][a]] = 1.0

        #set final matrix
        for s in range(len(reduced_dfa.keys())):
            if s in final_states:
                with torch.no_grad():
                    self.fin_matrix[s, 1] = 1.0
            else:
                with torch.no_grad():
                    self.fin_matrix[s, 0] = 1.0







