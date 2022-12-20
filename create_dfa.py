import random

def create_random_DFA(numb_of_states, numb_of_symbols):
    transitions= {}
    acceptance = []
    for s in range(numb_of_states):
        trans_from_s = {}
        #Each state is equiprobably set to be accepting or rejecting
        acceptance.append(random.randrange(2))
        #evenly choose another state from [i + 1; N ] and adds a random-labeled transition
        if s < numb_of_states - 1:
            s_prime = random.randrange(s + 1 , numb_of_states)
            a_start = random.randrange(numb_of_symbols)

            trans_from_s[a_start] = s_prime
        else:
            a_start = None
        for a in range(numb_of_symbols):
            if a != a_start:
                trans_from_s[a] = random.randrange(numb_of_states)
        transitions[s] = trans_from_s.copy()

    return transitions, acceptance

def from_transacc_2_dot(trans, acc, file_name):
    with open(file_name, "w") as f:
        f.write(
            "digraph MONA_DFA {\nrankdir = LR;\ncenter = true;\nsize = \"7.5,10.5\";\nedge [fontname = Courier];\nnode [height = .5, width = .5];\nnode [shape = doublecircle];")
        for i, rew in enumerate(acc):
                if rew == 1:
                    f.write(str(i) + ";")
        f.write("\nnode [shape = circle]; 0;\ninit [shape = plaintext, label = \"\"];\ninit -> 0;\n")

        for s in trans.keys():
            for a in trans[s].keys():
                s_prime = trans[s][a]
                f.write("{} -> {} [label=\"c{}\"];\n".format(s, s_prime, a))
        f.write("}\n")

class DFA:
        def __init__(self, trans=None, acc=None):
            self.transitions = trans
            self.acceptance = acc

        def create_random_DFA(self, numb_of_states, numb_of_symbols):
            transitions = {}
            acceptance = []
            for s in range(numb_of_states):
                trans_from_s = {}
                # Each state is equiprobably set to be accepting or rejecting
                acceptance.append(bool(random.randrange(2)))
                # evenly choose another state from [i + 1; N ] and adds a random-labeled transition
                if s < numb_of_states - 1:
                    s_prime = random.randrange(s + 1, numb_of_states)
                    a_start = str(random.randrange(numb_of_symbols))

                    trans_from_s[a_start] = s_prime
                else:
                    a_start = None
                for a in range(numb_of_symbols):
                    a = str(a)
                    if a != a_start:
                        trans_from_s[a] = random.randrange(numb_of_states)
                transitions[s] = trans_from_s.copy()

            self.transitions = transitions
            self.acceptance = acceptance
            self.alphabet = [str(a) for a in range(numb_of_symbols)]

        def accepts(self, string):
            if string == '':
                return self.acceptance[0]
            return self.accepts_from_state(0, string)

        def accepts_from_state(self, state, string):
            assert string != ''

            a = string[0]
            next_state = self.transitions[state][a]

            if len(string) == 1:
                return self.acceptance[next_state]

            return self.accepts_from_state(next_state, string[1:])

'''
t, a = create_random_DFA(5,2)
print("transitions:")
print(t)
print("acceptance")
print(a)
from_transacc_2_dot(t, a, "random_S=5_A=2.dot")
'''

def Tomita1():
    trans = {0: {0:1, 1:0}, 1:{0: 1, 1:1}}
    acc = [1, 0]
    return trans, acc

def Tomita2():
    trans = {0:{0:2, 1:1}, 1:{0:0, 1:2}, 2:{0:2, 1:2}}
    acc = [1, 0, 0]
    return trans, acc

def Tomita3():
    trans = {0:{0:0,1:1},1:{0:2,1:0}, 2:{0:3,1:4}, 3:{0:2,1:3}, 4:{0:4,1:4}}
    acc = [1, 1, 0, 1, 0]
    return trans, acc


def Tomita4():
    trans = {0:{0:1,1:0},1:{0:2,1:0}, 2:{0:3,1:0}, 3:{0:3,1:3}}
    acc = [1, 1, 1, 0]
    return trans, acc


def Tomita5():
    trans = {0:{0:1,1:3},1:{0:0,1:2}, 2:{0:3,1:1}, 3:{0:2,1:0}}
    acc = [1, 0, 0, 0]
    return trans, acc


def Tomita6():
    trans = {0:{0:2,1:1},1:{0:0,1:2}, 2:{0:1,1:0}}
    acc = [1, 0, 0]
    return trans, acc


def Tomita7():
    trans = {0:{0:0,1:1},1:{0:2,1:1}, 2:{0:2,1:3},  3:{0:4,1:3}, 4:{0:4,1:4}}
    acc = [1, 1, 1, 1, 0]
    return trans, acc
