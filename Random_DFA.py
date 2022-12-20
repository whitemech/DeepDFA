import random

class Random_DFA:
  def __init__(self, numb_of_states, numb_of_symbols):
      transitions= {}
      acceptance = []
      for s in range(numb_of_states):
          trans_from_s = {}
          #Each state is equiprobably set to be accepting or rejecting
          acceptance.append(bool(random.randrange(2)))
          #evenly choose another state from [i + 1; N ] and adds a random-labeled transition
          if s < numb_of_states - 1:
              s_prime = random.randrange(s + 1 , numb_of_states)
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
      self.alphabet = ""
      for a in range(numb_of_symbols):
        self.alphabet += str(a)

  def init_from_transacc(self, trans, acc):
      self.transitions = trans
      self.acceptance = acc

  def accepts(self, string):
    if string == '':
      return self.acceptance[0]
    return self.accepts_from_state(0, string)

  def accepts_from_state(self, state,string):
    assert string != ''

    a = string[0]
    next_state = self.transitions[state][a]

    if len(string) == 1:
      return self.acceptance[next_state]

    return self.accepts_from_state(next_state, string[1:])