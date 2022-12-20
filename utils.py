import torch
import random
from numpy.random import RandomState
import os
import numpy as np
from copy import deepcopy
from pythomata import SymbolicAutomaton, PropositionalInterpretation, SimpleDFA
import pickle
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def set_seed(seed: int) -> RandomState:
    """ Method to set seed across runs to ensure reproducibility.
    It fixes seed for single-gpu machines.
    Args:
        seed (int): Seed to fix reproducibility. It should different for
            each run
    Returns:
        RandomState: fixed random state to initialize dataset iterators
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state


def eval_learnt_DFA_acceptance_no_batch(automa, dataset, automa_implementation='logic_circuit', temp=1.0, alphabet=None):

    #automa implementation =
    #   - 'dfa' use the discretized probabilistic automaton #TODO
    #   - 'logic_circuit'
    #   - 'lstm' use the lstm model in automa

    total = 0
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for i in range(len(dataset[0])):
            sym = dataset[0][i]#.unsqueeze(0)#.to(device)
            label = dataset[1][i]

            #secondo modo: usando il circuito logico continuo
            # terzo modo: usando la lstm
            if automa_implementation == 'logic_circuit' or automa_implementation == 'lstm':
                sym = sym.unsqueeze(0)#<-- questa serve quando non traino batch
                pred_acceptace = automa(sym, temp)
                output = torch.argmax(pred_acceptace).item()
            elif automa_implementation == 'dfa':
                sym_trace = tensor2symtrace(sym, alphabet)
                output = int(automa.accepts(sym_trace))
            else:
                print("INVALID AUTOMA IMPLEMENTATION: ", automa_implementation)

            total += 1


            correct += int(output==label)


            accuracy = 100. * correct/(float)(total)

    return accuracy

def eval_learnt_DFA_acceptance(automa, dataset, automa_implementation='logic_circuit', temp=1.0, alphabet=None):

    #automa implementation =
    #   - 'dfa' use the discretized probabilistic automaton #TODO
    #   - 'logic_circuit'
    #   - 'lstm' use the lstm model in automa

    total = 0
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for i in range(len(dataset[0])):
            sym = dataset[0][i].to(device)#.unsqueeze(0)
            if automa_implementation != "dfa":
                label = dataset[1][i].to(device)
            else:
                label = dataset[1][i]

            #secondo modo: usando il circuito logico continuo
            # terzo modo: usando la lstm
            if automa_implementation == 'logic_circuit' or automa_implementation == 'lstm':
                pred_acceptace = automa(sym, temp)
                output = torch.argmax(pred_acceptace, dim= 1)
            elif automa_implementation == 'dfa':

                output = torch.zeros((sym.size()[0]), dtype=torch.int)
                #print("sym_size:", sym.size())
                for k in range(sym.size()[0]):

                    sym_trace = tensor2string(sym[k])
                    #print("sym_trace", sym_trace)
                    output[k] = int(automa.accepts(sym_trace))
                    #print("out:", int(automa.accepts(sym_trace)))

            else:
                print("INVALID AUTOMA IMPLEMENTATION: ", automa_implementation)
            total += output.size()[0]


            correct += sum(output==label).item()


            accuracy = 100. * correct/(float)(total)

    return accuracy

def tensor2symtrace(tensor, alphabet):
    truth_value = {}

    for c in alphabet:
        truth_value[c] = False

    symtrace = []
    tensor=tensor.tolist()

    for sym in tensor:
        #sym = tensor[0,i]
        #print("sym0",sym)
        step = truth_value.copy()
        step["c"+str(sym)] = True
        symtrace.append(step)

    return symtrace

def tensor2string(tensor):
    string = ""
    tensor=tensor.tolist()

    for sym in tensor:
        string += str(sym)

    return string

#questo è fatto ad hoc per i dfa fatti con mona
def dot2pythomata(dot_file_name, action_alphabet):#, dfa_file_name):

        fake_action = "(~"+action_alphabet[0]
        for sym in action_alphabet[1:]:
            fake_action+=" & ~"+sym
        fake_action+=") | ("+action_alphabet[0]
        for sym in action_alphabet[1:]:
            fake_action+=" & "+sym
        fake_action+=")"
        #print("fake_action: ", fake_action)

        file1 = open(dot_file_name, 'r')
        Lines = file1.readlines()

        count = 0
        states = set()

        for line in Lines:
            count += 1
            if count >= 11:
                if line.strip()[0] == '}':
                    break
                action = line.strip().split('"')[1]
                states.add(line.strip().split(" ")[0])
            else:
                if "doublecircle" in line.strip():
                    final_states = line.strip().split(';')[1:-1]

        automaton = SymbolicAutomaton()
        state_dict = dict()
        state_dict['0'] = 0
        for state in states:
            if state == '0':
                continue
            state_dict[state] = automaton.create_state()

        final_state_list = []
        for state in final_states:
            state = int(state)
            state = str(state)
            final_state_list.append(state)

        for state in final_state_list:
            automaton.set_accepting_state(state_dict[state], True)

        count = 0
        for line in Lines:
            count += 1
            if count >= 11:
                if line.strip()[0] == '}':
                    break
                action = line.strip().split('"')[1]
                #print("action : ", action)
                action_label = action
                for sym in action_alphabet:
                    if sym != action:
                        action_label += " & ~"+sym
                #print("action_label: ", action_label)
                init_state = line.strip().split(" ")[0]
                final_state = line.strip().split(" ")[2]
                automaton.add_transition((state_dict[init_state], action_label, state_dict[final_state]))
                automaton.add_transition((state_dict[init_state], fake_action, state_dict[init_state]))

        automaton.set_initial_state(state_dict['0'])

        #with open(dfa_file_name, 'wb') as outp:
        #    pickle.dump(automaton, outp, pickle.HIGHEST_PROTOCOL)
        return automaton

def from_dfainductor_2_transacc(picklepath):
    with open(picklepath, "rb") as f:
        dfa = pickle.load(f)
    print("dfa_ind:")
    print(dfa.__dict__)
    trans = {}
    acc = []
    dfa = dfa.__dict__["_states"]

    for s in dfa:
        trans[s.id_] = {}
        acc.append(int(s.is_accepting()))
        for action in s.children.keys():
            action_int = int(action)
            trans[s.id_][action_int] = s.children[action].id_
    print("trans acc")
    print(trans)
    print(acc)
    return trans, acc

def transacc2pythomata(trans, acc, action_alphabet):
    accepting_states = set()
    for i in range(len(acc)):
        if acc[i]:
            accepting_states.add(i)

    automaton = SimpleDFA.from_transitions(0, accepting_states, trans)

    return automaton


def dataset_from_dict(path):
    with open(path, "rb") as f:
        ds_dict = pickle.load(f)

    strings = []
    labels = []

    sorted_ds_dict = sorted(list(ds_dict.items()), key=lambda x: len(x[0]))
    len0 = 0
    batch_size = 64

    for string,label in sorted_ds_dict:
        if string=='':
            continue
        l = len(string)
        if l > len0:
            len0 = l
            strings.append(torch.zeros((0,len(string)),dtype=torch.int))
            labels.append([])
        #else:
        strings[-1] = torch.cat((strings[-1], torch.zeros((1, len(string)),dtype=torch.int)))
        labels[-1].append(label)

        for i, char in enumerate(string):
            strings[-1][-1][i] = int(char)

    labels = [torch.LongTensor(label) for label in labels]
    #print("-----statistics------")
    #print([s.size()[0] for s in strings])
    return strings, labels

def dataset_with_errors_from_dict(path, error_rate):
    with open(path, "rb") as f:
        ds_dict = pickle.load(f)

    strings = []
    labels = []

    sorted_ds_dict = sorted(list(ds_dict.items()), key=lambda x: len(x[0]))

    if sorted_ds_dict[0][0] == '':
        sorted_ds_dict = sorted_ds_dict[1:]

    len_ds = len(sorted_ds_dict)

    n_errors = round(error_rate*len_ds)
    errors = random.sample(list(range(len_ds)), n_errors)

    len0 = 0


    for i in range(len_ds):
        string, label = sorted_ds_dict[i]
        if i in errors:
            label = not label
        if string=='':
            continue
        l = len(string)
        if l > len0:
            len0 = l
            strings.append(torch.zeros((0,len(string)),dtype=torch.int))
            labels.append([])
        #else:
        strings[-1] = torch.cat((strings[-1], torch.zeros((1, len(string)),dtype=torch.int)))


        labels[-1].append(label)

        for i, char in enumerate(string):
            strings[-1][-1][i] = int(char)

    labels = [torch.LongTensor(label) for label in labels]
    return strings, labels


def abadingo_dataset_from_dict(input_file, output_file, alphabet):
    with open(input_file, "rb") as f:
        ds_dict = pickle.load(f)

    sorted_ds_dict = sorted(list(ds_dict.items()), key=lambda x: len(x[0]))


    if sorted_ds_dict[0][0] == '':
        len_ds = len(sorted_ds_dict) -1
    else:
        len_ds = len(sorted_ds_dict)

    n_symbols = len(alphabet)

    f = open(output_file, "w")

    f.write("{} {}\n".format(len_ds, n_symbols))

    for string,label in sorted_ds_dict:
        if string=='':
            continue
        f.write("{} {}".format(int(label), len(string)))
        for char in string:
            f.write(" {}".format(char))
        f.write("\n")

def abadingo_dataset_with_errors_from_dict(input_file, output_file, alphabet, error_rate):
    with open(input_file, "rb") as f:
        ds_dict = pickle.load(f)

    sorted_ds_dict = sorted(list(ds_dict.items()), key=lambda x: len(x[0]))

    if sorted_ds_dict[0][0] == '':
        sorted_ds_dict = sorted_ds_dict[1:]

    len_ds = len(sorted_ds_dict)

    n_errors = round(error_rate*len_ds)
    errors = random.sample(list(range(len_ds)), n_errors)

    n_symbols = len(alphabet)

    f = open(output_file, "w")

    f.write("{} {}\n".format(len_ds, n_symbols))

    for i in range(len_ds):
        string, label = sorted_ds_dict[i]
        if string=='':
            continue
        if i in errors:
            #print("add error")
            label = not label
        f.write("{} {}".format(int(label), len(string)))
        for char in string:
            f.write(" {}".format(char))
        f.write("\n")
