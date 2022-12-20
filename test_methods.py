from Neural_DFA_identification import Neural_DFA_identification
from utils import from_dfainductor_2_transacc, eval_learnt_DFA_acceptance
from create_dfa import DFA

def test_neurosym(formula_name, symbolic_dataset, numb_states, numb_symbols, num_exp, log_dir, plot_dir, gt_dfa_dir, pred_dfa_dir, epochs=50):
    ltl_ground =Neural_DFA_identification( formula_name, symbolic_dataset, numb_states, numb_symbols,  log_dir, plot_dir, gt_dfa_dir, pred_dfa_dir, num_exp=num_exp)

    ltl_ground.train_DFA(27,epochs)

def test_dfa_inductor(formula, alphabet, symbolic_test_dataset, log_dir="Results/"):
    path = "DFA_predicted_sat/dfainductor_"+formula+".pckl"
    print(path)
    #2transacc
    trans, acc = from_dfainductor_2_transacc(path)

    trans2 = {}
    for s in trans.keys():
        trans2[s] = {}
        for a in trans[s].keys():
            trans2[s][str(a)] = trans[s][a]
    print(trans2)
    dfa = DFA(trans2, acc)

    with open("DFA_predicted_sat/dfainductor_"+formula+"_min_num_states", "w") as f:
        f.write("{}\n".format(len(acc)))
    accuracy = eval_learnt_DFA_acceptance(dfa, symbolic_test_dataset, automa_implementation="dfa",  alphabet=alphabet)

    with open(log_dir+"dfainductor_"+formula+"_test_acc", "w") as f:
        f.write(str(accuracy))