
import absl.flags
import absl.app
from utils import set_seed, transacc2pythomata, dataset_from_dict, dataset_with_errors_from_dict
from create_dfa import Tomita1, Tomita2, Tomita3, Tomita4, Tomita5, Tomita6, Tomita7
from test_methods import test_neurosym
import os
import pickle
from Random_DFA import Random_DFA

#flags
absl.flags.DEFINE_string("dataset_dir", "datasets_Tomita/", "path to the datasets")
absl.flags.DEFINE_string("results_dir", "Results/", "path to save the results")
absl.flags.DEFINE_string("plot_dir", "Plots/", "path to save the plots")
absl.flags.DEFINE_string("ground_truth_dfa_dir", "DFA_ground_truth/", "path to save ground truth automatons")
absl.flags.DEFINE_string("predicted_dfa_dir", "DFA_predicted_nesy/", "path to save predicted automatons")
absl.flags.DEFINE_float("error_rate", 0.0, "portion of errated labels in the training dataset")
absl.flags.DEFINE_string("experiment", "tomita", "which experiment to execute, choose one in the list ['tomita', 'random_dfa']")

FLAGS = absl.flags.FLAGS

############################################# EXPERIMENTS ######################################################################
def main(argv):
    if not os.path.isdir(FLAGS.results_dir):
        os.makedirs(FLAGS.results_dir)
    if not os.path.isdir(FLAGS.plot_dir):
        os.makedirs(FLAGS.plot_dir)
    if not os.path.isdir(FLAGS.ground_truth_dfa_dir):
        os.makedirs(FLAGS.ground_truth_dfa_dir)
    if not os.path.isdir(FLAGS.predicted_dfa_dir):
        os.makedirs(FLAGS.predicted_dfa_dir)

    if FLAGS.experiment == "tomita":
        #################### TOMITA LANG EXPERIMENTS
        num_exp = 5
        tomita_langs = [(Tomita1()), (Tomita2()) , (Tomita3()) , (Tomita4()) , (Tomita5()) , (Tomita6()) , (Tomita7()) ]

        for i,tomita in enumerate(tomita_langs):

            lang_name = "Tomita"+str(i+1)

            alphabet = ["c" + str(j) for j in range(2)]

            trans, accept = tomita[0], tomita[1]

            #2pythomata
            dfa = transacc2pythomata(trans, accept, alphabet)
            try:
                dfa.to_graphviz().render("DFA_ground_truth/"+lang_name+".dot")
            except:
                print("Not able to render automa")

            numb_states = 30
            numb_of_symbols = 2

            ##### SYMBOLIC traces dataset
            if FLAGS.error_rate == 0.0:
                train_traces, train_acceptance = dataset_from_dict(FLAGS.dataset_dir+"train_ds_"+lang_name+".pckl")
            else:
                train_traces, train_acceptance = dataset_with_errors_from_dict(
                    FLAGS.dataset_dir + "train_ds_" + lang_name + ".pckl", FLAGS.error_rate)
            dev_traces, dev_acceptance = dataset_from_dict(FLAGS.dataset_dir+"dev_ds_"+lang_name+".pckl")
            test_traces, test_acceptance = dataset_from_dict(FLAGS.dataset_dir+"test_ds_"+lang_name+".pckl")
            symbolic_dataset = (train_traces, dev_traces, test_traces, train_acceptance, dev_acceptance, test_acceptance)
            for exp in range(num_exp):
                set_seed(9+exp)
                print("###################### NEW TEST ###########################")
                print("formula = {},\texperiment = {}".format(lang_name, exp))
                test_neurosym( lang_name, symbolic_dataset, numb_states, numb_of_symbols, exp, FLAGS.results_dir, FLAGS.plot_dir, FLAGS.ground_truth_dfa_dir, FLAGS.predicted_dfa_dir)
                #test_dfa_inductor(lang_name, alphabet, (test_traces, test_acceptance))

    elif FLAGS.experiment == "random_dfa":
        #################### RANDOM EXPERIMENTS
        num_exp = 10

        STATE_SIZES = [10, 20, 30]
        ACTION_SIZES = [2, 3]

        for S in STATE_SIZES:
            for A in ACTION_SIZES:
                alphabet = ""
                for ch in range(A):
                    alphabet += str(ch)
                for N in range(5):
                    if A == 3 and N < 2:
                        continue

                    lang_name = "random_dfa_S={}_A={}_N={}".format(S, A, N)
                    alphabet = [str(j) for j in range(A)]

                    # random DFA
                    with open("random_DFAs/" + lang_name + ".pckl", "rb") as f:
                        dfa = pickle.load(f)
                    trans, accept = dfa.transitions, dfa.acceptance

                    # 2pythomata
                    dfa = transacc2pythomata(trans, accept, alphabet)
                    try:
                        dfa.to_graphviz().render("DFA_ground_truth/" + lang_name + ".dot")
                    except:
                        print("Not able to render automa")

                    if S < 30:
                        numb_states = 100
                    else:
                        numb_states = 200

                    ##### SYMBOLIC traces dataset
                    if FLAGS.error_rate == 0.0:
                        train_traces, train_acceptance = dataset_from_dict(
                            FLAGS.dataset_dir + "train_ds_" + lang_name + ".pckl")
                    else:
                        train_traces, train_acceptance = dataset_with_errors_from_dict(
                            FLAGS.dataset_dir + "train_ds_" + lang_name + ".pckl", FLAGS.error_rate)
                    dev_traces, dev_acceptance = dataset_from_dict(FLAGS.dataset_dir + "dev_ds_" + lang_name + ".pckl")
                    test_traces, test_acceptance = dataset_from_dict(
                        FLAGS.dataset_dir + "test_ds_" + lang_name + ".pckl")
                    symbolic_dataset = (
                    train_traces, dev_traces, test_traces, train_acceptance, dev_acceptance, test_acceptance)

                    for exp in range(num_exp):
                        # set_seed
                        set_seed(9 + exp)
                        print("###################### NEW TEST ###########################")
                        print("formula = {},\texperiment = {}".format(lang_name, exp))

                        test_neurosym(lang_name, symbolic_dataset, numb_states, A, exp, FLAGS.results_dir, FLAGS.plot_dir,
                                      FLAGS.ground_truth_dfa_dir, FLAGS.predicted_dfa_dir)

                    # test_dfa_inductor("random_DFA_S={}_A={}_N={}".format(S,A,N), alphabet, (test_traces, test_acceptance), results_dir=FLAGS.LOG_DIR )


if __name__ == '__main__':
    absl.app.run(main)


