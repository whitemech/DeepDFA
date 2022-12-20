import torch
import pickle
from DeepAutoma import LSTMAutoma, ProbabilisticAutoma
import math
from statistics import mean

from utils import eval_learnt_DFA_acceptance
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
print("device = ", device)
import time

class Neural_DFA_identification:
    def __init__(self, formula_name, symbolic_dataset, numb_of_states, numb_of_symbols, log_dir, plot_dir, gt_dfa_dir, pred_dfa_dir, automa_implementation = 'logic_circuit', lstm_output= "acceptance", num_exp=0):
        self.ltl_formula_string = formula_name
        self.log_dir = log_dir
        self.plot_dir = plot_dir
        self.gt_dfa_dir = gt_dfa_dir
        self.pred_dfa_dir = pred_dfa_dir
        self.exp_num=num_exp

        self.numb_of_symbols = numb_of_symbols
        self.numb_of_states = numb_of_states

        self.alphabet = ["c"+str(i) for i in range(self.numb_of_symbols) ]

        #################### networks
        self.hidden_dim = numb_of_states
        self.automa_implementation = automa_implementation


        if self.automa_implementation == 'lstm':
            if lstm_output== "states":
                self.deepAutoma = LSTMAutoma(self.hidden_dim, self.numb_of_symbols, self.numb_of_states)
            elif lstm_output == "acceptance":
                self.deepAutoma = LSTMAutoma(self.hidden_dim, self.numb_of_symbols, 2)
            else:
                print("INVALID LSTM OUTPUT. Choose between 'states' and 'acceptance'")
        elif self.automa_implementation == 'logic_circuit':
            self.deepAutoma = ProbabilisticAutoma(self.numb_of_symbols, self.numb_of_states, 2)
        else:
            print("INVALID AUTOMA IMPLEMENTATION. Choose between 'lstm' and 'logic_circuit'")

        #dataset
        self.train_traces, self.dev_traces, self.test_traces, self.train_acceptance_tr, self.dev_acceptance_tr, self.test_acceptance_tr = symbolic_dataset

        self.temperature = 1.0

        #traces
        self.positive_traces = set()
        self.negative_traces = set()


    def eval_learnt_DFA(self, automa_implementation, temp, mode="dev"):
        if mode=="dev":
            if automa_implementation == 'dfa':
                train_acc = eval_learnt_DFA_acceptance(self.dfa, (self.train_traces, self.train_acceptance_tr),
                                                       automa_implementation, temp, alphabet=self.alphabet)
                test_acc = eval_learnt_DFA_acceptance(self.dfa, (self.dev_traces, self.dev_acceptance_tr),
                                                       automa_implementation, temp, alphabet=self.alphabet)
            else:
                train_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.train_traces, self.train_acceptance_tr), automa_implementation, temp)
                test_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.dev_traces, self.dev_acceptance_tr), automa_implementation, temp)
        else:
            if automa_implementation == 'dfa':
                train_acc = eval_learnt_DFA_acceptance(self.dfa, (self.train_traces, self.train_acceptance_tr),
                                                       automa_implementation, temp, alphabet=self.alphabet)
                test_acc = eval_learnt_DFA_acceptance(self.dfa, (self.test_traces, self.test_acceptance_tr),
                                                      automa_implementation, temp, alphabet=self.alphabet)
            else:
                train_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.train_traces, self.train_acceptance_tr),
                                                       automa_implementation, temp)
                test_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.test_traces, self.test_acceptance_tr),
                                                      automa_implementation, temp)
        return train_acc, test_acc


    def train_DFA(self, batch_size, num_of_epochs, decay = 0.999, freezed=False):
        #def get_lr(optim):
        #    for param_group in optim.param_groups:
        #        return param_group['lr']
        decay=1.0
        tot_size = len(self.train_traces)
        mean_loss = 1000000

        train_file = open(self.log_dir+self.ltl_formula_string+"_train_acc_NS_exp"+str(self.exp_num), 'w')
        dev_file = open(self.log_dir+self.ltl_formula_string+"_dev_acc_NS_exp"+str(self.exp_num), 'w')

        train_file_dfa = open(self.log_dir+self.ltl_formula_string+"_train_acc_dfa_NS_exp"+str(self.exp_num), 'w')
        dev_file_dfa = open(self.log_dir+self.ltl_formula_string+"_dev_acc_dfa_NS_exp"+str(self.exp_num), 'w')
        test_file_dfa = open(self.log_dir+self.ltl_formula_string+"_test_acc_dfa_NS_exp"+str(self.exp_num), 'w')
        loss_file = open(self.log_dir+self.ltl_formula_string+"_loss_dfa_NS_exp"+str(self.exp_num), 'w')


        cross_entr = torch.nn.CrossEntropyLoss()
        print("_____________training the DFA_____________")
        print("training on {} sequences using {} automaton states".format(tot_size, self.numb_of_states))

        params = [self.deepAutoma.trans_prob] + [self.deepAutoma.fin_matrix]
        optimizer = torch.optim.Adam(params, lr=0.01)
        #sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-04)

        min_temp = 0.00001
        self.temperature =1.0

        if freezed:
            self.temperature = min_temp

        start_time = time.time()
        epoch= -1
        while True:
            epoch+=1
            print("epoch: ", epoch)
            losses = []
            for i in range(len(self.train_traces)):

                batch_trace_dataset = self.train_traces[i].to(device)
                batch_acceptance = self.train_acceptance_tr[i].to(device)
                optimizer.zero_grad()

                predictions= self.deepAutoma(batch_trace_dataset, self.temperature)

                loss = cross_entr(predictions, batch_acceptance)

                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            train_accuracy, test_accuracy = self.eval_learnt_DFA(automa_implementation='logic_circuit', temp=self.temperature)
            mean_loss_new = mean(losses)
            print("SEQUENCE CLASSIFICATION (LOGIC CIRCUIT): train accuracy : {}\ttest accuracy : {}\tloss : {}".format(train_accuracy, test_accuracy, mean_loss_new))

            train_file.write("{}\n".format(train_accuracy))
            dev_file.write("{}\n".format(test_accuracy))
            train_accuracy, test_accuracy = self.eval_learnt_DFA(automa_implementation='logic_circuit', temp=min_temp)
            print("SEQUENCE CLASSIFICATION (DFA): train accuracy : {}\ttest accuracy : {}".format(train_accuracy, test_accuracy))

            train_file_dfa.write("{}\n".format(train_accuracy))
            dev_file_dfa.write("{}\n".format(test_accuracy))
            loss_file.write("{}\n".format(mean(losses)))

            #decrease temperature
            if freezed:
                self.temperature = min_temp
            else:
                self.temperature = max([min_temp, self.temperature*decay])

            #sheduler.step(mean_loss_new)
            #print("lr: ", get_lr(optimizer))
            if mean_loss_new < 0.318 and abs(mean_loss_new - mean_loss) < 0.0001:
                break
            if epoch > 200 and abs(mean_loss_new - mean_loss) < 0.0001:
                break
            mean_loss = mean_loss_new

        ######################## net2dfa
        #save the minimized dfa
        self.dfa = self.deepAutoma.net2dfa( min_temp)
        ex_time =  time.time() - start_time

        with open(self.pred_dfa_dir+self.ltl_formula_string+"_exp"+str(self.exp_num)+".ex_time", "w") as f:
            f.write("{}\n".format(ex_time))

        #print it
        try:
            self.dfa.to_graphviz().render(self.pred_dfa_dir+self.ltl_formula_string+"_exp"+str(self.exp_num)+"_minimized.dot")
        except:
            print("Not able to render automa")
        with open(self.pred_dfa_dir+self.ltl_formula_string, 'wb') as outp:
            pickle.dump(self.dfa, outp, pickle.HIGHEST_PROTOCOL)

        with open(self.pred_dfa_dir+self.ltl_formula_string+"_exp"+str(self.exp_num)+"_min_num_states", "w") as f:
            f.write(str(len(self.dfa._states)))

        #LAST TEST using the DFA on the TEST set
        train_accuracy, test_accuracy = self.eval_learnt_DFA(automa_implementation='dfa', temp=min_temp, mode="test")
        print("FINAL SEQUENCE CLASSIFICATION ON TEST SET: {}".format(test_accuracy))

        test_file_dfa.write("{}\n".format(test_accuracy))


    def train_lstm(self, num_of_epochs):
        train_file = open(self.log_dir+self.ltl_formula_string+"_train_acc_DL_exp"+str(self.exp_num), 'w')
        test_clss_file = open(self.log_dir+self.ltl_formula_string+"_test_clss_acc_DL_exp"+str(self.exp_num), 'w')
        test_aut_file = open(self.log_dir+self.ltl_formula_string+"_test_aut_acc_DL_exp"+str(self.exp_num), 'w')
        test_hard_file = open(self.log_dir+self.ltl_formula_string+"_test_hard_acc_DL_exp"+str(self.exp_num), 'w')
        print("_____________training classifier+lstm_____________")
        loss_crit = torch.nn.CrossEntropyLoss()


        params =  self.deepAutoma.parameters()
        optimizer = torch.optim.Adam(params=params, lr=0.001)
        batch_size = 64
        tot_size = len(self.train_traces)
        self.deepAutoma.to(device)

        for epoch in range(num_of_epochs):
            print("epoch: ", epoch)
            for b in range(math.floor(tot_size/batch_size)):
                start = batch_size*b
                end = min(batch_size*(b+1), tot_size)
                batch_trace_dataset = self.train_traces[start:end]
                batch_acceptance = self.train_acceptance_tr[start:end]
                optimizer.zero_grad()
                losses = torch.zeros(0 ).to(device)

                for i in range(len(batch_trace_dataset)):
                    target = batch_acceptance[i]
                    target = torch.LongTensor([target]).to(device)
                    sym_sequence = batch_trace_dataset[i].to(device)
                    acceptance = self.deepAutoma.predict(sym_sequence)
                    loss = loss_crit(acceptance.unsqueeze(0), target)
                    losses = torch.cat((losses, loss.unsqueeze(dim=0)), 0)

                loss = losses.mean()
                loss.backward()
                optimizer.step()
            train_accuracy, test_accuracy_clss, test_accuracy_aut, test_accuracy_hard = self.eval_automa_acceptance(automa_implementation='lstm')
            print("__________________________train accuracy : {}\ttest accuracy(clss) : {}\ttest accuracy(aut) : {}\ttest accuracy(hard) : {}".format(train_accuracy,
                                                                                                 test_accuracy_clss, test_accuracy_aut, test_accuracy_hard))


            train_file.write("{}\n".format(train_accuracy))
            test_clss_file.write("{}\n".format(test_accuracy_clss))
            test_aut_file.write("{}\n".format(test_accuracy_aut))
            test_hard_file.write("{}\n".format(test_accuracy_hard))