import torch 
import transformers
from transformers import AutoModel, AutoTokenizer, AdamW
from sklearn.metrics import f1_score, precision_score, recall_score 
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn as nn 
import numpy as np 
import random
import utils
import time
import sys


class MTLModel(nn.Module):
    """ base model """
    def __init__(self, bert_config = 'vinai/bertweet-base', exp_hidden_size = 64, cls_hidden_size = 64, num_classes = 6):
        super(MTLModel, self).__init__()
        self.base_model = AutoModel.from_pretrained(bert_config)
        self.exp_hidden_size = exp_hidden_size
        self.cls_hidden_size = cls_hidden_size 
        self.num_classes = num_classes 

        class ExpLayers(nn.Module):
            def __init__(self, input_size, hidden_size):
                super(ExpLayers, self).__init__()
                self.exp_gru = nn.GRU(input_size, hidden_size, bidirectional = True)
                self.exp_linear = nn.Linear(2*hidden_size, 1, bias = True)
                self.exp_out = nn.Sigmoid()
            
            def forward(self, input):
                return self.exp_out(self.exp_linear(self.exp_gru(input)[0]))
        
        class ClsLayers(nn.Module):
            def __init__(self, hidden_size, output_size):
                super(ClsLayers, self).__init__()
                self.cls_dropout = nn.Dropout(0.1)
                self.cls_linear = nn.Linear(hidden_size, output_size, bias = True)

            def forward(self, input):
                return self.cls_linear(self.cls_dropout(input))
        
        self.exp_layers = ExpLayers(self.base_model.config.hidden_size, self.exp_hidden_size)
        self.cls_layers = ClsLayers(self.base_model.config.hidden_size, self.num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask = attention_mask)
        exp_out = self.exp_layers(outputs[0]).squeeze() * attention_mask
        cls_out = self.cls_layers(outputs[1])

        return cls_out, exp_out 
    

class MTLTrainer:
    def __init__(self, args, data = None, cls_labels= None, exp_labels = None):
        self.args = args 
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_config)
        
        if data is not None:
            self.data = np.array([self.tokenizer.cls_token + " " + x + " " + self.tokenizer.sep_token for x in data])
            self.exp_labels = exp_labels 
            self.num_classes = len(set(cls_labels))

            #conver data to tensor
            self.cls_labels = torch.tensor(cls_labels, dtype = torch.long)
            self.tokenized_data, self.input_ids, self.attention_masks, self.tokenized_data_slides = \
                            utils.tokenize_text(self.tokenizer, self.data, self.tokenizer.pad_token)
            self.exp_labels_mapping = utils.map_exp_labels(self.tokenizer, self.data, exp_labels)
            self.tokenized_data, self.input_ids = np.array(self.tokenized_data,dtype=object), torch.tensor(self.input_ids, dtype = torch.long)
            self.attention_masks, self.tokenized_data_slides= torch.tensor(self.attention_masks, dtype = torch.long), np.array(self.tokenized_data_slides,dtype=object)
            self.exp_labels_mapping = torch.tensor(self.exp_labels_mapping, dtype = torch.long)
        
    def fit(self, input_ids, attention_masks, cls_labels, exp_labels, cls_criterion, exp_criterion, exp_weight, batch_size):
        """fit model"""
        self.model.train()
        total_loss = 0 
        epoch_indices = random.sample([i for i in range(len(input_ids))], len(input_ids))
        batch_num = 0 
        
        for batch_start in range(0, len(input_ids), batch_size):
            batch_end = min(batch_start+ batch_size, len(input_ids))
            batch_indices = epoch_indices[batch_start:batch_end]
            batch_input_ids = input_ids[batch_indices]
            batch_attention_masks = attention_masks[batch_indices]
            batch_cls_labels = cls_labels[batch_indices]
            batch_exp_labels = exp_labels[batch_indices]
            cls_preds, exp_preds = self.model(batch_input_ids.to(self.args.device), 
                            batch_attention_masks.to(self.args.device))
        
            cls_loss = cls_criterion(cls_preds, batch_cls_labels.to(self.args.device)).mean(dim = -1).sum()
            exp_loss = exp_criterion(exp_preds, batch_exp_labels.to(self.args.device).float()).mean(dim = -1).sum()
            loss = cls_loss + exp_weight * exp_loss
            self.optimizer.zero_grad()
            total_loss += loss.item()
            batch_num += 1
            loss.backward()
            # clip the norm of the gradients to 1, to prevent exploding
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # update parameters
            self.optimizer.step()

            # update learning rate
            self.scheduler.step()
        
        return total_loss/batch_num
    
    def predict(self, input_ids, attention_masks, cls_labels=None, exp_labels=None, cls_criterion=None, exp_criterion=None, exp_weight=0.07, batch_size=128):
        """make prediction"""
        self.model.eval()

        cls_preds = []
        exp_preds = []
        exp_probs = []
        cls_probs = []
        batch_start = 0
        total_loss = 0 
        batch_num = 0
        with torch.no_grad():
            for batch_start in range(0, len(input_ids), batch_size):
                
                batch_end = min(batch_start + batch_size, len(input_ids))
                batch_input_ids = input_ids[batch_start: batch_end]
                batch_attention_masks = attention_masks[batch_start: batch_end]
                
                cls_outs, exp_outs = self.model(batch_input_ids.to(self.args.device), 
                                batch_attention_masks.to(self.args.device))
    
                if cls_labels is not None:
                    batch_cls_labels = cls_labels[batch_start: batch_end]
                    batch_exp_labels = exp_labels[batch_start: batch_end]
                    cls_loss = cls_criterion(cls_outs, batch_cls_labels.to(self.args.device)).mean(dim = -1).sum()
                    exp_loss = exp_criterion(exp_outs, batch_exp_labels.to(self.args.device).float()).mean(dim = -1).sum()
                    loss = cls_loss + exp_weight * exp_loss
                    total_loss += loss.item()
                
                cls_outs = nn.Softmax(dim = -1)(cls_outs)
                cls_outs = cls_outs.max(dim = -1)
                cls_pred_labels = cls_outs.indices.cpu()
                cls_pred_probs = cls_outs.values.cpu()
                exp_outs = torch.round(exp_outs).long().cpu()  
                
                exp_probs += exp_outs.cpu()
                exp_preds += exp_outs
                cls_preds += cls_pred_labels
                cls_probs += cls_pred_probs
                batch_num += 1  

        return cls_preds, exp_preds, cls_probs, exp_probs, total_loss/batch_num

    def eval(self, train_indices = None, valid_indices = None, test_indices= None):
        """train, eval, test"""
        train_data, valid_data, test_data = self.data[train_indices], self.data[valid_indices], self.data[test_indices]
        train_input_ids, valid_input_ids, test_input_ids = self.input_ids[train_indices], self.input_ids[valid_indices], self.input_ids[test_indices]
        train_attention_masks, valid_attention_masks, test_attention_masks = self.attention_masks[train_indices], self.attention_masks[valid_indices], self.attention_masks[test_indices]
        train_tokenized_data_slides, valid_tokenized_data_slides, test_tokenized_data_slides = self.tokenized_data_slides[train_indices], self.tokenized_data_slides[valid_indices], self.tokenized_data_slides[test_indices]
        train_exp_labels, valid_exp_labels, test_exp_labels = self.exp_labels_mapping[train_indices], self.exp_labels_mapping[valid_indices], self.exp_labels_mapping[test_indices]
        train_cls_labels, valid_cls_labels, test_cls_labels = self.cls_labels[train_indices], self.cls_labels[valid_indices], self.cls_labels[test_indices]

        # initialize base model
        self.model = MTLModel(bert_config = self.args.model_config, cls_hidden_size = self.args.cls_hidden_size, 
                    exp_hidden_size = self.args.exp_hidden_size, num_classes = self.num_classes)
        self.model.to(self.args.device)

        n_batches = int(np.ceil(len(train_indices)/self.args.train_batch_size))
        # print("#batches for each epoch: {}".format(n_batches))
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr, eps=1e-8)
        total_step = n_batches * self.args.n_epochs

        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                num_warmup_steps=0,
                                                                num_training_steps=total_step)
        
        cls_criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
        exp_criterion = utils.resampling_rebalanced_crossentropy(seq_reduction = 'none')
        
        best_epoch = {}
        best_model_state_dict = None
        best_loss = float('inf')
        exp_weight = self.args.exp_weight
        
        for epoch in range(self.args.n_epochs):
            begin_time = time.time()
            #fit model
            train_loss = self.fit(train_input_ids, train_attention_masks, train_cls_labels, train_exp_labels, cls_criterion, 
                            exp_criterion, exp_weight, self.args.train_batch_size)
        
            #evaluate on validation set
            cls_pred_labels, exp_pred_labels, _, _, valid_loss = self.predict(valid_input_ids, valid_attention_masks, 
                    valid_cls_labels, valid_exp_labels, cls_criterion, exp_criterion, exp_weight, self.args.test_batch_size)
            exp_true = utils.max_pooling(valid_exp_labels, valid_tokenized_data_slides, valid_data)
            exp_pred = utils.max_pooling(exp_pred_labels, valid_tokenized_data_slides, valid_data)
            

            exp_f1 = np.mean([f1_score(y_true, y_pred) for y_true, y_pred in zip(exp_true, exp_pred) if sum(y_true)!=0])
        
            cls_f1 = f1_score(valid_cls_labels, cls_pred_labels, average='macro')
            
            print("Epoch: %d, train_loss: %.3f, valid_loss: %.3f, valid_cls_f1: %.3f, time: %.3f" %(epoch, 
                train_loss, valid_loss, cls_f1, time.time()-begin_time))                
            if best_loss > valid_loss:
                best_loss = valid_loss 
                best_epoch = {'epoch': epoch, 'valid_loss': valid_loss}
                best_model_state_dict = OrderedDict({k: v.cpu() for k, v in self.model.state_dict().items()})
            
            if epoch - best_epoch['epoch'] > self.args.patience:
                break

        print("+ Training ends!")
        
        print("+ Load best model {}---valid_loss: {}".format(best_epoch['epoch'], best_epoch['valid_loss']))
        self.model.load_state_dict(best_model_state_dict)
        self.model.to(self.args.device)
        
        # extract predicted results on train data
        train_cls_pred_labels, train_exp_pred_labels, _, _, _ = self.predict(train_input_ids, train_attention_masks, 
                    train_cls_labels, train_exp_labels, cls_criterion, exp_criterion, exp_weight, self.args.test_batch_size)
        train_exp_pred_labels = utils.max_pooling(train_exp_pred_labels, train_tokenized_data_slides, train_data)
       
        # extract predicted exp on train data 
        train_exp_pred_data = []
        new_train_labels = []
        for data, exp, cls_pred, cls_true in zip(train_data, train_exp_pred_labels, train_cls_pred_labels, train_cls_labels):
            if (self.args.full_train == False) and (cls_pred != cls_true):
                continue 
            new_train_labels.append(cls_true)
            text = data.split(" ")
            train_exp_pred_data.append(' '.join([text[i] if exp[i] == 1 else "*" for i in range(1, len(text)-1)]).strip())
        

        # extract predicted exp on valid data 
        valid_cls_pred_labels, valid_exp_pred_labels, _, _, _ = self.predict(valid_input_ids, valid_attention_masks, 
                    valid_cls_labels, valid_exp_labels, cls_criterion, exp_criterion, exp_weight, self.args.test_batch_size)
        valid_exp_pred_labels = utils.max_pooling(exp_pred_labels, valid_tokenized_data_slides, valid_data)
        
        valid_exp_pred_data =[]
      
        for data, exp in zip(valid_data, valid_exp_pred_labels):
            text = data.split(' ')
            valid_exp_pred_data.append(' '.join([text[i] if exp[i] == 1 else "*" for i in range(1, len(text)-1)]).strip())
        
        # extract predicted exp on test data
        test_cls_pred_labels, test_exp_pred_labels, _, _, _ = self.predict(test_input_ids, test_attention_masks, 
                        test_cls_labels, test_exp_labels, cls_criterion, exp_criterion, exp_weight, self.args.test_batch_size)
        test_exp_true_labels = utils.max_pooling(test_exp_labels, test_tokenized_data_slides, test_data)
        test_exp_pred_labels = utils.max_pooling(test_exp_pred_labels, test_tokenized_data_slides, test_data)
        
        test_exp_pred_data = []
        
        for data, exp in zip(test_data, test_exp_pred_labels):
            text = data.split(' ')
            test_exp_pred_data.append(' '.join([text[i] if exp[i] == 1 else "*" for i in range(1, len(text)-1)]).strip())
        
        test_cls_f1 = f1_score(test_cls_labels, test_cls_pred_labels, average = 'macro')
        test_exp_f1 = np.mean([f1_score(y_true, y_pred) for y_true, y_pred in zip(test_exp_true_labels, test_exp_pred_labels)])
        test_exp_p = np.mean([precision_score(y_true, y_pred) for y_true, y_pred in zip(test_exp_true_labels, test_exp_pred_labels)])
        test_exp_r = np.mean([recall_score(y_true, y_pred) for y_true, y_pred in zip(test_exp_true_labels, test_exp_pred_labels)])

        print("++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++%s" % "Test results: ")
        print("++ CLS F1: %.4f" %(test_cls_f1))
        print("++ Exp F1: %.4f, exp_p: %.4f, exp_r: %.4f" %(test_exp_f1, test_exp_p, test_exp_r))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++")
      
        self.model.cpu()

        return np.array(train_exp_pred_data), np.array(new_train_labels), np.array(valid_exp_pred_data), \
                    np.array(valid_cls_labels), np.array(test_exp_pred_data), np.array(test_cls_labels)



    def train(self, saved_model_path = None):
        """train model on entire data"""
        exp_labels = self.exp_labels_mapping
        cls_labels = self.cls_labels
        # initialize base model
        self.model = MTLModel(bert_config = self.args.model_config, cls_hidden_size = self.args.cls_hidden_size, 
                    exp_hidden_size = self.args.exp_hidden_size, num_classes = self.num_classes)
        self.model.to(self.args.device)

        n_batches = int(np.ceil(len(self.data)/self.args.train_batch_size))
        # print("#batches for each epoch: {}".format(n_batches))
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr, eps=1e-8)
        total_step = n_batches * self.args.n_epochs

        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                num_warmup_steps=0,
                                                                num_training_steps=total_step)
        
        cls_criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
        exp_criterion = utils.resampling_rebalanced_crossentropy(seq_reduction = 'none')
        
        
        exp_weight = self.args.exp_weight
        
        for epoch in range(self.args.n_epochs):
            begin_time = time.time()
            #fit model
            train_loss = self.fit(self.input_ids, self.attention_masks, cls_labels, exp_labels, cls_criterion, 
                            exp_criterion, exp_weight, self.args.train_batch_size)
            print("Epoch: %d, train_loss: %.3f, time: %.3f" %(epoch, train_loss, time.time() - begin_time))

        
        cls_pred_labels, exp_pred_labels, _, _, _ = self.predict(self.input_ids, self.attention_masks, 
                    cls_labels, exp_labels, cls_criterion, exp_criterion, exp_weight, self.args.test_batch_size)
        exp_pred_labels = utils.max_pooling(exp_pred_labels, self.tokenized_data_slides, self.data)
        exp_true_labels = utils.max_pooling(exp_labels, self.tokenized_data_slides, self.data)
        cls_f1 = f1_score(cls_labels, cls_pred_labels, average = 'macro')
        exp_f1 = np.mean([f1_score(y_true, y_pred) for y_true, y_pred in zip(exp_true_labels, exp_pred_labels)])
        print("++ Training CLS F1: {}, EXP F1: {}".format(cls_f1, exp_f1))

        # extract predicted results 
        exp_pred_data = []
        new_labels = []
        for data, exp, cls_pred, cls_true in zip(self.data, exp_pred_labels, cls_pred_labels, cls_labels):
            if (self.args.full_train == False) and (cls_pred != cls_true):
                continue 
            new_labels.append(cls_true)
            text = data.split(" ")
            exp_pred_data.append(' '.join([text[i] if exp[i] == 1 else "*" for i in range(1, len(text)-1)]).strip())

        self.model.cpu()
        # saved model
        if saved_model_path is not None:
            print("Save model to path: {}".format(saved_model_path))
            torch.save(self.model.state_dict(), saved_model_path)

        return np.array(exp_pred_data), np.array(new_labels)

    def load(self, num_classes = 6, saved_model_path = None):
        if saved_model_path == None:
            print("Please enter the model path...")
            sys.exit(-1)
        try:
            self.model = MTLModel(bert_config = self.args.model_config, cls_hidden_size = self.args.cls_hidden_size, 
                        exp_hidden_size = self.args.exp_hidden_size, num_classes = num_classes)
            self.model.load_state_dict(torch.load(saved_model_path))
        
        except Exception as e:
            print("Exception")
            print(e)
        

    def classify(self, new_data):
        """ classify new data """
       
        self.model.to(self.args.device)
        
        data = np.array([self.tokenizer.cls_token+" "+x+ " "+ self.tokenizer.sep_token for x in new_data])
        tokenized_data, input_ids, attention_masks, tokenized_data_slides = \
                        utils.tokenize_text(self.tokenizer, data, self.tokenizer.pad_token)
        
        tokenized_data, input_ids = np.array(tokenized_data, dtype=object), torch.tensor(input_ids, dtype = torch.long)
        attention_masks, tokenized_data_slides= torch.tensor(attention_masks, dtype = torch.long), np.array(tokenized_data_slides, dtype=object)
        cls_preds, exp_preds, cls_probs, exp_probs, _ = self.predict(input_ids, attention_masks, 
                    batch_size = self.args.test_batch_size)
 
        exp_preds = utils.max_pooling(exp_preds, tokenized_data_slides, data)
        exp_probs = utils.max_pooling(exp_probs, tokenized_data_slides, data, prob = True)
        
        exp_texts = []
        exp_text_masked = []
        exp_text_probs = []
        for prepro_txt, exp_label, exp_prob in zip(data, exp_preds, exp_probs):
            try:
                text = prepro_txt.split(" ")
                exp_text_i = ' '.join(text[i] for i in range(1, len(text) - 1) if (exp_label[i]==1))
                exp_text_masked_i = ' '.join(text[i] if (exp_label[i]==1) else "*" for i in range(1, len(text) - 1))
                pred_text_prob_i = ' '.join(text[i]+"§§§"+str(exp_prob[i]) for i in range(1, len(text)-1))
                exp_texts.append(exp_text_i)
                exp_text_masked.append(exp_text_masked_i)
                exp_text_probs.append(pred_text_prob_i)
            except Exception as e:
                print("Exception: ...")
                print(e)
        
        return cls_preds, cls_probs, exp_texts, exp_text_masked, exp_text_probs
