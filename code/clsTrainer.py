import numpy as np 
import transformers
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.metrics import f1_score, precision_score, recall_score 
from collections import OrderedDict
import torch
import torch.nn as nn
import random
import utils
import time


class CLSModel(nn.Module):
    """base model"""
    def __init__(self, bert_config = "vinai/bertweet-base", hidden_size = 768, num_classes = 6):
        super(CLSModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_config)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes, bias = True)
    
    def forward(self, input_ids = None, attention_mask = None, token_type_ids = None, return_dict=None):
        outputs = self.bert(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, return_dict = return_dict)
        outputs = self.linear(self.dropout(outputs[1]))
        return outputs

class CLSTrainer:
    def __init__(self, data = None, labels = None, args = None):
        self.args = args 
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_config)
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.num_classes = len(set(labels))
        
    
    def predict(self, data, labels = None, loss_func = None, batch_size = 128):
        """make prediction"""
        self.model.eval()
        
        total_loss = 0 
        outputs = []
        output_probs = []

        with torch.no_grad():
            for batch_start in range(0, len(data), batch_size):
                batch_end = min(batch_start+batch_size, len(data))
                X_batch = data[batch_start:batch_end]

                sents_tensor, masks_tensor, _ = utils.convert_sents_to_ids_tensor(self.tokenizer, X_batch, self.tokenizer.pad_token)
                out = self.model(input_ids = sents_tensor.to(self.args.device), attention_mask = masks_tensor.to(self.args.device))
                if labels is not None:
                    y_batch = labels[batch_start:batch_end]
                    loss = loss_func(out, torch.tensor(y_batch, dtype = torch.long).to(self.args.device))
                    total_loss = loss.item()
                
                out = nn.Softmax(dim = -1)(out).max(dim = -1)
                outputs += out.indices.cpu()
                output_probs += out.values.cpu()
            
        loss = total_loss/len(data)
        return loss, outputs, output_probs 

    def fit(self, data, labels, batch_size, loss_func):
        """fit model"""
        self.model.train()
        train_loss = 0
        epoch_indices = random.sample([i for i in range(len(data))], len(data))
        batch_num = 0
        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start+batch_size, len(data))
            batch_indices = epoch_indices[batch_start: batch_end]
            X_batch = data[batch_indices]
            y_batch = labels[batch_indices]
            sents_tensor, masks_tensor, _ = utils.convert_sents_to_ids_tensor(self.tokenizer, X_batch, self.tokenizer.pad_token)
            output = self.model(input_ids = sents_tensor.to(self.args.device), attention_mask = masks_tensor.to(self.args.device))
            self.model.zero_grad()
            
            loss = loss_func(output, y_batch)
            train_loss += loss.item()
            batch_num += 1
            loss.backward()

            # clip the norm of the gradients to 1, to prevent exploding
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # update parameters
            self.optimizer.step()

            # update learning rate
            self.scheduler.step()

        return train_loss/batch_num

    def eval(self, train_data = None, train_labels = None, valid_data = None, valid_labels = None, 
            test_data = None, test_labels = None, train_indices = None, valid_indices = None, test_indices = None):
        """train model"""

        if (train_indices is not None):
            train_data, valid_data, test_data = self.data[train_indices], self.data[valid_indices], self.data[test_indices]
            train_labels, valid_labels, test_labels = self.labels[train_indices], self.labels[valid_indices], self.labels[test_indices]

        train_data = np.array([self.tokenizer.cls_token +" " + x +" " + self.tokenizer.sep_token for x in train_data])
        valid_data = np.array([self.tokenizer.cls_token +" " + x +" " + self.tokenizer.sep_token for x in valid_data])
        test_data = np.array([self.tokenizer.cls_token +" " + x +" " + self.tokenizer.sep_token for x in test_data])

        self.model = CLSModel(self.args.model_config, num_classes = self.num_classes)
        
        self.optimizer = AdamW(self.model.parameters(), self.args.lr, eps = 1e-8)

        n_batches = int(np.ceil(len(train_data)/self.args.train_batch_size))
        # print("Number of batches: ", n_batches)
        total_step = n_batches * self.args.n_epochs
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                 num_warmup_steps=0,
                                                                 num_training_steps=total_step)
        
        self.model.to(self.args.device)
        criterion = torch.nn.CrossEntropyLoss()

        train_labels = torch.tensor(train_labels, dtype = torch.long).to(self.args.device)
        valid_loss_hist = []
        best_epoch = {} 
        best_model_state_dict = None 
        best_loss = float('inf')

        for epoch in range(self.args.n_epochs):
            begin_time = time.time()
            train_loss = self.fit(train_data, train_labels, self.args.train_batch_size, criterion)

            #validation: 
            valid_loss, _, _ = self.predict(valid_data, valid_labels, criterion, self.args.test_batch_size)
            print("Epoch: %d, train loss: %.3f, valid loss: %.3f, time: %.3f" %(epoch, train_loss, valid_loss, time.time()-begin_time))
            valid_loss_hist.append(valid_loss)

        
            if self.args.patience!=0:
                if best_loss > valid_loss:
                    best_loss = valid_loss
                    best_epoch = {'epoch': epoch, 'valid_loss': valid_loss}
                    best_model_state_dict = OrderedDict({k: v.cpu() for k, v in self.model.state_dict().items()})
                if epoch - best_epoch['epoch']> self.args.patience:
                    break

        print("+ Training ends!")
        print("+ Load best model {}---valid_loss: {}".format(best_epoch['epoch'], best_epoch['valid_loss']))
        self.model.load_state_dict(best_model_state_dict)
        self.model.to(self.args.device)
        
        # make prediction on test data
        _, y_pred, _ = self.predict(test_data, test_labels, criterion, batch_size = self.args.test_batch_size)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++ CLS F1: %.4f                           +" %(f1_score(test_labels, y_pred, average='macro')))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++")

        self.model.cpu()
    

    def train(self, saved_model_path = None):
        train_data = np.array([self.tokenizer.cls_token +" " + x +" " + self.tokenizer.sep_token for x in self.data])
        self.model = CLSModel(self.args.model_config, num_classes = self.num_classes)
        self.optimizer = AdamW(self.model.parameters(), self.args.lr, eps = 1e-8)

        n_batches = int(np.ceil(len(train_data)/self.args.train_batch_size))
        # print("Number of batches: ", n_batches)
        total_step = n_batches * self.args.n_epochs
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                 num_warmup_steps=0,
                                                                 num_training_steps=total_step)
        
        self.model.to(self.args.device)
        criterion = torch.nn.CrossEntropyLoss()

        train_labels = torch.tensor(self.labels, dtype = torch.long).to(self.args.device)
        
        print("Training...")
        for epoch in range(self.args.n_epochs):
            begin_time = time.time()
            train_loss = self.fit(train_data, train_labels, self.args.train_batch_size, criterion)

            print("Epoch: %d, train loss: %.3f, time: %.3f" %(epoch, train_loss, time.time()-begin_time))

        _, y_pred, _ = self.predict(self.data, self.labels, criterion, batch_size = self.args.test_batch_size)
        print("++ Train CLS F1: {}".format(f1_score(self.labels, y_pred, average = 'macro')))
        
        self.model.cpu()
        if saved_model_path is not None:
            print("Save model to path: {}".format(saved_model_path))
            torch.save(self.model.state_dict(), saved_model_path)
    
    def load(self, saved_model_path = None):
        try: 
            self.model = CLSModel(self.args.model_config, num_classes = self.num_classes)
            self.model.load_state_dict(torch.load(saved_model_path))
        except Exception as e:
            print("Exception")
            print(e)

    def classify(self, new_data):
        data = np.array([self.tokenizer.cls_token +" " + x +" " + self.tokenizer.sep_token for x in new_data])
        self.model.to(self.args.device)
        
        _, y_preds, y_probs = self.predict(self.data, batch_size = self.args.test_batch_size)

        return y_preds, y_probs
                   