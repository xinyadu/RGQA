from json import load
import os 
import json 
import re 
from collections import defaultdict, OrderedDict
import argparse 

import transformers 
from transformers import BartTokenizer
import torch 
from torch.utils.data import DataLoader 
import pytorch_lightning as pl 

from .data import IEDataset, my_collate
from .utils import load_ontology

from sentence_transformers import SentenceTransformer, util
import numpy as np


class ACEWIKIDataModule(pl.LightningDataModule):
    '''
    Dataset processing for ACEWIKI. Load Train from ACE, dev/test from WikiEvent. Involves chunking for long documents.
    '''
    def __init__(self, args):
        super().__init__() 
        self.hparams = args 
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.tokenizer.add_tokens([' <arg>',' <tgr>', ' <demo>'])
        self.sim_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        # self.MAX_LENGTH=400 # demo length (<400?)
        self.MAX_LENGTH=340
        # self.MAX_CONTEXT_LENGTH=200
        # self.MAX_CONTEXT_LENGTH=240
        # self.MAX_CONTEXT_LENGTH=20
        self.MAX_CONTEXT_LENGTH=30
        # self.MAX_TGT_LENGTH=50
        # self.MAX_TGT_LENGTH=60
        self.MAX_TGT_LENGTH=45
        # self.MAX_TGT_LENGTH=40

    def create_question_input_output_ace(self, question, anss, ex, ontology_dict, mark_trigger=True, augment_ir=False, index=0):
        '''
        If there are multiple events per example, use index parameter.
        question:
        Input context: 
        Output: 
        '''
        # get tokenized input context
        evt_type = ex['event_mentions'][index]['event_type']
        context_words = ex['tokens']

        trigger = ex['event_mentions'][index]['trigger']
        # trigger span does not include last index 
        if mark_trigger:
            prefix = self.tokenizer.tokenize(' '.join(context_words[:trigger['start']]), add_prefix_space=True) 
            tgt = self.tokenizer.tokenize(' '.join(context_words[trigger['start']: trigger['end']]), add_prefix_space=True)
            
            suffix = self.tokenizer.tokenize(' '.join(context_words[trigger['end']:]), add_prefix_space=True)
            context = prefix + [' <tgr>', ] + tgt + [' <tgr>', ] + suffix 
        else:
            context = self.tokenizer.tokenize(' '.join(context_words), add_prefix_space=True)

        # question
        tokenized_question = self.tokenizer.tokenize(question, add_prefix_space=True)

        # output
        ans_text = ' <arg> '.join(anss)
        space_tokenized_ans = ans_text.split()
        for idx in range(len(space_tokenized_ans)):
            if space_tokenized_ans[idx] == '<arg>':
                space_tokenized_ans[idx] = ' <arg>'
        tokenized_ans = [' <arg>'] 
        for w in space_tokenized_ans:
            tokenized_ans.extend(self.tokenizer.tokenize(w, add_prefix_space=True))


        return context, tokenized_question, tokenized_ans


    def create_question_input_output(self, question, anss, ex, ontology_dict, mark_trigger=True, augment_ir=False, index=0):
        '''
        If there are multiple events per example, use index parameter.
        question:
        Input context: 
        Output: 
        '''
        # get tokenized input context
        evt_type = ex['event_mentions'][index]['event_type']

        trigger = ex['event_mentions'][index]['trigger']
        offset = 0 
            # trigger span does not include last index 
        context_words = ex['tokens']
        center_sent = trigger['sent_idx']
        if len(context_words) > self.MAX_CONTEXT_LENGTH:
            cur_len = len(ex['sentences'][center_sent][0])
            context_words = [tup[0] for tup in ex['sentences'][center_sent][0]]
            if cur_len > self.MAX_CONTEXT_LENGTH:
                # one sentence is very long
                trigger_start = trigger['start']
                start_idx = max(0, trigger_start- self.MAX_CONTEXT_LENGTH//2 )
                end_idx = min(len(context_words), trigger_start + self.MAX_CONTEXT_LENGTH //2  )
                context_words = context_words[start_idx: end_idx]  # little bug here: for the first argument context_words if empty
                offset = start_idx 

            else:
                # take a sliding window 
                left = center_sent -1 
                right = center_sent +1 
                
                total_sents = len(ex['sentences'])
                prev_len =0 
                while cur_len >  prev_len:
                    prev_len = cur_len 
                    # try expanding the sliding window 
                    if left >= 0:
                        left_sent_tokens = [tup[0] for tup in ex['sentences'][left][0]]
                        if cur_len + len(left_sent_tokens) <= self.MAX_CONTEXT_LENGTH:
                            context_words = left_sent_tokens + context_words
                            left -=1 
                            cur_len += len(left_sent_tokens)
                    
                    if right < total_sents:
                        right_sent_tokens = [tup[0] for tup in ex['sentences'][right][0]]
                        if cur_len + len(right_sent_tokens) <= self.MAX_CONTEXT_LENGTH:
                            context_words = context_words + right_sent_tokens
                            right +=1 
                            cur_len += len(right_sent_tokens)
                # update trigger offset 
                offset = sum([len(ex['sentences'][idx][0]) for idx in range(left+1)])
        

        assert(len(context_words) <= self.MAX_CONTEXT_LENGTH) 

        trigger['start'] = trigger['start'] - offset 
        trigger['end'] = trigger['end'] - offset 
        if mark_trigger:
            prefix = self.tokenizer.tokenize(' '.join(context_words[:trigger['start']]), add_prefix_space=True) 
            tgt = self.tokenizer.tokenize(' '.join(context_words[trigger['start']: trigger['end']]), add_prefix_space=True)
            
            suffix = self.tokenizer.tokenize(' '.join(context_words[trigger['end']:]), add_prefix_space=True)
            context = prefix + [' <tgr>', ] + tgt + [' <tgr>', ] + suffix 
        else:
            context = self.tokenizer.tokenize(' '.join(context_words), add_prefix_space=True)

        # question
        tokenized_question = self.tokenizer.tokenize(question, add_prefix_space=True)

        # output
        ans_text = ' <arg> '.join(anss)
        space_tokenized_ans = ans_text.split()
        for idx in range(len(space_tokenized_ans)):
            if space_tokenized_ans[idx] == '<arg>':
                space_tokenized_ans[idx] = ' <arg>'
        tokenized_ans = [' <arg>'] 
        for w in space_tokenized_ans:
            tokenized_ans.extend(self.tokenizer.tokenize(w, add_prefix_space=True))


        return context, tokenized_question, tokenized_ans

            
    def prepare_data(self):
        if self.hparams.tmp_dir:
            data_dir = self.hparams.tmp_dir
        elif self.hparams.ir:
            # self.MAX_LENGTH=340 # epoch=4, epoch=5
            # self.MAX_TGT_LENGTH=50
            if self.hparams.add_yn:
                # self.MAX_TGT_LENGTH=50
                data_dir = 'preprocessed_ir_yn_{}'.format(self.hparams.dataset)
            else:
                data_dir = 'preprocessed_ir_{}'.format(self.hparams.dataset)
        else:
            data_dir = 'preprocessed_{}'.format(self.hparams.dataset)
        
        if not os.path.exists(data_dir):
            print('creating tmp dir ....')
            os.makedirs(data_dir)
            ontology_dict = load_ontology(dataset=self.hparams.dataset) 

            # get all_tuple_embs_train (for retrieval base)
            all_tuples_tokenized_train = []           
            all_tuple_embs_train = []
            with open(self.hparams.train_file,'r') as reader:
                for lidx, line in enumerate(reader):
                    ex = json.loads(line.strip())

                    for i in range(len(ex['event_mentions'])):
                        evt_type = ex['event_mentions'][i]['event_type']

                        if evt_type not in ontology_dict: # should be a rare event type 
                            print(evt_type)
                            continue 
                        
                        # input_template, output_template, context= self.create_gold_gen(ex, ontology_dict, self.hparams.mark_trigger, index=i)
                        role_to_anss = dict()
                        for role in ontology_dict[evt_type]['questions']:
                            role_to_anss[role] = []
                        for argument in ex['event_mentions'][i]['arguments']:
                            try:
                                role_to_anss[argument['role']].append(argument['text'])
                            except KeyError:
                                print(split, ex, argument['role'])

                        for role in ontology_dict[evt_type]['questions']:
                            question = ontology_dict[evt_type]['questions'][role]
                            anss = role_to_anss[role]
                            context, tokenized_question, tokenized_ans = self.create_question_input_output_ace(question, anss, ex, ontology_dict, self.hparams.mark_trigger, self.hparams.ir, index=i)
                            all_tuples_tokenized_train.append([context, tokenized_question, tokenized_ans, "_".join([ex['sent_id'], str(i)]), role])

                            all_tuple_embs_train.append(torch.from_numpy(
                                                    np.concatenate((self.sim_model.encode(' '.join(ex['tokens']), show_progress_bar=False), 
                                                                   self.sim_model.encode(question, show_progress_bar=False), 
                                                                   self.sim_model.encode(ex['event_mentions'][i]['trigger']['text'], show_progress_bar=False)),
                                                                   axis=None)
                                                                  ))

            # generate processed files
            for split,f in [('train',self.hparams.train_file), ('val',self.hparams.val_file), ('test',self.hparams.test_file)]:
                if (split in ['train', 'val']) and not f: #possible for eval_only
                    continue 

                all_tuples_tokenized = []
                all_tuple_embs = []
                with open(f,'r') as reader:
                    for lidx, line in enumerate(reader):
                        ex = json.loads(line.strip())

                        for i in range(len(ex['event_mentions'])):
                            evt_type = ex['event_mentions'][i]['event_type']

                            if evt_type not in ontology_dict: # should be a rare event type 
                                print(evt_type)
                                continue 
                            
                            # input_template, output_template, context= self.create_gold_gen(ex, ontology_dict, self.hparams.mark_trigger, index=i)
                            role_to_anss = dict()
                            for role in ontology_dict[evt_type]['questions']:
                                role_to_anss[role] = []
                            for argument in ex['event_mentions'][i]['arguments']:
                                try:
                                    role_to_anss[argument['role']].append(argument['text'])
                                except KeyError:
                                    print(split, ex, argument['role'])

                            for role in ontology_dict[evt_type]['questions']:
                                question = ontology_dict[evt_type]['questions'][role]
                                anss = role_to_anss[role]
                                if split == "train":
                                    context, tokenized_question, tokenized_ans = self.create_question_input_output_ace(question, anss, ex, ontology_dict, self.hparams.mark_trigger, self.hparams.ir, index=i)
                                else:
                                    context, tokenized_question, tokenized_ans = self.create_question_input_output(question, anss, ex, ontology_dict, self.hparams.mark_trigger, self.hparams.ir, index=i)

                                if split == "train":
                                    all_tuples_tokenized.append([context, tokenized_question, tokenized_ans, "_".join([ex['sent_id'], str(i)]), role])
                                else:
                                    all_tuples_tokenized.append([context, tokenized_question, tokenized_ans, "_".join([ex['doc_id'], str(i)]), role])

                                all_tuple_embs.append(torch.from_numpy(
                                                        np.concatenate((self.sim_model.encode(' '.join(ex['tokens']), show_progress_bar=False), 
                                                                       self.sim_model.encode(question, show_progress_bar=False), 
                                                                       self.sim_model.encode(ex['event_mentions'][i]['trigger']['text'], show_progress_bar=False)),
                                                                       axis=None)
                                                                      ))
                
                with open(os.path.join(data_dir,'{}.jsonl'.format(split)), 'w') as writer:
                    # import ipdb; ipdb.set_trace()
                    for idx, (tuple_tokenized, tuple_emb) in enumerate(zip(all_tuples_tokenized, all_tuple_embs)):
                        # import ipdb; ipdb.set_trace()

                        context, tokenized_question, tokenized_ans, doc_key, role = tuple_tokenized
                        all_tokens_before_context = tokenized_question
                        all_tokenized_ans = tokenized_ans.copy()
                        if self.hparams.ir:
                            most_sim = util.semantic_search([tuple_emb], all_tuple_embs_train, top_k=2)[0][1]
                            sim_context, sim_tokenized_question, sim_tokenized_ans, _, _ = all_tuples_tokenized_train[most_sim['corpus_id']]
                            # import ipdb; ipdb.set_trace()
                            all_tokens_before_context = sim_tokenized_ans + [' <demo>'] + all_tokens_before_context
                            all_tokens_before_context = sim_tokenized_question + ['</s>', '</s>'] + sim_context + ['</s>', '</s>'] + all_tokens_before_context
                            if self.hparams.add_yn:
                                if (len(sim_tokenized_ans)>1 and len(tokenized_ans)>1):
                                    all_tokenized_ans = self.tokenizer.tokenize('Y', add_prefix_space=True) + tokenized_ans
                                else:
                                    all_tokenized_ans = self.tokenizer.tokenize('N', add_prefix_space=True) + tokenized_ans
                            # if len(tokenized_ans)==1: import ipdb; ipdb.set_trace()
                        
                        if len(all_tokens_before_context)+len(context)>self.MAX_LENGTH:
                            print(doc_key)
                        input_tokens = self.tokenizer.encode_plus(all_tokens_before_context, context, 
                                add_special_tokens=True,
                                add_prefix_space=True,
                                max_length=self.MAX_LENGTH,
                                truncation='only_second',
                                padding='max_length')
                        tgt_tokens = self.tokenizer.encode_plus(all_tokenized_ans, 
                        add_special_tokens=True,
                        add_prefix_space=True, 
                        max_length=self.MAX_TGT_LENGTH,
                        truncation=True,
                        padding='max_length')

                        # import ipdb; ipdb.set_trace()
                        processed_ex = {
                            'doc_key': doc_key, #this is not unique 
                            'role': role,
                            'input_token_ids':input_tokens['input_ids'],
                            'input_attn_mask': input_tokens['attention_mask'],
                            'tgt_token_ids': tgt_tokens['input_ids'],
                            'tgt_attn_mask': tgt_tokens['attention_mask'],
                            'ans': all_tokenized_ans
                        }
                        writer.write(json.dumps(processed_ex) + '\n')

    
    def train_dataloader(self):
        if self.hparams.tmp_dir:
            data_dir = self.hparams.tmp_dir
        elif self.hparams.ir:
            if self.hparams.add_yn:
                data_dir = 'preprocessed_ir_yn_{}'.format(self.hparams.dataset)
            else:
                data_dir = 'preprocessed_ir_{}'.format(self.hparams.dataset)
        else:
            data_dir = 'preprocessed_{}'.format(self.hparams.dataset)

        dataset = IEDataset(os.path.join(data_dir, 'train.jsonl'))
        
        dataloader = DataLoader(dataset, 
            pin_memory=True, num_workers=2, 
            collate_fn=my_collate,
            batch_size=self.hparams.train_batch_size, 
            shuffle=True)
        return dataloader 

    
    def val_dataloader(self):
        if self.hparams.tmp_dir:
            data_dir = self.hparams.tmp_dir
        elif self.hparams.ir:
            if self.hparams.add_yn:
                data_dir = 'preprocessed_ir_yn_{}'.format(self.hparams.dataset)
            else:
                data_dir = 'preprocessed_ir_{}'.format(self.hparams.dataset)
        else:
            data_dir = 'preprocessed_{}'.format(self.hparams.dataset)

        dataset = IEDataset(os.path.join(data_dir, 'val.jsonl'))
        
        
        dataloader = DataLoader(dataset, pin_memory=True, num_workers=2, 
            collate_fn=my_collate,
            batch_size=self.hparams.eval_batch_size, shuffle=False)
        return dataloader

    def test_dataloader(self):
        if self.hparams.tmp_dir:
            data_dir = self.hparams.tmp_dir
        elif self.hparams.ir:
            if self.hparams.add_yn:
                data_dir = 'preprocessed_ir_yn_{}'.format(self.hparams.dataset)
            else:
                data_dir = 'preprocessed_ir_{}'.format(self.hparams.dataset)
        else:
            data_dir = 'preprocessed_{}'.format(self.hparams.dataset)

        dataset = IEDataset(os.path.join(data_dir, 'test.jsonl'))
        
        
        dataloader = DataLoader(dataset, pin_memory=True, num_workers=2, 
            collate_fn=my_collate, 
            batch_size=self.hparams.eval_batch_size, shuffle=False)

        return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--train-file',type=str)
    parser.add_argument('--val-file', type=str)
    parser.add_argument('--test-file', type=str)
    parser.add_argument('--tmp_dir', default='tmp')
    parser.add_argument('--coref-dir', type=str, default='data/kairos/coref')
    parser.add_argument('--use_info', action='store_true', default=True, help='use informative mentions instead of the nearest mention.')
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='ACEWIKI')
    parser.add_argument('--mark-trigger', action='store_true', default=True)
    parser.add_argument('--ir', action='store_true', default=False)
    parser.add_argument("--add_yn", action='store_true', default=False, help='whether to add Y and N in ans')
    
    args = parser.parse_args() 

    dm = ACEWIKIDataModule(args=args)
    dm.prepare_data() 

    # training dataloader 
    dataloader = dm.train_dataloader() 
    # dataloader = dm.val_dataloader() 

    for idx, batch in enumerate(dataloader):
        print(batch)
        break 

    