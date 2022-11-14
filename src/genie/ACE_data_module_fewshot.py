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


class ACEDataModuleFewshot(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__() 
        self.hparams = args 
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.tokenizer.add_tokens([' <arg>',' <tgr>', ' <demo>'])
        self.sim_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.MAX_LENGTH=340

        self.MAX_TGT_LENGTH=50
        # MAX_TGT_LENGTH=50 (best)

        if self.hparams.gpus==None:
            self.num_worker = 8
        else:
            self.num_worker = 56
    
    def create_question_input_output(self, question, anss, ex, ontology_dict, mark_trigger=True, augment_ir=False, index=0):
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

            
    def prepare_data(self):
        # import ipdb; ipdb.set_trace()
        if self.hparams.eval_only:
            return

        if self.hparams.tmp_dir:
            data_dir = self.hparams.tmp_dir
        elif self.hparams.ir:
            self.MAX_LENGTH=340 # epoch=4, epoch=5
            self.MAX_TGT_LENGTH=50
            if self.hparams.add_yn:
                self.MAX_TGT_LENGTH=50
                data_dir = 'preprocessed_ir_yn_{}'.format(self.hparams.dataset)
            else:
                data_dir = 'preprocessed_ir_{}'.format(self.hparams.dataset)
            # data_dir = 'preprocessed_ir_{}'.format(self.hparams.dataset)
        else:
            data_dir = 'preprocessed_{}'.format(self.hparams.dataset)
        data_dir = "fewshot_" + data_dir


        if not os.path.exists(data_dir):
            print('creating tmp dir ....')
            os.makedirs(data_dir)

        if os.path.exists(os.path.join(data_dir,'val.jsonl')) and os.path.exists(os.path.join(data_dir,'test.jsonl')) and os.path.exists(os.path.join(data_dir,'train_{}_{}.jsonl'.format(self.hparams.sampling, self.hparams.shot))):
            return 

        print("prepare_data!!!!!!!!!!!!")

        if self.hparams.dataset == 'combined':
            ontology_dict = load_ontology(dataset='KAIROS')
        else:
            ontology_dict = load_ontology(dataset=self.hparams.dataset) 
        
        # get all_tuple_embs_train 
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
                        except KeyError: # since it's in train, not printed
                            print(split, ex, argument['role'])

                    for role in ontology_dict[evt_type]['questions']:
                        question = ontology_dict[evt_type]['questions'][role]
                        anss = role_to_anss[role]
                        context, tokenized_question, tokenized_ans = self.create_question_input_output(question, anss, ex, ontology_dict, self.hparams.mark_trigger, self.hparams.ir, index=i)
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

            if (split == 'val' or split == 'test') and os.path.exists(os.path.join(data_dir,'{}.jsonl'.format(split))):
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
                                # print(split, ex, argument['role'])
                                pass

                        for role in ontology_dict[evt_type]['questions']:
                            question = ontology_dict[evt_type]['questions'][role]
                            anss = role_to_anss[role]
                            context, tokenized_question, tokenized_ans = self.create_question_input_output(question, anss, ex, ontology_dict, self.hparams.mark_trigger, self.hparams.ir, index=i)
                            all_tuples_tokenized.append([context, tokenized_question, tokenized_ans, "_".join([ex['sent_id'], str(i)]), role])

                            all_tuple_embs.append(torch.from_numpy(
                                                    np.concatenate((self.sim_model.encode(' '.join(ex['tokens']), show_progress_bar=False), 
                                                                   self.sim_model.encode(question, show_progress_bar=False), 
                                                                   self.sim_model.encode(ex['event_mentions'][i]['trigger']['text'], show_progress_bar=False)),
                                                                   axis=None)
                                                                  ))
            
            # with open(os.path.join(data_dir,'{}.jsonl'.format(split)), 'w') as writer:
            if split == 'train':
                writer = open(os.path.join(data_dir,'{}_{}_{}.jsonl'.format(split, self.hparams.sampling, self.hparams.shot)), 'w')
            else:
                # if os.path.exists(os.path.join(data_dir,'{}.jsonl'.format(split))):
                #     continue
                writer = open(os.path.join(data_dir,'{}.jsonl'.format(split)), 'w')

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
                        # if (len(sim_tokenized_ans)>1 and len(tokenized_ans)>1) or (len(sim_tokenized_ans)==1 and len(tokenized_ans)==1):
                        if (len(sim_tokenized_ans)>1 and len(tokenized_ans)>1):
                        # if ((len(sim_tokenized_ans)>1 and len(tokenized_ans)>1)) and most_sim['score']>0.8:
                        # print(most_sim['score'])
                        # if most_sim['score']>0.8:
                            all_tokenized_ans = self.tokenizer.tokenize('Y', add_prefix_space=True) + tokenized_ans
                            # all_tokenized_ans = self.tokenizer.tokenize('<arg> Y', add_prefix_space=True) + tokenized_ans
                            # all_tokenized_ans = self.tokenizer.tokenize('Y <demo>', add_prefix_space=True) + tokenized_ans
                            # all_tokenized_ans = self.tokenizer.tokenize('1', add_prefix_space=True) + tokenized_ans
                            # all_tokenized_ans = self.tokenizer.tokenize('yes', add_prefix_space=True) + tokenized_ans
                        else:
                            all_tokenized_ans = self.tokenizer.tokenize('N', add_prefix_space=True) + tokenized_ans
                            # all_tokenized_ans = self.tokenizer.tokenize('<arg> N', add_prefix_space=True) + tokenized_ans
                            # all_tokenized_ans = self.tokenizer.tokenize('N <demo>', add_prefix_space=True) + tokenized_ans
                            # all_tokenized_ans = self.tokenizer.tokenize('0', add_prefix_space=True) + tokenized_ans
                            # all_tokenized_ans = self.tokenizer.tokenize('no', add_prefix_space=True) + tokenized_ans
                    # if len(tokenized_ans)==1: import ipdb; ipdb.set_trace()
                
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
        data_dir = "fewshot_" + data_dir

        dataset = IEDataset(os.path.join(data_dir,'train_{}_{}.jsonl'.format(self.hparams.sampling, self.hparams.shot)))
        
        dataloader = DataLoader(dataset, 
            pin_memory=True, num_workers=self.num_worker, 
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
        data_dir = "fewshot_" + data_dir

        dataset = IEDataset(os.path.join(data_dir, 'val.jsonl'))
        
        
        dataloader = DataLoader(dataset, pin_memory=True, num_workers=self.num_worker, 
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
        data_dir = "fewshot_" + data_dir

        dataset = IEDataset(os.path.join(data_dir, 'test.jsonl'))
        
        
        dataloader = DataLoader(dataset, pin_memory=True, num_workers=self.num_worker, 
            collate_fn=my_collate, 
            batch_size=self.hparams.eval_batch_size, shuffle=False)

        return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--train-file',type=str)
    parser.add_argument('--val-file', type=str)
    parser.add_argument('--test-file', type=str)
    parser.add_argument('--tmp_dir', default='tmp')
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--mark-trigger', action='store_true', default=True)
    parser.add_argument('--ir', action='store_true', default=False)
    parser.add_argument("--add_yn", action='store_true', default=False, help='whether to add Y and N in ans')
    parser.add_argument('--dataset', type=str, default='combined')

    # fewshot args
    parser.add_argument("--fewshot", action='store_true', default=False, help='if do fewshot train/test')
    parser.add_argument("--sampling", type=str, required=False, choices=['random', 'ppl', 'c_context', 'c_context_trg'])
    parser.add_argument("--shot", type=int, required=False, choices=[10, 20, 50, 100, 200])
    args = parser.parse_args() 

    dm = ACEDataModuleFewshot(args=args)
    dm.prepare_data() 

    # training dataloader 
    dataloader = dm.train_dataloader() 

    for idx, batch in enumerate(dataloader):
        print(batch)
        break 

    