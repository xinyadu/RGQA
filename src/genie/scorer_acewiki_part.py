import os 
import json 
import argparse 
import re 
from copy import deepcopy
from collections import defaultdict 
from collections import OrderedDict
from tqdm import tqdm
import spacy 


from utils import load_ontology,find_arg_span, compute_f1, get_entity_span, find_head, WhitespaceTokenizer

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

'''
Scorer for argument extraction on ACE & KAIROS. For the RAMS dataset, the official scorer is used. 
Outputs: F1, Head F1, # Coref F1 
'''
def clean_span(ex, span):
    tokens = ex['tokens']
    if tokens[span[0]].lower() in {'the', 'an', 'a'}:
        if span[0]!=span[1]:
            return (span[0]+1, span[1])
    return span 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen-file',type=str,default='checkpoints/gen-ACE-gen-pred/predictions.jsonl' )
    parser.add_argument('--test-file', type=str,default='data/ace/json/test.oneie.json')
    parser.add_argument('--coref-file', type=str)
    parser.add_argument('--head-only', action='store_true', default=False)
    parser.add_argument('--coref', action='store_true', default=False)
    parser.add_argument('--dataset',type=str, default='ACE', choices=['ACE', 'KAIROS','AIDA', 'ACEWIKI', 'WIKI'])
    args = parser.parse_args() 

    ontology_dict_ACE = load_ontology(dataset='ACE')
    ontology_dict_WIKI = load_ontology(dataset='WIKI')
    # import ipdb; ipdb.set_trace()
    ace_types = []
    for key in ontology_dict_ACE:
        h1, h2 = key.split(":")
        ace_types.append(h1)
        ace_types.append(h2)

    distinct_types = []
    for key in ontology_dict_WIKI:
        # import ipdb; ipdb.set_trace()
        # print(key)
        if ':' in key: continue

        h1, h2, h3 = key.split('.')[0], key.split('.')[1], key.split('.')[2]
        if h1 not in ace_types and h2 not in ace_types and h3 not in ace_types:
            distinct_types.append(key)
    # import ipdb; ipdb.set_trace()

    distinct_types = []
    # distinct_types += ['Disaster.Crash.Unspecified']
    # distinct_types += ['Control.ImpedeInterfereWith.Unspecified']
    distinct_types += ['Disaster.DiseaseOutbreak.Unspecified']
    distinct_types += ['Disaster.FireExplosion.Unspecified']

    distinct_types += ['GenericCrime.GenericCrime.GenericCrime']

    distinct_types += ['Medical.Diagnosis.Unspecified', 'Medical.Intervention.Unspecified', 'Medical.Vaccinate.Unspecified']

    distinct_types += ["Movement.Transportation.PreventPassage"]

    distinct_types += ['ArtifactExistence.DamageDestroyDisableDismantle.Damage', 'ArtifactExistence.DamageDestroyDisableDismantle.Destroy', 'ArtifactExistence.DamageDestroyDisableDismantle.DisableDefuse', 'ArtifactExistence.DamageDestroyDisableDismantle.Dismantle', 'ArtifactExistence.DamageDestroyDisableDismantle.Unspecified', 'ArtifactExistence.ManufactureAssemble.Unspecified']

    # distinct_types += ["Conflict.Defeat.Unspecified"]
    # distinct_types += ["Contact.Contact.Broadcast"]
    # distinct_types += ["Control.ImpedeInterfereWith.Unspecified"]

    distinct_types += ['Cognitive.IdentifyCategorize.Unspecified', 'Cognitive.Inspection.SensoryObserve', 'Cognitive.Research.Unspecified', 'Cognitive.TeachingTrainingLearning.Unspecified']

    distinct_types += ['Life.Illness.Unspecified'] 
    distinct_types += ["Life.Consume.Unspecified"]
    distinct_types += ["Life.Infect.Unspecified"]

    distinct_types += ['Justice.InvestigateCrime.Unspecified']

    distinct_types += ["Transaction.Donation.Unspecified"]

    distinct_types = sorted(distinct_types)

    for t in distinct_types:
        print(t, "\\\\")

    # distinct_types += ["Life.Injure.Unspecified"]

    ontology_dict = load_ontology(dataset=args.dataset)

    # if args.dataset == 'KAIROS' and args.coref and not args.coref_file:
    #     print('coreference file needed for the KAIROS dataset.')
    #     raise ValueError
    # if args.dataset == 'AIDA' and args.coref:
    #     raise NotImplementedError

    raw_examples = {}
    with open(args.gen_file,'r') as f:
        for line in f: # this solution relies on keeping the exact same order 
            pred = json.loads(line.strip()) 
            doc_key, role = pred['doc_key'], pred['role']
            if doc_key not in raw_examples:
                raw_examples[doc_key] = {}
            raw_examples[doc_key][role] = { # 'doc_key' also keeps order of events
                'predicted': pred['predicted'],
                'gold': pred['gold'],
            }

    examples = {}
    for doc_key in raw_examples:
        ex = raw_examples[doc_key]
        if doc_key not in examples:
            examples[doc_key] = {'predicted_args': {}, 'gold_args': {}}

        for role in ex:
            if role not in examples[doc_key]['predicted_args']:
                examples[doc_key]['predicted_args'][role] = []

            candidate_args = ex[role]['predicted'].split(" <arg>")
            for i in range(len(candidate_args)):
                arg = candidate_args[i].strip()
                # import ipdb; ipdb.set_trace()
                # if arg and i!=0:
                if arg and arg != 'Y' and arg != 'N':
                    examples[doc_key]['predicted_args'][role].append(arg)

    # import ipdb; ipdb.set_trace()
    # with open(args.test_file, 'r') as f:
    #     for line in f:
    #         doc = json.loads(line.strip())
    #         if 'sent_id' in doc.keys():
    #             doc_id = doc['sent_id']
    #             # print('evaluating on sentence level')
    #         else:
    #             doc_id = doc['doc_id']
    #             # print('evaluating on document level')
            # for idx, eid in enumerate(doc2ex[doc_id]):
            #     examples[eid]['tokens'] = doc['tokens']
            #     examples[eid]['event'] = doc['event_mentions'][idx]
            #     examples[eid]['entity_mentions'] = doc['entity_mentions']
    with open(args.test_file, 'r') as f:
        for line in f:
            sent = json.loads(line.strip())
            doc_id = sent['doc_id']
            tokens = sent['tokens']
            for i, event in enumerate(sent['event_mentions']):
                doc_key = "_".join([doc_id, str(i)])
                # evt_type not in ontology
                if doc_key not in examples:
                    print(event)
                    continue
                examples[doc_key]['tokens'] = sent['tokens']
                examples[doc_key]['event'] = sent['event_mentions'][i]
                examples[doc_key]['entity_mentions'] = sent['entity_mentions']


        

    pred_arg_num, gold_arg_num =0, 0
    arg_idn_num, arg_class_num =0, 0 
    # arg_idn_coref_num =0
    # arg_class_coref_num =0

    # for doc_key, ex in tqdm(examples.items())
    for doc_key, ex in examples.items():
        context_words = ex['tokens']
        doc = None 
        if args.head_only:
            doc = nlp(' '.join(context_words))
        
        evt_type = ex['event']['event_type']
        if evt_type not in ontology_dict: continue 

        # evt_major_types_distinct = ["Cognitive", "Disaster", "Medical", "Contact"]
        # evt_major_types_distinct = ["Cognitive"]
        # evt_major_types_distinct = ["Disaster"]
        # evt_major_types_distinct = ["Medical"]
        # evt_major_types_distinct = ["Medical", "Disaster"]
        # distinct_types = ["Life.Infect", "Life.Injure", "Life.Illness", "Medical", "Disaster"]
        # evt_major_types_distinct = ["Movement"]
        # evt_major_types_distinct = ["ArtifactExistence"]
        distinct = False
        # import ipdb; ipdb.set_trace()
        for evt_major_type in distinct_types:
            if evt_major_type == evt_type:
                # import ipdb; ipdb.set_trace()
                # continue
                distinct = True
        if not distinct: continue

        # # extract argument text 
        predicted_args = {}
        for role in ex['predicted_args']:
            if role not in predicted_args:
                predicted_args[role] = []
            arguments = ex['predicted_args'][role]
            for arg in arguments:
                predicted_args[role].append(arg.strip().split())

        # import ipdb; ipdb.set_trace() 
        # get predicted argument spans
        # get trigger 
        # extract argument span (get offset)
        trigger_start, trigger_end = ex['event']['trigger']['start'], ex['event']['trigger']['end']
        predicted_set = set() 
        for role in predicted_args:
            for entity in predicted_args[role]:# this argument span is inclusive, FIXME: this might be problematic 
                arg_span = find_arg_span(entity, context_words, 
                    trigger_start, trigger_end, head_only=args.head_only, doc=doc) 
                
                if arg_span:# if None means hullucination
                    predicted_set.add((arg_span[0], arg_span[1], evt_type, role))
                else:
                    new_entity = []
                    for w in entity:
                        if w == 'and' and len(new_entity) >0:
                            arg_span = find_arg_span(new_entity, context_words, trigger_start, trigger_end, head_only=args.head_only, doc=doc)
                            if arg_span: predicted_set.add((arg_span[0], arg_span[1], evt_type, role))
                            new_entity = []
                        else:
                            new_entity.append(w)
                    
                    if len(new_entity)>0: # last entity
                        arg_span = find_arg_span(new_entity, context_words, trigger_start, trigger_end, head_only=args.head_only, doc=doc)
                        if arg_span: predicted_set.add((arg_span[0], arg_span[1], evt_type, role))
        # import ipdb; ipdb.set_trace()    
        # get gold spans         
        gold_set = set() 
        # gold_canonical_set = set() # set of canonical mention ids, singleton mentions will not be here 
        for arg in ex['event']['arguments']:
            role = arg['role']
            entity_id = arg['entity_id']
            span = get_entity_span(ex, entity_id)
            span = (span[0], span[1]-1)
            span = clean_span(ex, span)
            # clean up span by removing `a` `the`
            if args.head_only and span[0]!=span[1]:
                span = find_head(span[0], span[1], doc=doc) 
            
            gold_set.add((span[0], span[1], evt_type, role))
            # if args.coref:
            #     if span in coref_mapping[doc_id]:
            #         canonical_id = coref_mapping[doc_id][span]
            #         gold_canonical_set.add((canonical_id, evt_type, argname))
        

        pred_arg_num += len(predicted_set)
        gold_arg_num += len(gold_set)
        # check matches 
        for pred_arg in predicted_set:
            arg_start, arg_end, event_type, role = pred_arg
            gold_idn = {item for item in gold_set if item[0] == arg_start and item[1] == arg_end and item[2] == event_type}
            if gold_idn:
                arg_idn_num += 1
                gold_class = {item for item in gold_idn if item[-1] == role}
                if gold_class:
                    arg_class_num += 1
            # elif args.coref:# check coref matches 
            #     arg_start, arg_end, event_type, role = pred_arg
            #     span = (arg_start, arg_end)
            #     if span in coref_mapping[doc_id]:
            #         canonical_id = coref_mapping[doc_id][span]
            #         gold_idn_coref = {item for item in gold_canonical_set 
            #             if item[0] == canonical_id and item[1] == event_type}
            #         if gold_idn_coref:
            #             arg_idn_coref_num +=1 
            #             gold_class_coref = {item for item in gold_idn_coref
            #             if item[2] == role}
            #             if gold_class_coref:
            #                 arg_class_coref_num +=1 

    # import ipdb; ipdb.set_trace()
    # calcualte p/r/f1 and report
    if args.head_only: print('Evaluation by matching head words only....')
    role_id_prec, role_id_rec, role_id_f = compute_f1(pred_arg_num, gold_arg_num, arg_idn_num)
    role_prec, role_rec, role_f = compute_f1(pred_arg_num, gold_arg_num, arg_class_num)
    print('Role identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
    print('Role: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(role_prec * 100.0, role_rec * 100.0, role_f * 100.0))

    # if args.coref:
    #     role_id_prec, role_id_rec, role_id_f = compute_f1(
    #     pred_arg_num, gold_arg_num, arg_idn_num + arg_idn_coref_num)
    #     role_prec, role_rec, role_f = compute_f1(
    #         pred_arg_num, gold_arg_num, arg_class_num + arg_class_coref_num)
    #     print('Coref Role identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
    #         role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
    #     print('Coref Role: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
    #         role_prec * 100.0, role_rec * 100.0, role_f * 100.0))
