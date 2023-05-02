import argparse
import os
import json
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm, trange
from data_utils import *
from dataset_classes.snli import SNLI_Partition
from dataset_classes.boolq import BoolQ_Partition


TASK_FILES = {'spider': ['all_examples'],
              'snli': {'original': ['train', 'dev', 'test'], 'processed': ['all_examples']},
              'boolq': {'original': ['train', 'val'], 'processed': ['all_examples']}
              }
TASK_PROCESSOR = {'spider': load_spider,
                  'snli': SNLI_Partition,
                  'boolq': BoolQ_Partition
                  }
TASK_QUERY_COLS = {'snli': 'hypothesis',
                   'boolq': 'question'}

def run_lm_strided(df, model, tokenizer, device):
    ppl_arr, lls_arr = [], []
    for st_idx in trange(len(df)):
        encodings = tokenizer(df.loc[st_idx, 'query'], return_tensors='pt')

        max_length = model.config.n_positions
        stride = 512

        nlls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        lls_arr.append(-torch.stack(nlls).sum().cpu().numpy())
        ppl_arr.append(ppl.cpu().numpy())

    lls_arr = np.array(lls_arr)
    ppl_arr = np.array(ppl_arr)
    return lls_arr, ppl_arr


def run_lm_strided_wikitext(text, model, tokenizer, device):
    lls_arr = []
    max_length = model.config.n_positions
    stride = 512

    encodings = tokenizer(text, return_tensors='pt')
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids)

            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_token_nll_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            per_token_nll_loss = per_token_nll_loss[-trg_len:]
            lls_arr.append(-per_token_nll_loss.detach().cpu().numpy())

    lls_arr = np.concatenate(lls_arr)
    return lls_arr


def run_lm_lambada(dset, model, tokenizer, prefix, batch_size, device):
    ppl_arr, lls_arr = [], []
    output_log = []
    for st_idx in trange(0, len(dset), batch_size):
        with torch.no_grad():
            batch_text = [[prefix + ' '.join(q_str.split(' ')[:-1]), ' ' + q_str.split(' ')[-1]]
                          for q_str in dset[st_idx:st_idx + batch_size]['text']]
            encodings = tokenizer(batch_text, padding='longest', truncation=True, return_tensors='pt',
                                  return_token_type_ids=True)
            model_inputs = model.prepare_inputs_for_generation(input_ids=encodings.input_ids,
                                                               attention_mask=encodings.attention_mask)

            encodings.input_ids = encodings.input_ids.to(device)
            encodings.attention_mask = encodings.attention_mask.to(device)
            position_ids = model_inputs['position_ids'].to(device)
            encodings.token_type_ids = encodings.token_type_ids.to(device)
            target_ids = encodings.input_ids.clone()
            target_ids[encodings.token_type_ids == 0] = -100
            # print(batch_text)
            # print(encodings)
            # print(target_ids)
            # import pdb; pdb.set_trace()

            outputs = model(encodings.input_ids, attention_mask=encodings.attention_mask, position_ids=position_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_token_nll_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            per_token_nll_loss = per_token_nll_loss.view(-1, outputs.logits.size(1) - 1)
            neg_log_likelihood = per_token_nll_loss.sum(1)
            batch_lens = encodings.token_type_ids.sum(1)  # if add_prefix else encodings.attention_mask[..., 1:].sum(1)
            ppl = torch.exp(neg_log_likelihood / batch_lens).detach().cpu().numpy()

            for b_ctr, b_text in enumerate(batch_text):
                ex_out = {'text': b_text[0], 'answer': b_text[1], 'preds': []}
                last_word_st = torch.nonzero(shift_labels[b_ctr] != -100)[0]
                for pos_id in range(last_word_st, shift_logits.shape[1]):
                    ex_out['preds'].append(tokenizer.convert_ids_to_tokens(torch.argsort(shift_logits[b_ctr, pos_id])[-10:]))
                output_log.append(ex_out)

            # if np.any(np.array(ppl) > tokenizer.vocab_size):
            #     import pdb; pdb.set_trace()
            neg_log_likelihood = neg_log_likelihood.detach().cpu().numpy()

        lls_arr.append(-neg_log_likelihood)
        ppl_arr.append(ppl)

    lls_arr = np.concatenate(lls_arr)
    ppl_arr = np.concatenate(ppl_arr)
    return lls_arr, ppl_arr, output_log


def run_lm(df, model, tokenizer, prefix, batch_size, device, space_after_prefix=True, other_args=None):
    def strip_db_id_from_query(q_, do_strip=False):
        if do_strip:
            q_parts = q_.split(':', 1)
            return q_parts[0].strip(), q_parts[1].strip()
        else:
            return '', q_

    def strip_schema_from_query(q_):
        return q_.split(' | ', 1)[0].strip()

    ppl_arr, lls_arr = [], []
    # TODO: Fix this ugly thing
    if other_args.task in {'snli', 'boolq'}:
        all_ll_queries = df.get_ll_query(with_label=other_args.with_label)
    query_col = TASK_QUERY_COLS.get(other_args.task, 'query')
    add_prefix = (not (prefix is None or prefix == '')) or (other_args.task == 'spider' and other_args.db_id_as_prefix) or (other_args.task in {'snli', 'boolq'})
    for st_idx in trange(0, len(df), batch_size):
        with torch.no_grad():
            # TODO: Fix this ugly thing
            if other_args.task not in {'snli', 'boolq'}:
                query_strings = df[st_idx:st_idx + batch_size][query_col].tolist()

            if other_args.task in {'snli', 'boolq'}:
                # prefixes = [f"Premise: {prefix_str} Hypothesis:"
                            # for prefix_str in df[st_idx:st_idx + batch_size]['premise'].tolist()]
                # prefixes = df[st_idx:st_idx + batch_size]['premise'].tolist()
                pass
            elif other_args.task == 'spider' and other_args.db_id_as_prefix:
                prefixes = [prefix + strip_db_id_from_query(q_str, True)[0] + ":"
                            for q_str in query_strings]
            else:
                prefixes = [prefix] * len(query_strings)

            if other_args.task == 'spider' and other_args.strip_db_id:
                query_strings = [strip_db_id_from_query(q_str, other_args.strip_db_id)[1] for q_str in query_strings]
                query_strings = [strip_schema_from_query(q_str) for q_str in query_strings]

            if other_args.task in {'snli', 'boolq'}:
                batch_text = all_ll_queries[st_idx:st_idx + batch_size]
                batch_text = [(f"{tokenizer.bos_token}{prefix_str}", f"{query_str}{tokenizer.eos_token}")
                              for prefix_str, query_str in batch_text]
            elif not add_prefix:
                batch_text = query_strings
            else:
                batch_text = [[pref, ' ' + q_str if space_after_prefix else q_str]
                              for pref, q_str in zip(prefixes, query_strings)]

            if st_idx == 0:
                print("====== DEBUGGING: ======")
                print(batch_text)
                print("====== ======")
            encodings = tokenizer(batch_text, padding='longest', truncation=True, return_tensors='pt',
                                  return_token_type_ids=True)
            model_inputs = model.prepare_inputs_for_generation(input_ids=encodings.input_ids,
                                                               attention_mask=encodings.attention_mask)

            encodings.input_ids = encodings.input_ids.to(device)
            encodings.attention_mask = encodings.attention_mask.to(device)
            position_ids = model_inputs['position_ids'].to(device)
            encodings.token_type_ids = encodings.token_type_ids.to(device)
            target_ids = encodings.input_ids.clone()
            target_ids[encodings.token_type_ids == 0 if add_prefix else encodings.attention_mask == 0] = -100
            # print(encodings)
            # print(target_ids)
            # import pdb; pdb.set_trace()

            outputs = model(encodings.input_ids, attention_mask=encodings.attention_mask, position_ids=position_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_token_nll_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            per_token_nll_loss = per_token_nll_loss.view(-1, shift_logits.size(1))
            neg_log_likelihood = per_token_nll_loss.sum(1)
            batch_lens = encodings.token_type_ids.sum(1) if add_prefix else encodings.attention_mask[..., 1:].sum(1)
            ppl = torch.exp(neg_log_likelihood / batch_lens).detach().cpu().numpy()
            # if np.any(np.array(ppl) > tokenizer.vocab_size)
            #     import pdb; pdb.set_trace()
            neg_log_likelihood = neg_log_likelihood.detach().cpu().numpy()

        lls_arr.append(-neg_log_likelihood)
        ppl_arr.append(ppl)

    lls_arr = np.concatenate(lls_arr)
    ppl_arr = np.concatenate(ppl_arr)
    # assert len(lls_arr) == len(ppl_arr) == len(df)
    return lls_arr, ppl_arr


def main(args):
    """
    Reference: https://huggingface.co/transformers/perplexity.html
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.task == 'lambada' or args.task == 'openai_lambada':
        from datasets import load_dataset
        df_split = load_dataset('craffel/openai_lambada' if args.task == 'openai_lambada' else 'lambada', split='test')
        split_lls, split_ppls, split_log = run_lm_lambada(df_split, model, tokenizer, args.prefix, args.batch_size,
                                                          device)
        print(np.mean(split_lls))
        print(np.mean(split_ppls), np.max(split_ppls), np.min(split_ppls), np.std(split_ppls))
        np.save(os.path.join(args.output_dir, f'_loglikelihoods.npy'), split_lls)
        np.save(os.path.join(args.output_dir, f'_perplexity.npy'), split_ppls)
        with open(os.path.join(args.output_dir, f'prediction_output.log'), 'w') as fout:
            json.dump(split_log, fout, indent=2)
    elif args.task == 'wikitext-2-raw-v1' or args.task == 'wikitext-103-raw-v1':
        from datasets import load_dataset
        df_split = load_dataset('wikitext', args.task, split='test')
        split_text = "\n\n".join(df_split["text"])
        split_lls = run_lm_strided_wikitext(split_text, model, tokenizer, device)
        print(f"np.mean(lls): {np.mean(split_lls):0.4f}")
        print(f"PPL: {np.exp(-np.mean(split_lls)):0.4f}")
        np.save(os.path.join(args.output_dir, f'_loglikelihoods.npy'), split_lls)
    else:
        if args.task in {'snli', 'boolq'}:
            task_files = TASK_FILES[args.task]['processed']
        else:
            task_files = TASK_FILES[args.task]
        for split in task_files:
            df_split = TASK_PROCESSOR[args.task](args.data_dir, split)
            split_lls, split_ppls = run_lm(df_split, model, tokenizer, args.prefix, args.batch_size, device,
                                           args.space_after_prefix, args)
            print(f"np.mean(lls): {np.mean(split_lls)}\nnp.median(lls): {np.median(split_lls)}")
            print(f"np.mean(ppls): {np.mean(split_ppls)}, "
                  f"np.max(ppls): {np.max(split_ppls)}, "
                  f"np.min(ppls): {np.min(split_ppls)}, "
                  f"np.std(ppls): {np.std(split_ppls)}")
            np.save(os.path.join(args.output_dir,
                                 f'{split}_{args.prefix}_{args.space_after_prefix}_loglikelihoods.npy'), split_lls)
            np.save(os.path.join(args.output_dir,
                                 f'{split}_{args.prefix}_{args.space_after_prefix}_perplexity.npy'), split_ppls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--prefix", type=str, default='\n')  # '<|endoftext|>'
    parser.add_argument("--space_after_prefix", action='store_true')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_name_or_path", type=str, default='gpt2-large')
    # SPIDER Specific Args
    parser.add_argument("--db_id_as_prefix", action='store_true')
    parser.add_argument("--strip_db_id", action='store_true')
    # SNLI Specific Args
    parser.add_argument("--with_label", action='store_true')
    args_ = parser.parse_args()

    print(vars(args_))
    main(args_)
