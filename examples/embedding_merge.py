import torch
import transformers
import datasets
import math
import random
import copy


def few_shot_embedding_icl(model, tokenizer, prompts, input_text, few_shot_dataset=None, verbalizer=None, n_shot=0, max_seq_length=4096, embed_model=None, projector=None, embedding_matrix=None, **kwargs):
    # This function is used to merge the few shot prompts
    # The model is assumed to be a hf model, with def get_input_embeddings
    assert n_shot > 0, "Number of shots should be greater than 0"
    assert embed_model is not None, "Embed model should be provided"

    embeddings = model.get_input_embeddings()

    if tokenizer.pad_token is None:
        print("WARNING: Replacing PAD Token with EOS Token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # tokenizer bos token
    if tokenizer.bos_token is not None:
        final_representation = [embeddings(torch.tensor(tokenizer.bos_token_id).to(model.device)).unsqueeze(0)]
    else:
        final_representation = []

    main_text_lst = []
    main_label_lst = []
    for data_class in verbalizer.keys():
        sampled_few_shot_dataset = few_shot_dataset.filter(lambda x: x['label'] == verbalizer[data_class])
        sampled_few_shot_dataset = sampled_few_shot_dataset.shuffle()
        sampled_few_shot_dataset = sampled_few_shot_dataset.select(range(n_shot))
        for few_shot_ctr, example in enumerate(sampled_few_shot_dataset):
            main_text_lst.append(example['text'].strip())
            main_label_lst.append(example['label'])


    # randomize the order
    idx_lst = list(range(len(main_text_lst)))
    random.shuffle(idx_lst)
    main_text_lst = [main_text_lst[x] for x in idx_lst]
    main_label_lst = [main_label_lst[x] for x in idx_lst]

    main_text_embeds = embed_model.encode(main_text_lst, max_batch_size=1).squeeze(1).to(model.device).type(model.dtype)
    main_text_embeds = projector(main_text_embeds)
    sep_representation = embeddings(torch.tensor(tokenizer("\n", truncation=False, padding=False, add_special_tokens=False)['input_ids']).to(model.device))
    pre_indicator_embeds = embeddings(torch.tensor(tokenizer("(", padding=False, truncation=False, add_special_tokens=False)['input_ids']).to(model.device))
    post_indicator_embeds = embeddings(torch.tensor(tokenizer(")'s sentiment is", padding=False, truncation=False, add_special_tokens=False)['input_ids']).to(model.device))
    for idx, main_label in enumerate(main_label_lst):
        label_emb = embeddings(torch.tensor(tokenizer(main_label, truncation=False, padding=False, add_special_tokens=False)['input_ids']).to(model.device))
        # example -> label \n
        final_representation.append(pre_indicator_embeds)
        final_representation.append(main_text_embeds[idx,:].unsqueeze(0))
        final_representation.append(post_indicator_embeds)
        final_representation.append(label_emb)
        final_representation.append(sep_representation)
    
    # add the input text
    final_representation = torch.cat(final_representation, dim=0)


    # Now add the batch dim
    final_representation = final_representation.unsqueeze(0)    
    return final_representation, None, None


def few_shot_embedding_icl_generation(model, tokenizer, few_shot_dataset=None, n_shot=0, max_seq_length=4096, embed_model=None, projector=None, prefix=None, postfix=None,  **kwargs):
    # This function is used to merge the few shot prompts
    # The model is assumed to be a hf model, with def get_input_embeddings
    # text -> label
    assert n_shot > 0, "Number of shots should be greater than 0"
    assert embed_model is not None, "Embed model should be provided"
    assert (prefix is not None) and (postfix is not None), "Prefix and Postfix should be provided"

    embeddings = model.get_input_embeddings()

    if tokenizer.pad_token is None:
        print("WARNING: Replacing PAD Token with EOS Token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # tokenizer bos token
    if tokenizer.bos_token is not None:
        final_representation = [embeddings(torch.tensor(tokenizer.bos_token_id).to(model.device)).unsqueeze(0)]
    else:
        final_representation = []

    main_text_lst = []
    main_label_lst = []

    few_shot_dataset = few_shot_dataset.shuffle()
    few_shot_dataset = few_shot_dataset.select(range(n_shot))
    for few_shot_ctr, example in enumerate(few_shot_dataset):
        main_text_lst.append(example['text'])
        main_label_lst.append(example['label'])


    # randomize the order
    idx_lst = list(range(len(main_text_lst)))
    random.shuffle(idx_lst)
    main_text_lst = [main_text_lst[x] for x in idx_lst]
    main_label_lst = [main_label_lst[x] for x in idx_lst]

    main_text_embeds = embed_model.encode(main_text_lst, max_batch_size=1).squeeze(1).to(model.device).type(model.dtype)
    main_text_embeds = projector(main_text_embeds)
    sep_representation = embeddings(torch.tensor(tokenizer("\n", truncation=False, padding=False, add_special_tokens=False)['input_ids']).to(model.device))
    pre_indicator_embeds = embeddings(torch.tensor(tokenizer(prefix, padding=False, truncation=False, add_special_tokens=False)['input_ids']).to(model.device))
    post_indicator_embeds = embeddings(torch.tensor(tokenizer(postfix, padding=False, truncation=False, add_special_tokens=False)['input_ids']).to(model.device))
    for idx, main_label in enumerate(main_label_lst):
        label_emb = embeddings(torch.tensor(tokenizer(main_label, truncation=False, padding=False, add_special_tokens=False)['input_ids']).to(model.device))
        # example -> label \n
        final_representation.append(pre_indicator_embeds)
        final_representation.append(main_text_embeds[idx,:].unsqueeze(0))
        final_representation.append(post_indicator_embeds)
        final_representation.append(label_emb)
        final_representation.append(sep_representation)
    
    # add the input text
    final_representation = torch.cat(final_representation, dim=0)

    # Now add the batch dim
    final_representation = final_representation.unsqueeze(0)    
    return final_representation, None, None


def few_shot_embedding_icl_generation_baseline(model, tokenizer, few_shot_dataset=None, n_shot=0, max_seq_length=4096, embed_model=None, projector=None, prefix=None, postfix=None,  **kwargs):
    # This function is used to merge the few shot prompts
    # The model is assumed to be a hf model, with def get_input_embeddings
    # text -> label
    assert n_shot > 0, "Number of shots should be greater than 0"
    assert (prefix is not None) and (postfix is not None), "Prefix and Postfix should be provided"

    embeddings = model.get_input_embeddings()

    if tokenizer.pad_token is None:
        print("WARNING: Replacing PAD Token with EOS Token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # tokenizer bos token
    if tokenizer.bos_token is not None:
        final_representation = [embeddings(torch.tensor(tokenizer.bos_token_id).to(model.device)).unsqueeze(0)]
    else:
        final_representation = []

    main_text_lst = []
    main_label_lst = []

    few_shot_dataset = few_shot_dataset.shuffle()
    few_shot_dataset = few_shot_dataset.select(range(n_shot))
    for few_shot_ctr, example in enumerate(few_shot_dataset):
        main_text_lst.append(example['text'])
        main_label_lst.append(example['label'])


    # randomize the order
    idx_lst = list(range(len(main_text_lst)))
    random.shuffle(idx_lst)
    main_text_lst = [main_text_lst[x] for x in idx_lst]
    main_label_lst = [main_label_lst[x] for x in idx_lst]

    # shape = [k_shot + 1, Hidden_size]
    main_text_embeds = []
    with torch.no_grad():
        for enc_batch in range(len(main_text_lst)):
            main_text_embeds.append(model.get_input_embeddings()(torch.tensor(tokenizer(main_text_lst[enc_batch], truncation=False, padding=False, add_special_tokens=False)['input_ids']).type(torch.LongTensor).to(model.device)))
        main_text_embeds = torch.cat(main_text_embeds, dim=0)
    sep_representation = embeddings(torch.tensor(tokenizer("\n", truncation=False, padding=False, add_special_tokens=False)['input_ids']).to(model.device))
    pre_indicator_embeds = embeddings(torch.tensor(tokenizer(prefix, padding=False, truncation=False, add_special_tokens=False)['input_ids']).to(model.device))
    post_indicator_embeds = embeddings(torch.tensor(tokenizer(postfix, padding=False, truncation=False, add_special_tokens=False)['input_ids']).to(model.device))
    for idx, main_label in enumerate(main_label_lst):
        label_emb = embeddings(torch.tensor(tokenizer(main_label, truncation=False, padding=False, add_special_tokens=False)['input_ids']).to(model.device))
        # example -> label \n
        final_representation.append(pre_indicator_embeds)
        final_representation.append(main_text_embeds[idx,:].unsqueeze(0))
        final_representation.append(post_indicator_embeds)
        final_representation.append(label_emb)
        final_representation.append(sep_representation)
    
    # add the input text
    final_representation = torch.cat(final_representation, dim=0)
    # Now add the batch dim
    final_representation = final_representation.unsqueeze(0)    
    return final_representation, None, None


def few_shot_embedding_icl_brain(model, tokenizer, few_shot_dataset=None, n_shot=0, max_seq_length=4096, embed_model=None, projector=None, postfix="Response: ", **kwargs):
    # This function is used to merge the few shot prompts
    # The model is assumed to be a hf model, with def get_input_embeddings
    # text -> label
    assert n_shot > 0, "Number of shots should be greater than 0"

    embeddings = model.get_input_embeddings()

    if tokenizer.pad_token is None:
        print("WARNING: Replacing PAD Token with EOS Token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # tokenizer bos token
    if tokenizer.bos_token is not None:
        final_representation = [embeddings(torch.tensor(tokenizer.bos_token_id).to(model.device)).unsqueeze(0)]
    else:
        final_representation = []
    
    main_instruction_lst = []
    main_rep_lst = []
    main_text_lst = []

    few_shot_dataset = few_shot_dataset.shuffle()
    few_shot_dataset = few_shot_dataset.select(range(n_shot))
    for few_shot_ctr, example in enumerate(few_shot_dataset):
        
        main_instruction_lst.append(example['instruction'] + " Text: ")
        main_rep_lst.append(example['representation'])
        main_text_lst.append(postfix + example['text'])


    # randomize the order
    idx_lst = list(range(len(main_text_lst)))
    random.shuffle(idx_lst)
    main_instruction_lst = [main_instruction_lst[x] for x in idx_lst]
    main_rep_lst = [main_rep_lst[x] for x in idx_lst]
    main_text_lst = [main_text_lst[x] for x in idx_lst]

    # shape = [k_shot, Hidden_size]
    main_text_embeds = torch.tensor(main_rep_lst).to(projector.device)
    main_text_embeds = projector(main_text_embeds)

    sep_representation = embeddings(torch.tensor(tokenizer("\n", truncation=False, padding=False, add_special_tokens=False)['input_ids']).to(model.device))
    
    
    for idx, (main_instruction, main_text) in enumerate(zip(main_instruction_lst, main_text_lst)):
        prefix, postfix = main_instruction, main_text
        pre_indicator_embeds = embeddings(torch.tensor(tokenizer(prefix, padding=False, truncation=False, add_special_tokens=False)['input_ids']).to(model.device))
        label_emb = embeddings(torch.tensor(tokenizer(postfix, padding=False, truncation=False, add_special_tokens=False)['input_ids']).to(model.device))
        # example -> label \n
        final_representation.append(pre_indicator_embeds)
        final_representation.append(main_text_embeds[idx,:].unsqueeze(0))
        final_representation.append(label_emb)
        final_representation.append(sep_representation)
    
    # add the input text
    final_representation = torch.cat(final_representation, dim=0)
    # Now add the batch dim
    final_representation = final_representation.unsqueeze(0)
    
    return final_representation, None, None
    


def few_shot_embedding_icl_timeseries(model, tokenizer, few_shot_dataset=None, n_shot=0, max_seq_length=4096, embed_model=None, projector=None, prefix=None, postfix=None,  **kwargs):
    # This function is used to merge the few shot prompts
    # The model is assumed to be a hf model, with def get_input_embeddings
    # text -> label
    assert n_shot > 0, "Number of shots should be greater than 0"
    assert embed_model is not None, "Embed model should be provided"
    assert (prefix is not None) and (postfix is not None), "Prefix and Postfix should be provided"

    embeddings = model.get_input_embeddings()

    if tokenizer.pad_token is None:
        print("WARNING: Replacing PAD Token with EOS Token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # tokenizer bos token
    if tokenizer.bos_token is not None:
        final_representation = [embeddings(torch.tensor(tokenizer.bos_token_id).to(model.device)).unsqueeze(0)]
    else:
        final_representation = []

    main_text_lst = []
    main_label_lst = []
    main_graph_rep_lst = []
    label_dict = {0: 'negative.', 1: 'positive.'}

    for bctr, batch in enumerate(few_shot_dataset):
        with torch.no_grad():
            #batch = {k: v.to(embed_model.model.device) for k, v in batch.items()}
            outputs = embed_model.transform(batch['features'])
            main_graph_rep_lst.append(outputs)
            main_label_lst.append(label_dict[batch['labels'].detach().cpu().item()])
        if bctr > n_shot:
            break

    # shape = [k_shot, Hidden_size]

    # randomize the order
    idx_lst = list(range(len(main_graph_rep_lst)))
    random.shuffle(idx_lst)
    main_graph_rep_lst = [main_graph_rep_lst[x] for x in idx_lst]
    main_label_lst = [main_label_lst[x] for x in idx_lst]


    main_text_embeds = []
    with torch.no_grad():
        main_text_embeds = torch.cat(main_graph_rep_lst, dim=0)
        main_text_embeds = projector(main_text_embeds.to(projector.device))
    sep_representation = embeddings(torch.tensor(tokenizer("\n", truncation=False, padding=False, add_special_tokens=False)['input_ids']).to(model.device))
    pre_indicator_embeds = embeddings(torch.tensor(tokenizer(prefix, padding=False, truncation=False, add_special_tokens=False)['input_ids']).to(model.device))
    post_indicator_embeds = embeddings(torch.tensor(tokenizer(postfix, padding=False, truncation=False, add_special_tokens=False)['input_ids']).to(model.device))
    for idx, main_label in enumerate(main_label_lst):
        label_emb = embeddings(torch.tensor(tokenizer(main_label, truncation=False, padding=False, add_special_tokens=False)['input_ids']).to(model.device))
        # example -> label \n
        final_representation.append(pre_indicator_embeds)
        final_representation.append(main_text_embeds[idx,:].unsqueeze(0))
        final_representation.append(post_indicator_embeds)
        final_representation.append(label_emb)
        final_representation.append(sep_representation)
    
    # add the input text
    final_representation = torch.cat(final_representation, dim=0)

    # Now add the batch dim
    final_representation = final_representation.unsqueeze(0)    
    return final_representation, None, None


def few_shot_embedding_icl_graph(model, tokenizer, few_shot_dataset=None, n_shot=0, max_seq_length=4096, embed_model=None, projector=None, prefix=None, postfix=None,  **kwargs):
    # This function is used to merge the few shot prompts
    # The model is assumed to be a hf model, with def get_input_embeddings
    # text -> label
    assert n_shot > 0, "Number of shots should be greater than 0"
    assert embed_model is not None, "Embed model should be provided"
    assert (prefix is not None) and (postfix is not None), "Prefix and Postfix should be provided"

    embeddings = model.get_input_embeddings()

    if tokenizer.pad_token is None:
        print("WARNING: Replacing PAD Token with EOS Token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # tokenizer bos token
    if tokenizer.bos_token is not None:
        final_representation = [embeddings(torch.tensor(tokenizer.bos_token_id).to(model.device)).unsqueeze(0)]
    else:
        final_representation = []

    main_text_lst = []
    main_label_lst = []
    main_graph_rep_lst = []
    label_dict = {0: 'negative.', 1: 'positive.'}

    for bctr, batch in enumerate(few_shot_dataset):
        with torch.no_grad():
            batch = {k: v.to(embed_model.device) for k, v in batch.items()}
            outputs = embed_model(**batch)['last_hidden_state'][:,0,:]
            main_graph_rep_lst.append(outputs)
            main_label_lst.append(label_dict[batch['labels'].detach().cpu().item()])
            #main_label_lst.append(str(batch['labels'].detach().cpu().item()))
        if bctr > n_shot:
            break

    # randomize the order
    idx_lst = list(range(len(main_graph_rep_lst)))
    random.shuffle(idx_lst)
    main_graph_rep_lst = [main_graph_rep_lst[x] for x in idx_lst]
    main_label_lst = [main_label_lst[x] for x in idx_lst]


    main_text_embeds = []
    with torch.no_grad():
        main_text_embeds = torch.cat(main_graph_rep_lst, dim=0)
        main_text_embeds = projector(main_text_embeds)
    sep_representation = embeddings(torch.tensor(tokenizer("\n", truncation=False, padding=False, add_special_tokens=False)['input_ids']).to(model.device))
    pre_indicator_embeds = embeddings(torch.tensor(tokenizer(prefix, padding=False, truncation=False, add_special_tokens=False)['input_ids']).to(model.device))
    post_indicator_embeds = embeddings(torch.tensor(tokenizer(postfix, padding=False, truncation=False, add_special_tokens=False)['input_ids']).to(model.device))
    for idx, main_label in enumerate(main_label_lst):
        label_emb = embeddings(torch.tensor(tokenizer(main_label, truncation=False, padding=False, add_special_tokens=False)['input_ids']).to(model.device))
        # example -> label \n
        final_representation.append(pre_indicator_embeds)
        final_representation.append(main_text_embeds[idx,:].unsqueeze(0))
        final_representation.append(post_indicator_embeds)
        final_representation.append(label_emb)
        final_representation.append(sep_representation)
    
    # add the input text
    final_representation = torch.cat(final_representation, dim=0)

    # Now add the batch dim
    final_representation = final_representation.unsqueeze(0)    
    return final_representation, None, None


def few_shot_embedding_icl_mid(model, tokenizer, prompts, input_text, few_shot_dataset=None, verbalizer=None, n_shot=0, max_seq_length=4096, embed_model=None, projector=None, embedding_matrix=None, **kwargs):
    # This function is used to merge the few shot prompts
    # The model is assumed to be a hf model, with def get_input_embeddings
    assert n_shot > 0, "Number of shots should be greater than 0"
    assert embed_model is not None, "Embed model should be provided"

    embeddings = model.get_input_embeddings()

    if tokenizer.pad_token is None:
        print("WARNING: Replacing PAD Token with EOS Token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # tokenizer bos token
    if tokenizer.bos_token is not None:
        final_representation = [embeddings(torch.tensor(tokenizer.bos_token_id).to(model.device)).unsqueeze(0)]
    else:
        final_representation = []

    last_states = embed_model.encode(prompts, max_length=max_seq_length).to(model.device).type(model.dtype).detach()
    last_states = projector(last_states).mean(dim=0, keepdim=True)
    main_text_lst = []
    main_label_lst = []
    for data_class in verbalizer.keys():
        sampled_few_shot_dataset = few_shot_dataset.filter(lambda x: x['label'] == data_class)
        sampled_few_shot_dataset = sampled_few_shot_dataset.shuffle()
        sampled_few_shot_dataset = sampled_few_shot_dataset.select(range(n_shot))
        for few_shot_ctr, example in enumerate(sampled_few_shot_dataset):
            main_text_lst.append(example['text'].strip())
            main_label_lst.append(verbalizer[example['label']])


    # randomize the order
    idx_lst = list(range(len(main_text_lst)))
    random.shuffle(idx_lst)
    main_text_lst = [main_text_lst[x] for x in idx_lst]
    main_label_lst = [main_label_lst[x] for x in idx_lst]


    # now the final step
    input_representation = embeddings(torch.tensor(tokenizer(input_text, truncation=False, padding=False, add_special_tokens=False)['input_ids']).to(model.device))

    sep_representation = embeddings(torch.tensor(tokenizer("\n", truncation=False, padding=False, add_special_tokens=False)['input_ids']).to(model.device))
    for idx, main_label in enumerate(main_label_lst):
        main_text = main_text_lst[idx]
        text_emb = embeddings(torch.tensor(tokenizer(main_text, truncation=False, padding=False, add_special_tokens=False)['input_ids']).to(model.device))
        label_emb = embeddings(torch.tensor(tokenizer(main_label, truncation=False, padding=False, add_special_tokens=False)['input_ids']).to(model.device))
        # example -> label \n
        final_representation.append(text_emb)
        final_representation.append(last_states)
        final_representation.append(label_emb)
        final_representation.append(sep_representation)
    
    # add the input text
    final_representation.append(input_representation)
    final_representation.append(last_states)
    final_representation = torch.cat(final_representation, dim=0)

    # Now add the batch dim
    final_representation = final_representation.unsqueeze(0)    
    return final_representation, None, None



def encode_digits_to_one_hot(digits=128):
    # for each decimal place in the number, create a one hot vector
    digits = int(digits)
    one_hot_proj = torch.eye(10)
    output = []
    for digit in str(digits).zfill(10):
        output.append(one_hot_proj[int(digit)])
    output = torch.cat(output, dim=0) # shape = (len(digits), 10)
    output = output.view(1, -1) # shape = (1, len(digits)*10)
    return output


def few_shot_embedding_icl_digits(model, tokenizer, few_shot_dataset=None, n_shot=0, max_seq_length=4096, embed_model=None, projector=None, postfix="Response: ", one_hot_proj=None, **kwargs):
    # This function is used to merge the few shot prompts
    # The model is assumed to be a hf model, with def get_input_embeddings
    # text -> label
    assert n_shot > 0, "Number of shots should be greater than 0"

    embeddings = model.get_input_embeddings()

    if tokenizer.pad_token is None:
        print("WARNING: Replacing PAD Token with EOS Token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # tokenizer bos token
    if tokenizer.bos_token is not None:
        final_representation = [embeddings(torch.tensor(tokenizer.bos_token_id).to(model.device)).unsqueeze(0)]
    else:
        final_representation = []
    
    main_instruction_lst = []
    main_rep_lst = []
    main_text_lst = []

    few_shot_dataset = few_shot_dataset.shuffle()
    few_shot_dataset = few_shot_dataset.select(range(n_shot))
    for few_shot_ctr, example in enumerate(few_shot_dataset):
        
        main_instruction_lst.append(example['instruction'])
        main_rep_lst.append(example['representation'])
        main_text_lst.append(postfix + example['text'])


    # randomize the order
    idx_lst = list(range(len(main_text_lst)))
    random.shuffle(idx_lst)
    main_instruction_lst = [main_instruction_lst[x] for x in idx_lst]
    main_rep_lst = [main_rep_lst[x] for x in idx_lst]
    main_text_lst = [main_text_lst[x] for x in idx_lst]

    # shape = [k_shot, Hidden_size]

    sep_representation = embeddings(torch.tensor(tokenizer("\n", truncation=False, padding=False, add_special_tokens=False)['input_ids']).to(model.device))
    
    
    for idx, (main_instruction, main_text) in enumerate(zip(main_instruction_lst, main_text_lst)):
        # A=[REP0], B=[REP1], A+B is [TEXT]
        Aeq = embeddings(torch.tensor(tokenizer("A=", padding=False, truncation=False, add_special_tokens=False)['input_ids']).to(model.device))
        Beq = embeddings(torch.tensor(tokenizer("B=", padding=False, truncation=False, add_special_tokens=False)['input_ids']).to(model.device))

        AplusBeq_label_emb = embeddings(torch.tensor(tokenizer(main_text, padding=False, truncation=False, add_special_tokens=False)['input_ids']).to(model.device))
        # example -> label \n
        final_representation.append(Aeq)
        final_representation.append(projector(encode_digits_to_one_hot(main_rep_lst[idx][0]).to(projector.device).type(projector.dtype)))
        final_representation.append(Beq)
        final_representation.append(projector(encode_digits_to_one_hot(main_rep_lst[idx][1]).to(projector.device).type(projector.dtype)))

        final_representation.append(AplusBeq_label_emb)
        final_representation.append(sep_representation)
    
    # add the input text
    final_representation = torch.cat(final_representation, dim=0)

    # Now add the batch dim
    final_representation = final_representation.unsqueeze(0)
    
    return final_representation, None, None