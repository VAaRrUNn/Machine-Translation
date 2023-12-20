# def train(model,
#           criterion,
#           optim,
#           eng_loader,
#           hi_loader,
#           device,
#           scheduler,
#           epochs,
#           hi_vocab_size,
#           eng_tokenizer,
#           hi_tokenizer):

#     model.to(device)
#     model.train()

#     losses = []
#     iteration = 0

#     for epoch in tqdm(range(epochs)):
#         for (eng_sen, enc_mask), (hi_sen, dec_mask) in tqdm(zip(eng_loader, hi_loader)):

#             # Move tensors to respective device
#             eng_sen = eng_sen.to(device)
#             enc_mask = enc_mask.to(device)
#             hi_sen = hi_sen.to(device)
#             dec_mask = dec_mask.to(device)

#             out = model(enc_b = eng_sen,
#                         dec_b = hi_sen,
#                         enc_mask = enc_mask,
#                         dec_mask = dec_mask)

#             optim.zero_grad()
#             loss = criterion(out.view(-1, hi_vocab_size),
#                              hi_sen.view(-1))
#             loss.backward()
#             optim.step()

#             losses.append(loss.item())
#             iteration += 1

#             if iteration % 100 == 0:
#                 print(f"Loss: {losses[-1]}")
#                 print(f"English sen: {eng_tokenizer.decode(eng_sen[0].tolist())}")
#                 print(f"Target Hindi sen: {hi_tokenizer.decode(hi_sen[0].tolist())}")
#                 print(f"Predicted sentence: {hi_tokenizer.decode(torch.argmax(out, dim=-1)[0].tolist())}")

#     return losses


# def decoder_collate_fn(x, tokenizer):
#     """s
#     x: List of sentences
#     """

#     start_token_id, pad_token_id, end_token_id = tokenizer...
#     encodings = tokenizer.encode_batch(x)
#     L = len(encodings[0].ids)

#     input_tensor = torch.zeros((len(x), L), dtype=torch.long)
#     out_tensor = torch.zeros((len(x), L), dtype=torch.long)

#     mask = torch.zeros((len(x), L, L))
#     look_ahead_mask = torch.tril(torch.ones(L, L))
#     look_ahead_mask[look_ahead_mask == 0] = -torch.inf
#     look_ahead_mask[look_ahead_mask == 1] = 0

#     for i, enc in enumerate(encodings):
#         input_tensor[i] = torch.tensor(enc.ids)
#         out_tensor[i] = input_tensor[i]

#         # concat it.
#         input_tensor[i] = torch.concat[torch.tensor(), input_tensor[1:]]

#         for a in range(len(out_tensor[i].tolist())):
#             if out_tensor[i, a] == pad_token_id:
#                 out_tensor[i, a] = end_token_id
#                 break


#         pad_mask = create_pad_mask(enc.ids)
#         mask[i] = pad_mask + look_ahead_mask

#     return input_tensor, out_tensor, mask


def translate_sen(model,
                  eng_sen: str,
                  eng_tokenizer,
                  hi_tokenizer,
                  MAX_SEQ_LEN,
                  max_len,
                  end_token_id,
                  start_token_id,
                  pad_token_id,
                  device):

    model.to(device)
    model.eval()
    eng_sen.to(device)

    with torch.inference_mode():
        eng_sen_ids = eng_tokenizer.encode(eng_sen).ids
        eng_sen_ids = torch.tensor([eng_sen_ids])

        # Filling with PAD token
        hi_sen = torch.full((1, MAX_SEQ_LEN), pad_token_id)
        hi_sen[0, 0] = start_token_id

        # To keep track of where to put next token in hi_sen
        idx = 1

        trans_sen = ""
        next_token_id = -1

        while max_len != 0 and next_token_id != end_token_id:

            enc_mask = ...
            dec_mask = ...
            hi_sen_pred = model(enc_b=eng_sen,
                        dec_b=hi_sen,
                        enc_mask=enc_mask,
                        dec_mask=dec_mask)
            
            hi_sen_pred = hi_sen_pred[:, 0, :]
            next_token_id = torch.argmax(hi_sen_pred, dim=-1)

            trans_sen += hi_tokenizer.decode(next_token_id.tolist())
            hi_sen[0, idx] = next_token_id

            idx += 1
            max_len -= 1
            
        