from torch.nn import LayerNorm
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import random
from src.modules import (
    PositionalEncoding,
    CustomAttentionLayer,
    LossSoftmax,
    TagEmbedding,
    CopyGateModule,
)
from src.coderdict import POSDecoderDict
from src.generate_functions import idx_to_tensor
import pprint

class TransformerGenerationModel(nn.Module):
    def __init__(self, config=None, encoder_dict=None, decoder_dict=None):
        assert config is not None
        assert encoder_dict is not None and decoder_dict is not None
        super(TransformerGenerationModel, self).__init__()
        # Setting bs here, used to verify that dimensions are
        # correct when switching sequence and batch-size in tensors
        self.conf = config
        self.bs = config["bs"]
        self.tf_ratio = 1.0

        # Load dictionaries etc that don't use cuda
        self.pos_decoder_dict = POSDecoderDict(config)

        # Dimensions
        dim_glove = config["glove_embedding_size"]
        dim_hidden = config["hidden_size"]
        dim_heads = config["number_heads"]
        dim_dropout = config["dropout"]
        dim_layers = config["number_layers"]
        dim_feedforward = config["self_attention_dim_feedforward"]
        dim_encoder_dict = len(encoder_dict)
        dim_decoder_dict = len(decoder_dict)


        dim_was_copied_in = config["output_was_copied_tag_in"]
        dim_was_copied_out = config["output_was_copied_tag_out"]
        dim_decoder_pos_copy_in = config["decoder_part_of_speech_tag_in"]
        dim_decoder_pos_copy_out = config["decoder_part_of_speech_tag_out"]
        dim_decoder_pos_in = len(self.pos_decoder_dict)

        # Before Encoder Modules
        self.tag_embedding = TagEmbedding(config)
        dim_encoder_idx_out = dim_hidden - dim_glove - self.tag_embedding.out_size
        self.encoder_idx_embedding = nn.Embedding(
            dim_encoder_dict,
            dim_encoder_idx_out,
            padding_idx=encoder_dict.pad_token_idx,
        )
        self.encoder_pos_encoder = PositionalEncoding(d_model=dim_hidden, max_len=128)
        self.decoder_pos_encoder = PositionalEncoding(d_model=dim_hidden, max_len=128)

        # Before Decoder Modules
        self.decoder_previous_step_copy = nn.Embedding(
            dim_was_copied_in, dim_was_copied_out
        )
        self.decoder_pos_copy_embedding = nn.Embedding(
            dim_decoder_pos_copy_in, dim_decoder_pos_copy_out,
        )
        self.decoder_pos_embedding = nn.Embedding(
            dim_decoder_pos_in, dim_decoder_pos_copy_out
        )
        self.decoder_idx_embedding = nn.Embedding(
            dim_decoder_dict,
            dim_hidden
            - dim_glove
            - dim_was_copied_out
            - dim_decoder_pos_copy_out
            - dim_decoder_pos_copy_out,
            padding_idx=decoder_dict.pad_token_idx if decoder_dict else 0,
        )
        print(self.decoder_idx_embedding.num_embeddings)
        print(dim_hidden
            - dim_glove
            - dim_was_copied_out
            - dim_decoder_pos_copy_out
            - dim_decoder_pos_copy_out)# Encoder Modules
        encoder_layer = nn.TransformerEncoderLayer(dim_hidden,
                                                   dim_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dim_dropout)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, dim_layers, norm=LayerNorm(dim_hidden))

        # Decoder Modules
        self.tgt_mask = None
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim_hidden,
                                       dim_heads,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dim_dropout),
            dim_layers,
            norm=LayerNorm(dim_hidden))

        # After Decoder
        self.copy_gate = CopyGateModule(config=config)

        self.transformer_prob_copy_1 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim_hidden,
                                       8,
                                       dim_feedforward=2048,
                                       dropout=dim_dropout),
            1,
            norm=LayerNorm(dim_hidden),
        )
        self.transformer_prob_copy_2 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim_hidden,
                                       4,
                                       dim_feedforward=2048,
                                       dropout=dim_dropout),
            1,
            norm=LayerNorm(dim_hidden),
        )
        self.transformer_prob_copy_3 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim_hidden,
                                       2,
                                       dim_feedforward=2048,
                                       dropout=dim_dropout),
            1,
            norm=LayerNorm(dim_hidden),
        )
        self.transformer_prob_copy_last = CustomAttentionLayer(config, nhead=1)

        self.post_decoder = nn.Linear(dim_hidden, dim_decoder_dict)
        self.softmax = LossSoftmax()

        # Seed
        self.seed(seed=config["seed"])
        self.init_weights()

    def seed(self, seed=42):
        """
            Note: not verified if all these commands need to be run for proper seeding
        """
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def init_weights(self):
        init_range = 0.0001
        self.post_decoder.weight.data.uniform_(-init_range, init_range)
        self.post_decoder.bias.data.zero_()
        self.copy_gate.linear1.weight.data.uniform_(-init_range, init_range)
        self.copy_gate.linear1.bias.data.zero_()

    def sequence_to_first_dimension(self, tensor, batch=None):
        assert batch is not None
        assert tensor.shape[0] == batch.bs
        return tensor.transpose(0, 1).contiguous()

    def bs_to_first_dimension(self, tensor, batch=None):
        assert batch is not None
        assert tensor.shape[1] == batch.bs
        return tensor.transpose(0, 1).contiguous()

    def _generate_square_subsequent_mask(self, sz, to_cuda=True):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        assert isinstance(mask, torch.FloatTensor)
        return mask if not to_cuda else mask.cuda()

    def make_key_masks(self, batch, to_cuda=False):
        """
            Returns torch.BoolTensor masks
            True will mask
            False will not mask
        """
        src_key_mask = torch.ones(batch.src_idx.shape)
        tgt_key_mask = torch.ones(batch.tgt_input_idx.shape)
        for b in range(batch.bs):
            src_key_mask[b, : int(batch.src_len[b])] = torch.zeros(
                int(batch.src_len[b])
            )
            tgt_key_mask[b, : int(batch.tgt_len[b])] = torch.zeros(
                int(batch.tgt_len[b])
            )
        if to_cuda:
            return src_key_mask.bool().cuda(), tgt_key_mask.bool().cuda()
        else:
            return src_key_mask.bool(), tgt_key_mask.bool()

    def get_pre_encoder_block(self, batch=None, encoder_dict=None):
        assert batch is not None
        assert encoder_dict is not None
        # still don't understand why we multiply by square-root of encoder dictionary size
        src = self.encoder_idx_embedding(batch.src_idx)
        src = torch.cat((src, self.tag_embedding(batch)), -1)
        src = torch.cat((src, batch.src_emb), -1)
        return src

    def calc_idx_emb(self, tensor):
        assert int(torch.max(tensor)) < self.decoder_idx_embedding.num_embeddings
        return self.decoder_idx_embedding(tensor)

    def calc_previous_step_copy(self, tensor):
        assert int(torch.max(tensor)) < self.decoder_previous_step_copy.num_embeddings
        return self.decoder_previous_step_copy(tensor)

    def calc_tgt_pos_tag(self, tensor):
        assert int(torch.max(tensor)) < self.decoder_pos_embedding.num_embeddings
        return self.decoder_pos_embedding(tensor)

    def calc_pos_copy_emb(self, tensor):
        assert int(torch.max(tensor)) < self.decoder_pos_copy_embedding.num_embeddings
        return self.decoder_pos_copy_embedding(tensor)

    def get_pre_decoder_block_from_batch(self, batch=None):
        assert batch is not None
        return self.get_pre_decoder_block(
            idx_emb=batch.tgt_input_idx,
            prev_word_copied=batch.tgt_previous_word_copied,
            tgt_pos_tag=batch.tgt_pos_tag,
            tgt_copy_pos=batch.tgt_copy_pos,
            tgt_emb=batch.tgt_emb,
        )

    def get_decoder_masks(self, input=None, src_key_mask=None, tgt_key_mask=None):
        src_tgt_key_mask, tgt_src_key_mask = self.make_3D_key_mask(
            src_key_mask, tgt_key_mask, n_head=self.conf["number_heads"]
        )
        tgt_attn_mask = self._generate_square_subsequent_mask(input.shape[0])
        return tgt_attn_mask, src_tgt_key_mask

    def get_decoder_0(
        self,
        input=None,
        memory=None,
        src_key_mask=None,
        tgt_key_mask=None,
        to_cuda=True,
    ):
        assert input is not None
        assert memory is not None
        assert src_key_mask is not None
        assert tgt_key_mask is not None

        tgt_attn_mask = self._generate_square_subsequent_mask(
            input.shape[0], to_cuda=to_cuda
        )
        return self.transformer_decoder(
            input,
            memory,
            tgt_mask=tgt_attn_mask,
            memory_mask=None,
            memory_key_padding_mask=src_key_mask,
            tgt_key_padding_mask=tgt_key_mask,
        )

    def adjust_heads_for_key_mask(self, key_mask=None, number_heads=None, bs=None):
        assert key_mask.shape[0] <= bs * number_heads
        assert key_mask.shape[0] <= bs * number_heads
        return key_mask[0 : bs * number_heads]

    def get_prob_copy(
        self,
        batch=None,
        input=None,
        memory=None,
        src_key_mask=None,
        tgt_key_mask=None,
        to_cuda=True,
    ):

        tgt_attn_mask = self._generate_square_subsequent_mask(
            input.shape[0], to_cuda=to_cuda
        )
        src_key_mask = self.adjust_heads_for_key_mask(key_mask=src_key_mask, number_heads=8, bs=batch.bs)
        tgt_key_mask = self.adjust_heads_for_key_mask(key_mask=tgt_key_mask, number_heads=8, bs=batch.bs)

        prob_copy = self.transformer_prob_copy_1(
            input,
            memory,
            tgt_mask=tgt_attn_mask,
            memory_mask=None,
            memory_key_padding_mask=src_key_mask,
            tgt_key_padding_mask=tgt_key_mask,
        )
        src_key_mask = self.adjust_heads_for_key_mask(key_mask=src_key_mask, number_heads=4, bs=batch.bs)
        tgt_key_mask = self.adjust_heads_for_key_mask(key_mask=tgt_key_mask, number_heads=4, bs=batch.bs)

        prob_copy = self.transformer_prob_copy_2(
            prob_copy,
            memory,
            tgt_mask=tgt_attn_mask,
            memory_mask=None,
            memory_key_padding_mask=src_key_mask,
            tgt_key_padding_mask=tgt_key_mask,
        )
        src_key_mask = self.adjust_heads_for_key_mask(key_mask=src_key_mask, number_heads=2, bs=batch.bs)
        tgt_key_mask = self.adjust_heads_for_key_mask(key_mask=tgt_key_mask, number_heads=2, bs=batch.bs)

        prob_copy = self.transformer_prob_copy_3(
            prob_copy,
            memory,
            tgt_mask=tgt_attn_mask,
            memory_mask=None,
            memory_key_padding_mask=src_key_mask,
            tgt_key_padding_mask=tgt_key_mask,
        )
        src_key_mask = self.adjust_heads_for_key_mask(key_mask=src_key_mask, number_heads=1, bs=batch.bs)
        tgt_key_mask = self.adjust_heads_for_key_mask(key_mask=tgt_key_mask, number_heads=1, bs=batch.bs)

        _, prob_copy = self.transformer_prob_copy_last(
            prob_copy,
            memory,
            tgt_mask=tgt_attn_mask,
            memory_mask=None,
            memory_key_padding_mask=src_key_mask,
            tgt_key_padding_mask=tgt_key_mask,
        )

        prob_copy_mask = src_key_mask.unsqueeze(1).repeat(1, tgt_key_mask.shape[-1], 1)
        prob_copy_mask = torch.where(
            prob_copy_mask == False,
            torch.ones(prob_copy_mask.shape).float()
            if not to_cuda
            else torch.ones(prob_copy_mask.shape).float().cuda(),
            torch.zeros(prob_copy_mask.shape).float()
            if not to_cuda
            else torch.zeros(prob_copy_mask.shape).float().cuda(),
        )

        prob_copy = prob_copy * prob_copy_mask
        prob_copy = self.sequence_to_first_dimension(prob_copy, batch=batch)

        return prob_copy

    def get_loss_stats(
        self,
        batch=None,
        loss=None,
        logprob=None,
        target=None,
        copy_gate=None,
        decoder_dict=None,
        verbose=False,
    ):
        assert decoder_dict is not None
        accu = self.get_logprob_accu(
            logprob=logprob,
            target=target,
            decoder_dict=decoder_dict,
            verbose=verbose
        )
        denominator = torch.pow(batch.tgt_len.float(), -1)
        accu = float(torch.sum(accu.sum(0).squeeze() * denominator) / batch.bs)

        copy_accu = self.get_copy_accu(logprob, batch)
        gate_accu = self.get_gate_accu(copy_gate, batch)
        gate_level = self.get_gate_level(copy_gate, batch)

        loss_dict = {
            "loss": loss,
            "accu": accu,
            "copy_accu": copy_accu,
            "gate_accu": gate_accu,
            "gate_level": gate_level,
        }

        return loss_dict

    def get_pre_decoder_block(
        self,
        idx_emb=None,
        prev_word_copied=None,
        tgt_pos_tag=None,
        tgt_copy_pos=None,
        tgt_emb=None,
    ):
        return torch.cat(
            (
                self.calc_idx_emb(idx_emb),
                self.calc_previous_step_copy(prev_word_copied),
                self.calc_tgt_pos_tag(tgt_pos_tag),
                self.calc_pos_copy_emb(tgt_copy_pos),
                tgt_emb,
            ),
            -1,
        )

    def get_start_of_decoder_block(self, batch=None):
        assert batch is not None
        return self.get_pre_decoder_block(
            idx_emb=batch.tgt_input_idx[:, :1],
            prev_word_copied=batch.tgt_previous_word_copied[:, :1],
            tgt_pos_tag=batch.tgt_pos_tag[:, :1],
            tgt_copy_pos=batch.tgt_copy_pos[:, :1],
            tgt_emb=batch.tgt_emb[:, :1],
        )

    def generate_text(
        self,
        batch,
        config=None,
        encoder_dict=None,
        decoder_dict=None,
        loader=None,
        to_cuda=True,
    ):
        assert batch.bs == 1
        assert config is not None
        assert encoder_dict is not None
        assert decoder_dict is not None
        src_key_mask, tgt_key_mask = self.make_key_masks(batch, to_cuda=to_cuda)
        src = self.get_pre_encoder_block(batch=batch, encoder_dict=encoder_dict)
        src = self.sequence_to_first_dimension(src, batch=batch)
        src = self.encoder_pos_encoder(src)
        memory = self.transformer_encoder(src, src_key_padding_mask=src_key_mask)
        tmp_tgt_input = self.get_start_of_decoder_block(batch=batch)
        persistent_tgt_input = tmp_tgt_input.cpu()
        text = []
        for i in range(1, int(batch.tgt_len)):
            topk = self.generate_topk(
                tmp_tgt_input=tmp_tgt_input,
                batch=batch,
                memory=memory,
                src_key_mask=src_key_mask,
                tgt_key_mask=tgt_key_mask,
                to_cuda=to_cuda,
            )
            topk = topk[-1, :, 0]
            (
                idx_emb,
                copied,
                tgt_pos_tag,
                tgt_copy_pos_tag,
                tgt_emb,
                word_list,
                copy_word_list,
            ) = idx_to_tensor(
                topk.cpu(), batch=batch, decoder_dict=decoder_dict, loader=loader
            )
            generated_tmp_tgt_input = self.get_pre_decoder_block(
                idx_emb=idx_emb if not to_cuda else idx_emb.cuda(),
                prev_word_copied=copied.long() if not to_cuda else copied.long().cuda(),
                tgt_pos_tag=tgt_pos_tag if not to_cuda else tgt_pos_tag.cuda(),
                tgt_copy_pos=tgt_copy_pos_tag
                if not to_cuda
                else tgt_copy_pos_tag.cuda(),
                tgt_emb=tgt_emb if not to_cuda else tgt_emb.cuda(),
            )
            persistent_tgt_input = torch.cat(
                (persistent_tgt_input, generated_tmp_tgt_input.cpu()), dim=1
            )
            # put input for next step on the GPU
            tmp_tgt_input = (
                persistent_tgt_input if not to_cuda else persistent_tgt_input.cuda()
            )
            word = word_list[0] if word_list[0] is not None else copy_word_list[0]
            text.append(word)
        return text

    def generate_topk(
        self,
        tmp_tgt_input=None,
        batch=None,
        memory=None,
        src_key_mask=None,
        tgt_key_mask=None,
        to_cuda=True,
    ):

        generated_len = tmp_tgt_input.shape[1]
        tgt_input = self.sequence_to_first_dimension(tmp_tgt_input, batch=batch)
        tgt_input = self.decoder_pos_encoder(tgt_input)
        output_0 = self.get_decoder_0(
            input=tgt_input,
            memory=memory,
            src_key_mask=src_key_mask,
            tgt_key_mask=tgt_key_mask[:, :generated_len],
            to_cuda=to_cuda,
        )

        prob_generate = self.post_decoder(output_0)
        prob_copy = self.get_prob_copy(
            batch=batch,
            input=output_0,
            memory=memory,
            src_key_mask=src_key_mask,
            tgt_key_mask=tgt_key_mask[:, :generated_len],
            to_cuda=to_cuda,
        )
        copy_gate = self.copy_gate(output_0)

        # merge probabilities
        assert prob_generate.shape[0:1] == copy_gate.shape[0:1]
        assert prob_copy.shape[0:1] == copy_gate.shape[0:1]
        prob_generate = prob_generate * (1 - copy_gate)
        prob_copy = prob_copy * copy_gate
        output = torch.cat((prob_generate, prob_copy), -1)
        # probabilities of decoder indexes
        logprob = self.eps_log_softmax(output, dim=2)
        _, topk = torch.topk(logprob, 1)
        return topk

    def get_generate_slice_map(self, generate_map: list):
        generate_slice_map = []
        start = 0
        end = 1
        for i, generate in enumerate(generate_map):
            if generate:
                if start < i:
                    generate_slice_map.append(slice(start, end))
                generate_slice_map.append(i)
                start = i + 1
                end = start + 1
            else:
                end = i + 1
        if start != len(generate_map):
            generate_slice_map.append(slice(start, end))
        return generate_slice_map

    def generate_decoder_input(
        self,
        batch,
        memory=None,
        decoder_dict=None,
        loader=None,
        src_key_mask=None,
        tgt_key_mask=None,
    ):
        self.training = False
        teacher_forcing_cutoff = self.tf_ratio

        # create map for generation
        generate_map = [
            1 if (random.random() > teacher_forcing_cutoff) else 0
            for _ in range(int(max(batch.tgt_len)))
        ]
        slice_map = self.get_generate_slice_map(generate_map)

        # TODO: verify somehow that the generated parts and the parts taken from batch are concatenated in a correct way
        with torch.no_grad():
            s = slice_map[0] if (type(slice_map[0]) == slice) else slice(0, 1)
            # setting the persistent input to be on the cpu to clear the tmp to save GPU-RAM
            accumulated_idx_emb = batch.tgt_input_idx[:, s].cpu()
            accumulated_copied = batch.tgt_previous_word_copied[:, s].long().cpu()
            accumulated_tgt_pos_tag = batch.tgt_pos_tag[:, s].cpu()
            accumulated_tgt_copy_pos = batch.tgt_copy_pos[:, s].cpu()
            accumulated_tgt_emb = batch.tgt_emb[:, s].cpu()
            tmp_tgt_input = self.get_pre_decoder_block(
                idx_emb=accumulated_idx_emb.cuda(),
                prev_word_copied=accumulated_copied.long().cuda(),
                tgt_pos_tag=accumulated_tgt_pos_tag.cuda(),
                tgt_copy_pos=accumulated_tgt_copy_pos.cuda(),
                tgt_emb=accumulated_tgt_emb.cuda(),
            )
            accumulated_tgt_input = tmp_tgt_input.cpu()
            for s in slice_map[1:]:
                if type(s) == int:
                    # if we encounter an index we generate the next input
                    topk = self.generate_topk(
                        tmp_tgt_input=tmp_tgt_input,
                        batch=batch,
                        memory=memory,
                        src_key_mask=src_key_mask,
                        tgt_key_mask=tgt_key_mask,
                    )
                    # clear memory, not certain if this actually helps
                    del tmp_tgt_input
                    torch.cuda.empty_cache()

                    # use only last generated indexes
                    topk = topk[-1, :, 0]
                    (
                        idx_emb,
                        copied,
                        tgt_pos_tag,
                        tgt_copy_pos_tag,
                        tgt_emb,
                        word_list,
                        copy_word_list,
                    ) = idx_to_tensor(
                        topk.cpu(),
                        batch=batch,
                        decoder_dict=decoder_dict,
                        loader=loader,
                    )
                    accumulated_idx_emb = torch.cat(
                        (accumulated_idx_emb, idx_emb.cpu()), 1
                    )
                    accumulated_copied = torch.cat(
                        (accumulated_copied, copied.long().cpu()), 1
                    )
                    accumulated_tgt_pos_tag = torch.cat(
                        (accumulated_tgt_pos_tag, tgt_pos_tag.cpu()), 1
                    )
                    accumulated_tgt_copy_pos = torch.cat(
                        (accumulated_tgt_copy_pos, tgt_copy_pos_tag.cpu()), 1
                    )
                    accumulated_tgt_emb = torch.cat(
                        (accumulated_tgt_emb, tgt_emb.cpu()), 1
                    )

                    generated_tmp_tgt_input = self.get_pre_decoder_block(
                        idx_emb=idx_emb.cuda(),
                        prev_word_copied=copied.long().cuda(),
                        tgt_pos_tag=tgt_pos_tag.cuda(),
                        tgt_copy_pos=tgt_copy_pos_tag.cuda(),
                        tgt_emb=tgt_emb.cuda(),
                    )
                else:
                    # TODO: assert that this is the correct index we get from batch when running teacher-forcing
                    idx_emb = batch.tgt_input_idx[:, s]
                    copied = batch.tgt_previous_word_copied[:, s]
                    tgt_pos_tag = batch.tgt_pos_tag[:, s]
                    tgt_copy_pos = batch.tgt_copy_pos[:, s]
                    tgt_emb = batch.tgt_emb[:, s]
                    generated_tmp_tgt_input = self.get_pre_decoder_block(
                        idx_emb=idx_emb,
                        prev_word_copied=copied,
                        tgt_pos_tag=tgt_pos_tag,
                        tgt_copy_pos=tgt_copy_pos,
                        tgt_emb=tgt_emb,
                    )
                    accumulated_idx_emb = torch.cat(
                        (accumulated_idx_emb, idx_emb.cpu()), 1
                    )
                    accumulated_copied = torch.cat(
                        (accumulated_copied, copied.long().cpu()), 1
                    )
                    accumulated_tgt_pos_tag = torch.cat(
                        (accumulated_tgt_pos_tag, tgt_pos_tag.cpu()), 1
                    )
                    accumulated_tgt_copy_pos = torch.cat(
                        (accumulated_tgt_copy_pos, tgt_copy_pos.cpu()), 1
                    )
                    accumulated_tgt_emb = torch.cat(
                        (accumulated_tgt_emb, tgt_emb.cpu()), 1
                    )

                # add new input to previous input
                accumulated_tgt_input = torch.cat(
                    (accumulated_tgt_input, generated_tmp_tgt_input.cpu()), dim=1
                )

                # put input for next step on the GPU
                tmp_tgt_input = accumulated_tgt_input.cuda()

        self.training = True
        generated_tgt_input = self.get_pre_decoder_block(
            idx_emb=accumulated_idx_emb.cuda(),
            prev_word_copied=accumulated_copied.cuda(),
            tgt_pos_tag=accumulated_tgt_pos_tag.cuda(),
            tgt_copy_pos=accumulated_tgt_copy_pos.cuda(),
            tgt_emb=accumulated_tgt_emb.cuda(),
        )
        return generated_tgt_input

    def forward(
        self, batch, config=None, encoder_dict=None, decoder_dict=None, loader=None, verbose=False
    ):
        assert config is not None
        assert encoder_dict is not None
        assert decoder_dict is not None
        src_key_mask, tgt_key_mask = self.make_key_masks(batch, to_cuda=True)

        """pre-encoder block"""
        src = self.get_pre_encoder_block(batch=batch, encoder_dict=encoder_dict)
        src = self.sequence_to_first_dimension(src, batch=batch)
        src = self.encoder_pos_encoder(src)

        """encoder block"""
        # no self-attention mask needed only padding between sequences in batch
        memory = self.transformer_encoder(
            src, mask=None, src_key_padding_mask=src_key_mask
        )

        """generate new input if teacher-forcing"""
        persistent_tgt_input = self.generate_decoder_input(
            batch,
            memory=memory,
            decoder_dict=decoder_dict,
            loader=loader,
            src_key_mask=src_key_mask,
            tgt_key_mask=tgt_key_mask,
        )

        tgt_input = self.sequence_to_first_dimension(
            persistent_tgt_input.cuda(), batch=batch
        )
        tgt_input = self.decoder_pos_encoder(tgt_input)

        """decoder block"""
        output_0 = self.get_decoder_0(
            input=tgt_input,
            memory=memory,
            src_key_mask=src_key_mask,
            tgt_key_mask=tgt_key_mask,
        )

        prob_generate = self.post_decoder(output_0)

        prob_copy = self.get_prob_copy(
            batch=batch,
            input=output_0,
            memory=memory,
            src_key_mask=src_key_mask,
            tgt_key_mask=tgt_key_mask,
        )
        copy_gate = self.copy_gate(output_0)

        """after decoder block"""
        assert prob_generate.shape[0:1] == copy_gate.shape[0:1]
        assert prob_copy.shape[0:1] == copy_gate.shape[0:1]
        prob_generate = prob_generate * (1 - copy_gate)
        prob_copy = prob_copy * copy_gate
        output = torch.cat((prob_generate, prob_copy), -1)

        """calculate loss"""
        logprob = self.eps_log_softmax(output, dim=2)


        target = self.sequence_to_first_dimension(
            batch.tgt_copy_output_idx, batch=batch
        )

        if verbose:
            self.turn_logprob_into_text(logprob, target=target, batch=batch)

        assert logprob.shape[0:1] == target.shape[0:1]
        loss = F.nll_loss(
            input=logprob.view(-1, logprob.shape[-1]),
            target=target.view(-1),
            ignore_index=decoder_dict.pad_token_idx,
        )

        """statistics"""
        loss_stats = self.get_loss_stats(
            batch=batch,
            loss=loss,
            logprob=logprob,
            target=target,
            copy_gate=copy_gate,
            decoder_dict=decoder_dict,
        )

        return loss_stats

    def turn_logprob_into_text(self, logprob, target=None, batch=None):
        _, widx_topk = torch.topk(logprob, 1)
        # pprint.pp("in turn_logprob_into_text")
        # pprint.pp("widx_topk.shape {}".format(widx_topk.shape))
        # pprint.pp("target.shape {}".format(target.shape))
        # pprint.pp("batch.tgt_output_idx.shape {}".format(batch.tgt_copy_output_idx.shape))
        # pprint.pp(batch.len_decoder_dict)
        guess = widx_topk.squeeze(-1).cpu()
        pprint.pp("guess:   {}".format(np.array(guess).squeeze()), width=1000)
        pprint.pp("target:  {}".format(np.array(target.cpu()).squeeze()), width=1000)
        pprint.pp("overlap: {}".format(np.array(torch.eq(guess, target.cpu())).squeeze()), width=1000)

    def validation_loss(
        self, config=None, loader=None, encoder_dict=None, decoder_dict=None
    ):
        assert config is not None
        assert loader is not None
        assert encoder_dict is not None
        assert decoder_dict is not None
        self.eval()
        with torch.no_grad():
            vloss = 0
            vaccu = 0
            vcopy_accu = 0
            vgate_level = 0
            generator = loader.batch_generator(
                bs=config["valid_bs"],
                _batch_list=loader.valid_batchtokens,
                encoder_dict=encoder_dict,
                decoder_dict=decoder_dict,
            )
            for i, batch in enumerate(generator):
                vloss_dict = self.forward(
                    batch=batch,
                    config=config,
                    encoder_dict=encoder_dict,
                    decoder_dict=decoder_dict,
                    loader=loader,
                )
                vloss += vloss_dict["loss"]
                vaccu += vloss_dict["accu"]
                vcopy_accu += vloss_dict["copy_accu"]
                vgate_level += vloss_dict["gate_level"]
            vloss /= i + 1
            vaccu /= i + 1
            vcopy_accu /= i + 1
            vgate_level /= i + 1

        del generator
        self.train()
        return {
            "vloss": vloss,
            "vaccu": vaccu,
            "vcopy_accu": vcopy_accu,
            "vgate_level": vgate_level,
        }

    def get_pre_decoder_block_generation(self, batch, tgt_input_raw):
        raise NotImplementedError
        tgt_input = self.decoder_idx_embedding(tgt_input_raw)
        tgt_input_embedding = torch.zeros(
            tgt_input_raw.size(0), 1, batch.tgt_emb.shape[-1]
        )

        for t, idx in enumerate(tgt_input_raw):
            if idx == self.decoder_dict.pad_token_idx:
                tgt_input[idx:] = torch.zeros(
                    tgt_input_raw.shape[0], 1, tgt_input.shape[-1]
                )
                break
            tgt_input_embedding[t, :] = torch.tensor(
                self.decoder_dict.idx_to_emb(int(idx))
            )

        tgt_input = torch.cat((tgt_input, tgt_input_embedding.cuda()), -1)
        return tgt_input

    def get_gate_level(self, g_t, batch):
        """
           measure the average gate value that is guessed for words that should be copied
        """
        gate = g_t.detach()
        if gate.dim() > 2:
            gate = gate.squeeze(-1)

        gate_level = gate * batch.tgt_copy_gate.transpose(0, 1).contiguous()
        gate_denominator = float(torch.sum(batch.tgt_copy_gate)) ** -1
        gate_level = float(torch.sum(gate_level)) * gate_denominator

        return gate_level

    def get_copy_accu(self, logprob, batch):

        target = batch.tgt_copy_output_idx.transpose(0, 1).contiguous()
        target_gate = batch.tgt_copy_gate.transpose(0, 1)

        target = torch.where(
            target_gate == 1.0, target, torch.zeros(target.shape).long().cuda()
        )

        _, widx_topk = torch.topk(logprob, 1)
        copy_accu = torch.eq(widx_topk.squeeze(-1).cpu(), target.cpu()).float()

        mask_out_generated = (
            torch.eq(target_gate.cpu().float(), torch.ones(target_gate.shape))
        ).float()
        copy_accu = copy_accu * mask_out_generated

        copy_denominator = float(torch.sum(batch.tgt_copy_gate)) ** -1
        return float(torch.sum(copy_accu)) * copy_denominator

    def get_gate_accu(self, g_t, batch, cutoff=0.5):
        gate = g_t.detach()
        if gate.dim() > 2:
            gate = gate.squeeze(-1)

        gate_result = torch.where(
            gate > cutoff,
            torch.ones(gate.shape).float().cuda(),
            torch.zeros(gate.shape).float().cuda(),
        )

        target = batch.tgt_copy_gate.transpose(0, 1).contiguous()

        gate_accu = torch.eq(gate_result, target).float()
        # remove guesses of zeros from gate_accu
        mask_generate_positions = (
            torch.eq(target.cpu().float(), torch.ones(target.shape))
        ).float()

        gate_accu = gate_accu * mask_generate_positions.cuda()

        gate_denominator = float(torch.sum(batch.tgt_copy_gate)) ** -1
        gate_accu = float(torch.sum(gate_accu)) * gate_denominator
        return gate_accu

    def get_logprob_accu(self, logprob=None, target=None, decoder_dict=None, verbose=False):
        assert logprob is not None
        assert target is not None
        assert decoder_dict is not None
        _, widx_topk = torch.topk(logprob, 1)
        guess = widx_topk.squeeze(-1).cpu()
        accu = torch.eq(guess, target.cpu()).float()
        # mask out the padded elements
        pad_mask = (
            ~torch.eq(
                target.cpu().float(),
                torch.ones(target.shape) * decoder_dict.pad_token_idx,
            )
        ).float()
        assert accu.shape == pad_mask.shape
        return accu * pad_mask

    def eps_log_softmax(self, entropy=None, dim=2):
        assert entropy is not None
        assert entropy.dim() == 3
        eps = 10 ** -6
        entropy = F.softmax(entropy, dim=dim)
        logprob = torch.log(entropy + eps)
        return logprob
