#Albert Transform GFN model
from pytorch_pretrained_bert.modeling import (
    BertPreTrainedModel as PreTrainedBertModel, # The name was changed in the new versions of pytorch_pretrained_bert
    BertModel,
    BertLayerNorm,
    gelu,
    BertEncoder,
    BertPooler,
    BertEmbeddings
)
import torch
from torch import nn


from utils import (
    fuzzy_find,
    find_start_end_after_tokenized,
    find_start_end_before_tokenized,
    bundle_part_to_batch,
)
from pytorch_pretrained_bert.tokenization import (
    whitespace_tokenize,
    BasicTokenizer,
    BertTokenizer,
)
import re
import pdb
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer, XLNetTokenizer, XLNetModel
from transformers.models.albert.modeling_albert import AlbertTransformer, AlbertEmbeddings, AlbertModel, AlbertPreTrainedModel
from transformers.models.albert.tokenization_albert import AlbertTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling


class MLP(nn.Module):
    def __init__(self, input_sizes, dropout_prob=0.5, bias=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(input_sizes)):
            self.layers.append(nn.Linear(input_sizes[i - 1], input_sizes[i], bias=bias))
        self.norm_layers = nn.ModuleList()
        if len(input_sizes) > 2:
            for i in range(1, len(input_sizes) - 1):
                self.norm_layers.append(nn.LayerNorm(input_sizes[i]))
        self.drop_out = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.drop_out(x))
            if i < len(self.layers) - 1:
                x = gelu(x)
                if len(self.norm_layers):
                    x = self.norm_layers[i](x)
        return x


class GCN(nn.Module):
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.05)

    def __init__(self, input_size):
        super(GCN, self).__init__()
        self.diffusion = nn.Linear(input_size, input_size, bias=False)
        self.retained = nn.Linear(input_size, input_size, bias=False)
        self.predict = MLP(input_sizes=(input_size, input_size, 1))
        self.apply(self.init_weights)

    def forward(self, A, x):
        layer1_diffusion = A.t().mm(gelu(self.diffusion(x)))
        x = gelu(self.retained(x) + layer1_diffusion)
        layer2_diffusion = A.t().mm(gelu(self.diffusion(x)))
        x = gelu(self.retained(x) + layer2_diffusion)
        return self.predict(x).squeeze(-1)


class BertEmbeddingsPlus(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, max_sentence_type=30):
        super(BertEmbeddingsPlus, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.sentence_type_embeddings = nn.Embedding(
            max_sentence_type, config.hidden_size
        )
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings((token_type_ids > 0).long())
        sentence_type_embeddings = self.sentence_type_embeddings(token_type_ids)

        embeddings = (
            words_embeddings
            + position_embeddings
            + token_type_embeddings
            + sentence_type_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModelPlus(BertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddingsPlus(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(
        self, input_ids, token_type_ids=None, attention_mask=None, output_hidden=-4
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask, output_all_encoded_layers=True
        )
        sequence_output = encoded_layers[-1]
        # pooled_output = self.pooler(sequence_output)


        #print(encoded_layers[-1])
        #print(encoded_layers[output_hidden])

        encoded_layers, hidden_layers = (
            encoded_layers[-1],
            encoded_layers[output_hidden],
        )
        return encoded_layers, hidden_layers


class AlbertModelPlus(AlbertModel):
    def __init__(self, config):
        super(AlbertModel, self).__init__(config)
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        #self.apply(self.init_bert_weights)
        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        output_hidden=-4
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # return BaseModelOutputWithPooling(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        # )
        encoded_layers = encoder_outputs[0]
        hidden_layers = encoder_outputs[0]
        return encoded_layers, hidden_layers

# class BertForMultiHopQuestionAnswering(PreTrainedBertModel):
#     def __init__(self, config):
#         super(BertForMultiHopQuestionAnswering, self).__init__(config)
#         self.bert = BertModelPlus(config)
#         self.qa_outputs = nn.Linear(config.hidden_size, 4)
#         self.apply(self.init_bert_weights)
#
#     def forward(
#             self,
#             input_ids,
#             token_type_ids=None,
#             attention_mask=None,
#             sep_positions=None,
#             hop_start_weights=None,
#             hop_end_weights=None,
#             ans_start_weights=None,
#             ans_end_weights=None,
#             B_starts=None,
#             allow_limit=(0, 0),
#     ):
#         """ Extract spans by System 1.
#
#         Args:
#             input_ids (LongTensor): Token ids of word-pieces. (batch_size * max_length)
#             token_type_ids (LongTensor): The A/B Segmentation in BERTs. (batch_size * max_length)
#             attention_mask (LongTensor): Indicating whether the position is a token or padding. (batch_size * max_length)
#             sep_positions (LongTensor): Positions of [SEP] tokens, mainly used in finding the num_sen of supporing facts. (batch_size * max_seps)
#             hop_start_weights (Tensor): The ground truth of the probability of hop start positions. The weight of sample has been added on the ground truth.
#                 (You can verify it by examining the gradient of binary cross entropy.)
#             hop_end_weights ([Tensor]): The ground truth of the probability of hop end positions.
#             ans_start_weights ([Tensor]): The ground truth of the probability of ans start positions.
#             ans_end_weights ([Tensor]): The ground truth of the probability of ans end positions.
#             B_starts (LongTensor): Start positions of sentence B.
#             allow_limit (tuple, optional): An Offset for negative threshold. Defaults to (0, 0).
#
#         Returns:
#             [type]: [description]
#         """
#         batch_size = input_ids.size()[0]
#         device = input_ids.get_device() if input_ids.is_cuda else torch.device("cpu")
#         sequence_output, hidden_output = self.bert(
#             input_ids, token_type_ids, attention_mask
#         )
#         semantics = hidden_output[:, 0]
#         # Some shapes: sequence_output [batch_size, max_length, hidden_size], pooled_output [batch_size, hidden_size]
#         if sep_positions is None:
#             return semantics  # Only semantics, used in bundle forward
#         else:
#             max_sep = sep_positions.size()[-1]
#         if max_sep == 0:
#             empty = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
#             return (
#                 empty,
#                 empty,
#                 semantics,
#                 empty,
#             )  # Only semantics, used in eval, the same ``empty'' variable is a mistake in general cases but simple
#
#         # Predict spans
#         logits = self.qa_outputs(sequence_output)
#         hop_start_logits, hop_end_logits, ans_start_logits, ans_end_logits = logits.split(
#             1, dim=-1
#         )
#         hop_start_logits = hop_start_logits.squeeze(-1)
#         hop_end_logits = hop_end_logits.squeeze(-1)
#         ans_start_logits = ans_start_logits.squeeze(-1)
#         ans_end_logits = ans_end_logits.squeeze(-1)  # Shape: [batch_size, max_length]
#
#         if hop_start_weights is not None:  # Train mode
#             lgsf = torch.nn.LogSoftmax(
#                 dim=1
#             )  # If there is no targeted span in the sentence, start_weights = end_weights = 0(vec)
#             hop_start_loss = -torch.sum(
#                 hop_start_weights * lgsf(hop_start_logits), dim=-1
#             )
#             hop_end_loss = -torch.sum(hop_end_weights * lgsf(hop_end_logits), dim=-1)
#             ans_start_loss = -torch.sum(
#                 ans_start_weights * lgsf(ans_start_logits), dim=-1
#             )
#             ans_end_loss = -torch.sum(ans_end_weights * lgsf(ans_end_logits), dim=-1)
#             hop_loss = torch.mean((hop_start_loss + hop_end_loss)) / 2
#             ans_loss = torch.mean((ans_start_loss + ans_end_loss)) / 2
#         else:
#             # In eval mode, find the exact top K spans.
#             K_hop, K_ans = 3, 1
#             hop_preds = torch.zeros(
#                 batch_size, K_hop, 3, dtype=torch.long, device=device
#             )  # (start, end, sen_num)
#             ans_preds = torch.zeros(
#                 batch_size, K_ans, 3, dtype=torch.long, device=device
#             )
#             ans_start_gap = torch.zeros(batch_size, device=device)
#             for u, (start_logits, end_logits, preds, K, allow) in enumerate(
#                     (
#                             (
#                                     hop_start_logits,
#                                     hop_end_logits,
#                                     hop_preds,
#                                     K_hop,
#                                     allow_limit[0],
#                             ),
#                             (
#                                     ans_start_logits,
#                                     ans_end_logits,
#                                     ans_preds,
#                                     K_ans,
#                                     allow_limit[1],
#                             ),
#                     )
#             ):
#                 for i in range(batch_size):
#                     if sep_positions[i, 0] > 0:
#                         values, indices = start_logits[i, B_starts[i]:].topk(K)
#                         for k, index in enumerate(indices):
#                             if values[k] <= start_logits[i, 0] - allow:  # not golden
#                                 if u == 1:  # For ans spans
#                                     ans_start_gap[i] = start_logits[i, 0] - values[k]
#                                 break
#                             start = index + B_starts[i]
#                             # find ending
#                             for j, ending in enumerate(sep_positions[i]):
#                                 if ending > start or ending <= 0:
#                                     break
#                             if ending <= start:
#                                 break
#                             ending = min(ending, start + 10)
#                             end = torch.argmax(end_logits[i, start:ending]) + start
#                             preds[i, k, 0] = start
#                             preds[i, k, 1] = end
#                             preds[i, k, 2] = j
#         return (
#             (hop_loss, ans_loss, semantics)
#             if hop_start_weights is not None
#             else (hop_preds, ans_preds, semantics, ans_start_gap)
#         )


class BertForMultiHopQuestionAnswering(AlbertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultiHopQuestionAnswering, self).__init__(config)
        self.albert = AlbertModelPlus(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 4)
        self._init_weights = self._init_weights

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        sep_positions=None,
        hop_start_weights=None,
        hop_end_weights=None,
        ans_start_weights=None,
        ans_end_weights=None,
        B_starts=None,
        allow_limit=(0, 0),
    ):
        """ Extract spans by System 1.
        
        Args:
            input_ids (LongTensor): Token ids of word-pieces. (batch_size * max_length)
            token_type_ids (LongTensor): The A/B Segmentation in BERTs. (batch_size * max_length)
            attention_mask (LongTensor): Indicating whether the position is a token or padding. (batch_size * max_length)
            sep_positions (LongTensor): Positions of [SEP] tokens, mainly used in finding the num_sen of supporing facts. (batch_size * max_seps)
            hop_start_weights (Tensor): The ground truth of the probability of hop start positions. The weight of sample has been added on the ground truth. 
                (You can verify it by examining the gradient of binary cross entropy.)
            hop_end_weights ([Tensor]): The ground truth of the probability of hop end positions.
            ans_start_weights ([Tensor]): The ground truth of the probability of ans start positions.
            ans_end_weights ([Tensor]): The ground truth of the probability of ans end positions.
            B_starts (LongTensor): Start positions of sentence B.
            allow_limit (tuple, optional): An Offset for negative threshold. Defaults to (0, 0).
        
        Returns:
            [type]: [description]
        """
        # model = BertModel.from_pretrained('/home/shaoai/CogQA/uncased_L-2_H-128_A-2')
        # outputs = model(input_ids)
        # sequence_output = outputs[0]
        # pooled_output = outputs[1]
        # print(sequence_output.shape)
        # print(pooled_output.shape)
        # print(self.bert(
        #     input_ids, token_type_ids, attention_mask
        # ))
        batch_size = input_ids.size()[0]
        device = input_ids.get_device() if input_ids.is_cuda else torch.device("cpu")
        sequence_output, hidden_output = self.albert(
            input_ids, token_type_ids, attention_mask
        )
        semantics = hidden_output[:, 0]
        # Some shapes: sequence_output [batch_size, max_length, hidden_size], pooled_output [batch_size, hidden_size]
        # print("1" , batch_size , '\n')
        # print("2" , device , '\n')
        # print("3" , sequence_output , '\n')
        # print("3" , len(sequence_output) , '\n')
        # print("3", len(sequence_output[0]), '\n')
        # print("4" , hidden_output , '\n')
        # print("4", len(hidden_output), '\n')
        # print("4", len(hidden_output[0]), '\n')
        # print("5" , semantics , '\n')
        # print("5", len(semantics), '\n')
        # print("5", len(semantics[0]), '\n')
        # print("6", semantics.shape, '\n')
        # print("6", sequence_output.shape, '\n')
        # print("6", hidden_output.shape, '\n')

        if sep_positions is None:
            return semantics  # Only semantics, used in bundle forward
        else:
            max_sep = sep_positions.size()[-1]
        if max_sep == 0:
            empty = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            return (
                empty,
                empty,
                semantics,
                empty,
            )  # Only semantics, used in eval, the same ``empty'' variable is a mistake in general cases but simple

        # Predict spans
        logits = self.qa_outputs(sequence_output)
        hop_start_logits, hop_end_logits, ans_start_logits, ans_end_logits = logits.split(
            1, dim=-1
        )
        hop_start_logits = hop_start_logits.squeeze(-1)
        hop_end_logits = hop_end_logits.squeeze(-1)
        ans_start_logits = ans_start_logits.squeeze(-1)
        ans_end_logits = ans_end_logits.squeeze(-1)  # Shape: [batch_size, max_length]

        if hop_start_weights is not None:  # Train mode
            lgsf = torch.nn.LogSoftmax(
                dim=1
            )  # If there is no targeted span in the sentence, start_weights = end_weights = 0(vec)
            hop_start_loss = -torch.sum(
                hop_start_weights * lgsf(hop_start_logits), dim=-1
            )
            hop_end_loss = -torch.sum(hop_end_weights * lgsf(hop_end_logits), dim=-1)
            ans_start_loss = -torch.sum(
                ans_start_weights * lgsf(ans_start_logits), dim=-1
            )
            ans_end_loss = -torch.sum(ans_end_weights * lgsf(ans_end_logits), dim=-1)
            hop_loss = torch.mean((hop_start_loss + hop_end_loss)) / 2
            ans_loss = torch.mean((ans_start_loss + ans_end_loss)) / 2
        else:
            # In eval mode, find the exact top K spans.
            K_hop, K_ans = 3, 1
            hop_preds = torch.zeros(
                batch_size, K_hop, 3, dtype=torch.long, device=device
            )  # (start, end, sen_num)
            ans_preds = torch.zeros(
                batch_size, K_ans, 3, dtype=torch.long, device=device
            )
            ans_start_gap = torch.zeros(batch_size, device=device)
            for u, (start_logits, end_logits, preds, K, allow) in enumerate(
                (
                    (
                        hop_start_logits,
                        hop_end_logits,
                        hop_preds,
                        K_hop,
                        allow_limit[0],
                    ),
                    (
                        ans_start_logits,
                        ans_end_logits,
                        ans_preds,
                        K_ans,
                        allow_limit[1],
                    ),
                )
            ):
                for i in range(batch_size):
                    if sep_positions[i, 0] > 0:
                        values, indices = start_logits[i, B_starts[i] :].topk(K)
                        for k, index in enumerate(indices):
                            if values[k] <= start_logits[i, 0] - allow:  # not golden
                                if u == 1: # For ans spans
                                    ans_start_gap[i] = start_logits[i, 0] - values[k]
                                break
                            start = index + B_starts[i]
                            # find ending
                            for j, ending in enumerate(sep_positions[i]):
                                if ending > start or ending <= 0:
                                    break
                            if ending <= start:
                                break
                            ending = min(ending, start + 10)
                            end = torch.argmax(end_logits[i, start:ending]) + start
                            preds[i, k, 0] = start
                            preds[i, k, 1] = end
                            preds[i, k, 2] = j
        return (
            (hop_loss, ans_loss, semantics)
            if hop_start_weights is not None
            else (hop_preds, ans_preds, semantics, ans_start_gap)
        )


class CognitiveGNN(nn.Module):
    def __init__(self, hidden_size):
        super(CognitiveGNN, self).__init__()
        self.gcn = GCN(hidden_size)
        self.both_net = MLP((hidden_size, hidden_size, 1))
        self.select_net = MLP((hidden_size, hidden_size, 1))

    def forward(self, bundle, model, device):
        batch = bundle_part_to_batch(bundle)
        batch = tuple(t.to(device) for t in batch)
        hop_loss, ans_loss, semantics = model(
            *batch
        )  # Shape of semantics: [num_para, hidden_size]
        num_additional_nodes = len(bundle.additional_nodes)

        if num_additional_nodes > 0:
            max_length_additional = max([len(x) for x in bundle.additional_nodes])
            ids = torch.zeros(
                (num_additional_nodes, max_length_additional),
                dtype=torch.long,
                device=device,
            )
            segment_ids = torch.zeros(
                (num_additional_nodes, max_length_additional),
                dtype=torch.long,
                device=device,
            )
            input_mask = torch.zeros(
                (num_additional_nodes, max_length_additional),
                dtype=torch.long,
                device=device,
            )
            for i in range(num_additional_nodes):
                length = len(bundle.additional_nodes[i])
                ids[i, :length] = torch.tensor(
                    bundle.additional_nodes[i], dtype=torch.long
                )
                input_mask[i, :length] = 1
            additional_semantics = model(ids, segment_ids, input_mask)

            semantics = torch.cat((semantics, additional_semantics), dim=0)

        assert semantics.size()[0] == bundle.adj.size()[0]

        if bundle.question_type == 0:  # Wh-
            pred = self.gcn(bundle.adj.to(device), semantics)
            ce = torch.nn.CrossEntropyLoss()
            final_loss = ce(
                pred.unsqueeze(0),
                torch.tensor([bundle.answer_id], dtype=torch.long, device=device),
            )
        else:
            x, y, ans = bundle.answer_id
            ans = torch.tensor(ans, dtype=torch.float, device=device)
            diff_sem = semantics[x] - semantics[y]
            classifier = self.both_net if bundle.question_type == 1 else self.select_net
            final_loss = 0.2 * torch.nn.functional.binary_cross_entropy_with_logits(
                classifier(diff_sem).squeeze(-1), ans.to(device)
            )
        return hop_loss, ans_loss, final_loss

import torch
import torch.nn as nn
import numpy as np
from transpytorch.transformer.Layers import EncoderLayer, DecoderLayer


__author__ = "Yu-Hsiang Huang"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))


from DFGN.model.layers import *
class GraphFusionNet(nn.Module):
    """
    Packing Query Version
    """
    def __init__(self, config):
        super(GraphFusionNet, self).__init__()
        self.config = config
        self.n_layers = config.n_layers
        self.max_query_length = 50

        self.bi_attention = BiAttention(input_dim=config.input_dim,
                                        memory_dim=config.input_dim,
                                        hid_dim=config.hidden_dim,
                                        dropout=config.bi_attn_drop)
        self.bi_attn_linear = nn.Linear(config.hidden_dim * 4, config.hidden_dim)

        h_dim = config.hidden_dim
        q_dim = config.hidden_dim if config.q_update else config.input_dim

        self.basicblocks = nn.ModuleList()
        self.query_update_layers = nn.ModuleList()
        self.query_update_linears = nn.ModuleList()

        for layer in range(self.n_layers):
            self.basicblocks.append(BasicBlock(h_dim, q_dim, layer, config))
            if config.q_update:
                self.query_update_layers.append(BiAttention(h_dim, h_dim, h_dim, config.bi_attn_drop))
                self.query_update_linears.append(nn.Linear(h_dim * 4, h_dim))

        q_dim = h_dim if config.q_update else config.input_dim
        if config.prediction_trans:
            self.predict_layer = TransformerPredictionLayer(self.config, q_dim)
        else:
            self.predict_layer = PredictionLayer(self.config, q_dim)

    def forward(self, batch, return_yp, debug=False):
        query_mapping = batch['query_mapping']
        entity_mask = batch['entity_mask']
        context_encoding = batch['context_encoding']

        # extract query encoding
        trunc_query_mapping = query_mapping[:, :self.max_query_length].contiguous()
        trunc_query_state = (context_encoding * query_mapping.unsqueeze(2))[:, :self.max_query_length, :].contiguous()
        # bert encoding query vec
        query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        attn_output, trunc_query_state = self.bi_attention(context_encoding, trunc_query_state, trunc_query_mapping)
        input_state = self.bi_attn_linear(attn_output)

        if self.config.q_update:
            query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        softmasks = []
        entity_state = None
        for l in range(self.n_layers):
            input_state, entity_state, softmask = self.basicblocks[l](input_state, query_vec, batch)
            softmasks.append(softmask)
            if self.config.q_update:
                query_attn_output, _ = self.query_update_layers[l](trunc_query_state, entity_state, entity_mask)
                trunc_query_state = self.query_update_linears[l](query_attn_output)
                query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        predictions = self.predict_layer(batch, input_state, query_vec, entity_state, query_mapping, return_yp)
        start, end, sp, Type, ent, yp1, yp2 = predictions

        if return_yp:
            return start, end, sp, Type, softmasks, ent, yp1, yp2
        else:
            return start, end, sp, Type, softmasks, ent


if __name__ == "__main__":

    # BERT_MODEL = 'bert-base-uncased'
    # tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)

    # tokenizer = BertTokenizer.from_pretrained("./albert_base")
    # BERT_MODEL = BertModel.from_pretrained("./albert_base")

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)
    BERT_MODEL = 'albert-base-v2'

    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
    # BERT_MODEL = RobertaModel.from_pretrained('roberta-base')

    # tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
    # BERT_MODEL = 'xlnet-base-cased'

    # BERT_MODEL = '/home/shaoai/CogQA/uncased_L-2_H-128_A-2'
    # tokenizer = BertTokenizer.from_pretrained('/home/shaoai/CogQA/uncased_L-2_H-128_A-2')

    orig_text = "".join(
        [
            "Theatre Centre is a UK-based theatre company touring new plays for young audiences aged 4 to 18, founded in 1953 by Brian Way, the company has developed plays by writers including which British writer, dub poet and Rastafarian?",
            " It is the largest urban not-for-profit theatre company in the country and the largest in Western Canada, with productions taking place at the 650-seat Stanley Industrial Alliance Stage, the 440-seat Granville Island Stage, the 250-seat Goldcorp Stage at the BMO Theatre Centre, and on tour around the province.",
        ]
    )
    tokenized_text = tokenizer.tokenize(orig_text)
    print(len(tokenized_text))

