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


from lstm.utils import *
from lstm.embedding import embed
from lstm.parameters import *

class rnn_crf(nn.Module):
    def __init__(self, cti_size, wti_size, num_tags):
        super().__init__()
        self.rnn = rnn(cti_size, wti_size, num_tags)
        self.crf = crf(num_tags)
        if CUDA: self = self.cuda()

    def forward(self, xc, xw, y0): # for training
        self.zero_grad()
        self.rnn.batch_size = y0.size(1)
        self.crf.batch_size = y0.size(1)
        mask = y0[1:].gt(PAD_IDX).float()
        h = self.rnn(xc, xw, mask)
        Z = self.crf.forward(h, mask)
        score = self.crf.score(h, y0, mask)
        return torch.mean(Z - score) # NLL loss

    def decode(self, xc, xw, lens): # for inference
        self.rnn.batch_size = len(lens)
        self.crf.batch_size = len(lens)
        if HRE:
            mask = [[1] * x + [PAD_IDX] * (lens[0] - x) for x in lens]
            mask = Tensor(mask).transpose(0, 1)
        else:
            mask = xw.gt(PAD_IDX).float()
        h = self.rnn(xc, xw, mask)
        return self.crf.decode(h, mask)

class rnn(nn.Module):
    def __init__(self, cti_size, wti_size, num_tags):
        super().__init__()
        self.batch_size = 0

        # architecture
        self.embed = embed(EMBED, cti_size, wti_size, HRE)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = EMBED_SIZE,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )
        self.out = nn.Linear(HIDDEN_SIZE, num_tags) # RNN output to tag

    def init_state(self): # initialize RNN states
        n = NUM_LAYERS * NUM_DIRS
        b = self.batch_size
        h = HIDDEN_SIZE // NUM_DIRS
        hs = zeros(n, b, h) # hidden state
        if RNN_TYPE == "LSTM":
            cs = zeros(n, b, h) # LSTM cell state
            return (hs, cs)
        return hs

    def forward(self, xc, xw, mask):
        hs = self.init_state()
        x = self.embed(xc, xw)
        if HRE: # [B * Ld, 1, H] -> [Ld, B, H]
            x = x.view(-1, self.batch_size, EMBED_SIZE)
        lens = mask.sum(0).int().cpu()
        x = nn.utils.rnn.pack_padded_sequence(x, lens, enforce_sorted = False)
        h, _ = self.rnn(x, hs)
        h, _ = nn.utils.rnn.pad_packed_sequence(h)
        h = self.out(h)
        h *= mask.unsqueeze(2)
        return h

class crf(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.batch_size = 0
        self.num_tags = num_tags

        # transition scores from j to i
        self.trans = nn.Parameter(randn(num_tags, num_tags))
        self.trans.data[SOS_IDX, :] = -10000 # no transition to SOS
        self.trans.data[:, EOS_IDX] = -10000 # no transition from EOS except to PAD
        self.trans.data[:, PAD_IDX] = -10000 # no transition from PAD except to PAD
        self.trans.data[PAD_IDX, :] = -10000 # no transition to PAD except from EOS
        self.trans.data[PAD_IDX, EOS_IDX] = 0
        self.trans.data[PAD_IDX, PAD_IDX] = 0

    def forward(self, h, mask): # forward algorithm
        score = Tensor(self.batch_size, self.num_tags).fill_(-10000)
        score[:, SOS_IDX] = 0.
        trans = self.trans.unsqueeze(0) # [1, C, C]
        for _h, _mask in zip(h, mask):
            _mask = _mask.unsqueeze(1)
            _emit = _h.unsqueeze(2) # [B, C, 1]
            _score = score.unsqueeze(1) + _emit + trans # [B, 1, C] -> [B, C, C]
            _score = log_sum_exp(_score) # [B, C, C] -> [B, C]
            score = _score * _mask + score * (1 - _mask)
        score = log_sum_exp(score + self.trans[EOS_IDX])
        return score # partition function

    def score(self, h, y0, mask):
        score = Tensor(self.batch_size).fill_(0.)
        h = h.unsqueeze(3) # [L, B, C, 1]
        trans = self.trans.unsqueeze(2) # [C, C, 1]
        for t, (_h, _mask) in enumerate(zip(h, mask)):
            _emit = torch.cat([_h[y0] for _h, y0 in zip(_h, y0[t + 1])])
            _trans = torch.cat([trans[x] for x in zip(y0[t + 1], y0[t])])
            score += (_emit + _trans) * _mask
        last_tag = y0.gather(0, mask.sum(0).long().unsqueeze(0)).squeeze(0)
        score += self.trans[EOS_IDX, last_tag]
        return score

    def decode(self, h, mask): # Viterbi decoding
        bptr = LongTensor()
        score = Tensor(self.batch_size, self.num_tags).fill_(-10000)
        score[:, SOS_IDX] = 0.
        for _h, _mask in zip(h, mask):
            _mask = _mask.unsqueeze(1)
            _score = score.unsqueeze(1) + self.trans # [B, 1, C] -> [B, C, C]
            _score, _bptr = _score.max(2) # best previous scores and tags
            _score += _h # add emission scores
            bptr = torch.cat((bptr, _bptr.unsqueeze(1)), 1)
            score = _score * _mask + score * (1 - _mask)
        score += self.trans[EOS_IDX]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(self.batch_size):
            i = best_tag[b]
            j = mask[:, b].sum().int()
            for _bptr in reversed(bptr[b][:j]):
                i = _bptr[i]
                best_path[b].append(i)
            best_path[b].pop()
            best_path[b].reverse()

        return best_path


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

