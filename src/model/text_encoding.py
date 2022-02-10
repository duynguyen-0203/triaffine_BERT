from typing import List
from transformers import RobertaPreTrainedModel, RobertaConfig, RobertaModel
import torch
import torch.nn as nn


class TextEncoding(RobertaPreTrainedModel):
    """Text encoding for Triaffine-BERT model for Named Entity Recognition"""

    def __init__(self, config: RobertaConfig, embed: torch.tensor, dropout: float, freeze_transformer: bool,
                 lstm_layers: int, lstm_dropout: float, pos_size: int, char_lstm_layers: int, char_lstm_dropout: float,
                 char_size: int, use_fasttext: bool, use_pos: bool, use_char_lstm: bool, n_poses: int,
                 list_char: List[str]):
        super(TextEncoding, self).__init__(config)

        self.bert = RobertaModel(config)
        self.wordvec_size = embed.size(-1)
        self.pos_size = pos_size
        self.use_pos = use_pos
        self.use_fasttext = use_fasttext
        self.use_char_lstm = use_char_lstm
        self.char_lstm_layers = char_lstm_layers
        self.char_lstm_dropout = char_lstm_dropout
        self.char_size = char_size
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.freeze_transformer = freeze_transformer
        self.dropout = nn.Dropout(dropout)

        lstm_input_size = config.hidden_size
        if self.use_fasttext:
            lstm_input_size += self.wordvec_size
            self.wordvec_embedding = nn.Embedding.from_pretrained(embed)
        if self.use_pos:
            lstm_input_size += self.pos_size
            self.pos_embedding = nn.Embedding(num_embeddings=n_poses, embedding_dim=self.pos_size)
        if self.use_char_lstm:
            self.list_char = list_char
            n_chars = len(self.list_char)
            lstm_input_size += self.char_size * 2
            self.char_embedding = nn.Embedding(num_embeddings=n_chars, embedding_dim=self.char_size)
            self.char_lstm = nn.LSTM(input_size=self.char_size, hidden_size=self.char_size,
                                     num_layers=self.char_lstm_layers, bidirectional=True,
                                     dropout=self.char_lstm_dropout, batch_first=True)

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=config.hidden_size // 2,
                            num_layers=self.lstm_layers, bidirectional=True,
                            dropout=self.lstm_dropout, batch_first=True)

        if self.freeze_transformer:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False

        self.init_weights()

    def forward(self, encoding: torch.tensor, context_masks: torch.tensor, token_masks: torch.tensor,
                token_masks_bool: torch.tensor, pos_encoding: torch.tensor = None,
                wordvec_encoding: torch.tensor = None, char_encoding: torch.tensor = None,
                token_masks_char: torch.tensor = None, char_count: torch.tensor = None):
        context_masks = context_masks.float()
        batch_size = encoding.shape[0]
        token_count = token_masks_bool.long().sum(-1, keepdim=True)

        bert_outputs = self.bert(input_ids=encoding, attention_mask=context_masks)
        h_token = combine(bert_outputs, token_masks, 'max')

        embeds = [h_token]
        if self.use_pos:
            pos_embed = self.pos_embedding(pos_encoding)
            pos_embed = self.dropout(pos_embed)
            embeds.append(pos_embed)
        if self.use_fasttext:
            word_embed = self.wordvec_embedding(wordvec_encoding)
            word_embed = self.dropout(word_embed)
            embeds.append(word_embed)
        if self.use_char_lstm:
            char_count = char_count.view(-1)
            max_token_count = char_encoding.size(1)
            max_char_count = char_encoding.size(2)

            char_encoding = char_encoding.view(max_token_count * max_char_count, max_char_count)
            char_encoding[char_count == 0][:, 0] = self.list_char.index('<EOT>')
            char_count[char_count == 0] = 1
            char_embed = self.char_embedding(char_encoding)
            char_embed = self.dropout(char_embed)
            char_embed_packed = nn.utils.rnn.pack_padded_sequence(input=char_embed, lengths=char_count.tolist(),
                                                                  enforce_sorted=False, batch_first=True)
            char_embed_packed_o, _ = self.char_lstm(char_embed_packed)
            char_embed, _ = nn.utils.rnn.pad_packed_sequence(char_embed_packed_o, batch_first=True)
            char_embed = char_embed.view(batch_size, max_token_count, max_char_count, self.char_size * 2)
            h_token_char = combine(char_embed, token_masks_char, 'max')
            embeds.append(h_token_char)

        if len(embeds) > 1:
            h_token_pos_wordvec_char = torch.cat(embeds, dim=-1)
            h_token_pos_wordvec_char_packed = nn.utils.rnn.pack_padded_sequence(
                input=h_token_pos_wordvec_char,
                lengths=token_count.squeeze(-1).cpu().tolist(),
                enforce_sorted=False, batch_first=True
            )
            h_token_pos_wordvec_char_packed_o, _ = self.lstm(h_token_pos_wordvec_char_packed)
            h_token_pos_wordvec_char, _ = nn.utils.rnn.pad_packed_sequence(h_token_pos_wordvec_char_packed_o,
                                                                           batch_first=True)

        else:
            return h_token

        return h_token_pos_wordvec_char


def combine(sub, sup_mask, pool_type='max'):
    """Combine different level representations"""

    sup = None
    if len(sub.shape) == len(sup_mask.shape):
        if pool_type == 'mean':
            size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
            sup = sup.sum(dim=2) / size
        elif pool_type == 'sum':
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
            sup = sup.sum(dim=2)
        elif pool_type == 'max':
            m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
            sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
            sup = sup.max(dim=2)[0]
            sup[sup == -1e30] = 0
    else:
        if pool_type == 'mean':
            size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub
            sup = sup.sum(dim=2) / size
        elif pool_type == 'sum':
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub
            sup = sup.sum(dim=2)
        elif pool_type == 'max':
            m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
            sup = m + sub
            sup = sup.max(dim=2)[0]
            sup[sup == -1e30] = 0

    return sup
