from util import *

class BertEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, ids, seg, mask, out_seq=False):
        seq_out = self.bert(input_ids=ids, token_type_ids=seg, attention_mask=mask)[0]
        seq_emb = self._average_seq_emb(seq_out, mask.float())
        if out_seq:
            return seq_emb, seq_out
        return seq_emb

    def _average_seq_emb(self, seq_out, mask):
        length = torch.sum(mask, dim=1)
        seq_emb = torch.sum(seq_out * mask[:, :, None], dim=1)
        seq_emb = seq_emb / length[:, None]
        return seq_emb

