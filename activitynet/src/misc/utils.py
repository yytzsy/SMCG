import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    seq = seq.cpu()
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j].item()
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out


class RewardCriterion(nn.Module):

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.contiguous().view(-1)
        reward = reward.contiguous().view(-1)
        mask = (seq > 0).float()
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1).cuda(),
                         mask[:, :-1]], 1).contiguous().view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        # truncate to the same size
        batch_size = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output



class LanguageModelCriterion_withSyntax(nn.Module):

    def __init__(self, caption_alpha, syntax_alpha, content_alpha):
        super(LanguageModelCriterion_withSyntax, self).__init__()

        self.loss_fn_caption = nn.NLLLoss(reduce=False)
        self.loss_fn_pos = nn.NLLLoss(reduce=False)
        self.loss_fn_content = nn.MSELoss(reduce=False)

        self.syntax_alpha = syntax_alpha
        self.content_alpha = content_alpha
        self.caption_alpha = caption_alpha

    def forward(self, logits_caption, target_caption, mask_caption, logits, target, mask, rec_fts, video_fts, feats_mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """

        batch_size = logits_caption.shape[0]
        target_caption = target_caption[:, :logits_caption.shape[1]]
        mask_caption = mask_caption[:, :logits_caption.shape[1]]
        logits_caption = logits_caption.contiguous().view(-1, logits_caption.shape[2])
        target_caption = target_caption.contiguous().view(-1)
        mask_caption = mask_caption.contiguous().view(-1)
        loss = self.loss_fn_caption(logits_caption, target_caption)
        output_caption = torch.sum(loss * mask_caption) / batch_size
        output_caption = output_caption * self.caption_alpha


        # truncate to the same size
        batch_size = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        mask = mask.contiguous().view(-1)
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        loss = self.loss_fn_pos(logits, target)
        output_syntax = torch.sum(loss * mask) / batch_size
        output_syntax = output_syntax * self.syntax_alpha

        batch_size, video_seq_num, video_fts_dim = np.shape(rec_fts)
        rec_fts = (feats_mask*rec_fts).contiguous().view(-1, video_fts_dim)
        video_fts = (feats_mask*video_fts).contiguous().view(-1, video_fts_dim)
        loss = self.loss_fn_content(rec_fts,video_fts)
        loss = torch.sum(loss)
        output_content = torch.sum(loss) / batch_size / video_seq_num
        output_content = output_content * self.content_alpha


        output = output_syntax + output_content + output_caption
        return output, output_syntax, output_content, output_caption