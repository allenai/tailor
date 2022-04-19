from transformers import Seq2SeqTrainer
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
"""
Some sample code
https://github.com/facebookresearch/unlikelihood_training/blob/master/custom/gpt2/run_gpt2.py
https://github.com/anthailan1702/unlikelihood-training/blob/master/unlikelihood_util.py
"""


def token_unlikelihood_loss(logits, targets, padding_idx, alpha=1.0):
    logprobs = F.log_softmax(logits, dim=-1)
    mle_loss = F.nll_loss(logprobs.view(-1, logprobs.size(-1)), targets.reshape(-1), reduction='mean')

    with torch.no_grad():
        ctx_cands = targets.unsqueeze(1).expand(targets.size(0), targets.size(-1), targets.size(-1))
        ctx_cands_ = (ctx_cands.tril(-1) + padding_idx)
        ctx_cands_ = ctx_cands_.triu()
        ctx_cands = ctx_cands.tril(-1) + ctx_cands_

        negative_targets = torch.zeros_like(logprobs).scatter_(-1, ctx_cands, 1)

    one_minus_probs = torch.clamp((1.0 - logprobs.exp()), min=1e-5)
    custom_loss = -torch.log(one_minus_probs) * negative_targets
    custom_loss = custom_loss.mean()

    loss = mle_loss + alpha * custom_loss
    return loss
    
class Seq2SeqUnlikelihoodTrainer(Seq2SeqTrainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        NULL_IDX = -100 # padding index
        if "weight" in inputs: inputs.pop("weight")
        rewards = inputs.pop("rewards") if "rewards" in inputs else None
        outputs = model(**inputs)
        #print(list(outputs))
        #print(self.args.use_unlikelihood)
        #print(list(inputs), type(model))
        
        #print("labels", labels)
        #print("rewards", rewards)
        #print(list(inputs), labels)
        
        logits = outputs.logits
        #loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        #total_loss, total_logits = model(**inputs, labels=labels, use_cache=False)[:2]
        #print(loss)
        
        scores = F.log_softmax(logits, dim=-1)
        #print(logits.shape)
        scores_view = scores.view(-1, scores.size(-1))
        labels = inputs.pop("labels") if "labels" in inputs else None
        labels_view = labels.view(-1)
        #print("scores", scores.shape)
        #print("scores_view", scores_view.shape)
        #print("targets", targets.shape)
        #print("targets_view", targets_view.shape)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        notnull = labels.ne(NULL_IDX)
        # separat cases that use mle or ul
        mle_notnull = notnull & (rewards >= 0).unsqueeze(1).expand_as(notnull)
        #print("mle_notnull", mle_notnull)
        #print(mle_notnull.float())
        mle_loss = (
            F.nll_loss(scores_view, labels_view, ignore_index=NULL_IDX, reduction='none').view_as(mle_notnull)
            * mle_notnull.float()
        ).sum()
        mle_target_tokens = mle_notnull.long().sum().item()
        
        if mle_target_tokens > 0:
            mle_loss = mle_loss / mle_target_tokens  # average loss per token
        # if not using unlikelihood, should just return the mle loss
        #print("mle_loss", mle_loss)
        if not self.args.use_unlikelihood:
            return (mle_loss, outputs) if return_outputs else mle_loss
        
        # get unlikelihood
        ul_notnull = notnull & (rewards < 0).unsqueeze(1).expand_as(notnull)
        ul_target_tokens = ul_notnull.long().sum().item()

        #print("mle_target_tokens", mle_target_tokens)
        #print("ul_target_tokens", ul_target_tokens)

        range_ = torch.arange(labels_view.size(0)).to(labels.device)
        ul_scores = scores_view[range_, labels_view]

        clamp_min = 1e-6 if self.args.fp16 else 1e-20
        ul_loss = (
            -torch.log(torch.clamp(1.0 - ul_scores.exp(), min=clamp_min))
            * ul_notnull.reshape(-1).float()
        ).sum()
        if ul_target_tokens > 0:
            ul_loss /= ul_target_tokens
        loss = mle_loss + self.args.ul_alpha * ul_loss
        #print("ul_loss", ul_loss)
        return (loss, outputs) if return_outputs else loss

    def _get_weighted_train_sampler(self):
        print("This is where we use resampling.")
        return WeightedRandomSampler(
            num_samples=len(self.train_dataset),
            weights=self.train_dataset["weight"],
            replacement=True) 

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if not self.args.use_resampling:
            train_sampler = self._get_train_sampler()
        else:
            train_sampler = self._get_weighted_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )