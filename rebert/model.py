import os

import torch
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification

# Embedding variants used in the paper and its ablations.
#   full      : word + sequential position + tree position   (paper default)
#   tree_only : word + tree position
#   no_tree   : word + sequential position                   (ablation)
VARIANTS = {
    'full': dict(use_seq_pos=True, use_tree_pos=True),
    'tree_only': dict(use_seq_pos=False, use_tree_pos=True),
    'no_tree': dict(use_seq_pos=True, use_tree_pos=False),
}


class ReBERT(nn.Module):
    """BERT pair classifier: do two bits belong to the same word?"""

    def __init__(self, variant='full', pretrained='bert-base-uncased'):
        super().__init__()
        cfg = VARIANTS[variant]
        self.variant = variant
        self.use_seq_pos = cfg['use_seq_pos']
        self.use_tree_pos = cfg['use_tree_pos']

        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        # Extend the tokenizer's vocabulary with netlist tokens
        self.tokenizer.add_tokens(['NAND', 'X'])
        self.model = BertForSequenceClassification.from_pretrained(pretrained)
        # Resize the model's token embeddings to accommodate the new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, text1, text2, pe, labels=None, output_attentions=False):
        device = self.model.device
        inputs = self.tokenizer(text1, text2, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        max_len = input_ids.shape[1]

        # Create input embeddings by combining token embeddings and positional encodings
        inputs_embeds = self.model.bert.embeddings.word_embeddings(input_ids)
        if self.use_seq_pos:
            position_ids = torch.arange(max_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            inputs_embeds = inputs_embeds + self.model.bert.embeddings.position_embeddings(position_ids)
        if self.use_tree_pos:
            inputs_embeds = inputs_embeds + pe[:, :max_len, :].to(device)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs['attention_mask'].to(device),
            token_type_ids=inputs['token_type_ids'].to(device),
            output_attentions=output_attentions,
        )

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(outputs.logits, labels.to(device))

        result = {'loss': loss, 'logits': outputs.logits}
        if output_attentions:
            result['attentions'] = outputs.attentions
            result['tokens'] = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        return result


def load_checkpoint(model, ckpt_dir):
    """Load weights saved by `Trainer.save_model` (safetensors or .bin) into `model`."""
    st_path = os.path.join(ckpt_dir, 'model.safetensors')
    if os.path.exists(st_path):
        from safetensors.torch import load_file
        state_dict = load_file(st_path)
    else:
        state_dict = torch.load(os.path.join(ckpt_dir, 'pytorch_model.bin'), map_location='cpu')
    model.load_state_dict(state_dict)
    return model
