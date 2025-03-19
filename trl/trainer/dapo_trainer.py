import torch
from typing import Any, Optional, Union
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from .grpo_trainer import GRPOTrainer, RewardFunc
from .dapo_config import DAPOConfig


class DAPOTrainer(GRPOTrainer):
    """
    Trainer for the Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) method.
    This algorithm extends GRPO with four key techniques:
    1. Clip-Higher: Promotes diversity and avoids entropy collapse
    2. Dynamic Sampling: Improves training efficiency and stability
    3. Token-Level Policy Gradient Loss: Critical in long-CoT RL scenarios
    4. Overlong Reward Shaping: Reduces reward noise and stabilizes training

    Example:
    ```python
    from datasets import load_dataset
    from trl import DAPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters
        return [float(len(set(completion))) for completion in completions]

    trainer = DAPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```
    """

    _tag_names = ["trl", "dapo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[DAPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        if args is None:
            args = DAPOConfig()
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        self.dynamic_sampling_buffer = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Computes the DAPO loss which includes:
        1. Token-level policy gradient loss
        2. Clip-Higher strategy with separate εlow and εhigh parameters
        """
        # Get logprobs and values
        logprobs = self._get_per_token_logps(
            model,
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["logits_to_keep"],
        )
        old_logprobs = inputs["old_logprobs"]

        # Compute importance sampling ratios
        ratios = torch.exp(logprobs - old_logprobs)

        # Apply Clip-Higher strategy with separate εlow and εhigh
        advantages = inputs["advantages"]
        clipped_ratios = torch.clamp(
            ratios,
            1.0 - self.args.epsilon_low,
            1.0 + self.args.epsilon_high,
        )
        policy_loss_1 = ratios * advantages
        policy_loss_2 = clipped_ratios * advantages
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # Return loss and outputs
        outputs = {"loss": policy_loss}
        return (policy_loss, outputs) if return_outputs else policy_loss

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Extends GRPO's generation and scoring with:
        1. Dynamic Sampling filtering
        2. Overlong Reward Shaping
        """
        # Generate completions using parent method
        outputs = super()._generate_and_score_completions(inputs)

        # Apply Overlong Reward Shaping
        completion_lengths = outputs["completion_lengths"]
        max_len = self.args.max_length
        cache_len = self.args.soft_punish_cache
        length_rewards = torch.zeros_like(outputs["rewards"])

        # Calculate length-based penalties
        for i, length in enumerate(completion_lengths):
            if length <= max_len - cache_len:
                continue
            elif length <= max_len:
                penalty = (max_len - cache_len - length) / cache_len
                length_rewards[i] = penalty
            else:
                length_rewards[i] = -1.0

        # Combine rewards
        outputs["rewards"] = outputs["rewards"] + length_rewards

        # Apply Dynamic Sampling filtering
        rewards = outputs["rewards"]
        accuracies = (rewards > 0).float()  # Assuming binary rewards for accuracy
        mask = (accuracies > self.args.dynamic_sampling_min_accuracy) & (accuracies < self.args.dynamic_sampling_max_accuracy)
        
        # Update outputs with filtered data
        for key in outputs:
            if torch.is_tensor(outputs[key]):
                outputs[key] = outputs[key][mask]

        # Add to dynamic sampling buffer
        self.dynamic_sampling_buffer.extend([{k: v[i] for k, v in outputs.items()} for i in range(len(outputs["rewards"]))])
        
        # If buffer is full, use it and clear
        if len(self.dynamic_sampling_buffer) >= self.args.dynamic_sampling_buffer_size:
            outputs = {
                k: torch.stack([item[k] for item in self.dynamic_sampling_buffer[:self.args.dynamic_sampling_buffer_size]])
                for k in self.dynamic_sampling_buffer[0].keys()
            }
            self.dynamic_sampling_buffer = []
            return outputs

        # Return empty outputs if buffer not full
        return {k: v[:0] for k, v in outputs.items()} 