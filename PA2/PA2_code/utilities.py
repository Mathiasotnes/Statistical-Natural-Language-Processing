import matplotlib.pyplot as plt
import torch
from transformer import MultiHeadAttention
from disentangled_attention import DisentangledSelfAttention

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size, show_plots=False, save_plots=False):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)
        
        # Use hooks to get the attention maps
        attn_maps = []
        def get_attention_maps(module, input, output):
            attn_map = module.attn_weights.detach().cpu()
            # Drop first dimension if present
            if attn_map.shape[0] == 1:
                attn_map = attn_map.squeeze(0)
            attn_maps.append(attn_map)
        
        # Register hooks on all attention modules
        hooks = []
        for module in self.model.modules():
            if isinstance(module, MultiHeadAttention) or isinstance(module, DisentangledSelfAttention):
                hook = module.register_forward_hook(get_attention_maps)
                hooks.append(hook)

        # Process the input tensor through the model
        _ = self.model(input_tensor)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Display the number of attention maps
        print("Number of attention maps:", len(attn_maps))

        # Visualize and save the attention maps for the first head
        for j, attn_map in enumerate(attn_maps):
            # Reshape attn_map to (batch_size, num_heads, T, T)
            B_num_heads, T, _ = attn_map.shape # (1, 2, 32, 32) - (1, 2, 32, 32)
            batch_size = 1  # Adjust if necessary
            num_heads = self.model.num_heads  # Ensure this is correctly set

            attn_map = attn_map.view(batch_size, num_heads, T, T)

            # Get the attention map for the first head
            att_map = attn_map[0, 0].detach().cpu().numpy()  # Shape: (T, T)

            # Check if the attention probabilities sum to 1 over rows
            total_prob_over_rows = attn_map[0, 0].sum(dim=1)
            if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                print("Total probability over rows:", total_prob_over_rows.numpy())

            # Create a heatmap of the attention map
            fig, ax = plt.subplots()
            cax = ax.imshow(att_map, cmap='hot', interpolation='nearest')
            ax.xaxis.tick_top()
            fig.colorbar(cax, ax=ax)  
            plt.title(f"Attention Map {j + 1}")

            # Save the plot
            if save_plots:
                plt.savefig(f"./plots/attention_map_{j + 1}.png")

            # Show the plot
            if show_plots:
                plt.show()

                


