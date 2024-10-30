import matplotlib.pyplot as plt
import torch
from transformer import MultiHeadAttention

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
            attn_maps.append(module.attn_weights.detach().cpu())
        
        # Register hooks on all MultiHeadAttention modules
        hooks = []
        for module in self.model.modules():
            if isinstance(module, MultiHeadAttention):
                hook = module.register_forward_hook(get_attention_maps)
                hooks.append(hook)

        # Process the input tensor through the encoder model
        _ = self.model(input_tensor)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Display the number of attention maps
        print("Number of attention maps:", len(attn_maps))

        # Visualize and save the attention maps for the first head
        for j, attn_map in enumerate(attn_maps):
            # Get the attention map for the first head
            att_map = attn_map.squeeze(0).detach().cpu().numpy()  # Remove batch dimension and convert to NumPy array

            # Check if the attention probabilities sum to 1 over rows
            total_prob_over_rows = torch.sum(attn_map[0], dim=2)
            if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                print("Total probability over rows:", total_prob_over_rows.numpy())

            # Create a heatmap of the attention map
            fig, ax = plt.subplots()
            cax = ax.imshow(att_map[0], cmap='hot', interpolation='nearest')
            ax.xaxis.tick_top()
            fig.colorbar(cax, ax=ax)  
            plt.title(f"Attention Map {j + 1}")
            
            # Save the plot
            if save_plots:
                plt.savefig(f"./plots/attention_map_{j + 1}.png")
            
            # Show the plot
            if show_plots:
                plt.show()
            


