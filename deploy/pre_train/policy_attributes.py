import torch

def inspect_pt_file(pt_path):
    """Print model architecture, state dict, and input/output info."""
    try:
        # Load the model (map_location allows loading on CPU if trained on GPU)
        model = torch.jit.load(pt_path, map_location='cpu')

        print("\n" + "="*75)
        print(f"Inspecting: {pt_path}")
        print("="*75 + "\n")

        # 0. Model Structure
        print("\n=== Model Structure ===")
        print(model)

        # 1. Print model code (if it's a TorchScript model)
        print("\n=== Model Code ===")
        print(model.code)

        # 2. Print model graph (if TorchScript)
        print("\n=== Model Graph ===")
        print(model.graph)

        # 3. Print input/output example (if available)
        if hasattr(model, "example_inputs"):
            print("\n=== Example Inputs ===")
            print(model.example_inputs)

        # 4. Print state_dict keys (if available)
        print("\n=== State Dict Keys ===")
        state_dict = model.state_dict()
        for key in state_dict.keys():
            print(key, state_dict[key].shape)

        # 5. Check expected input/output shapes
        print("\n=== Expected Input Shape ===")
        dummy_input = torch.randn(1, 123)  # Adjust based on your model
        try:
            output = model(dummy_input)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
        except RuntimeError as e:
            print(f"Input shape test failed (adjust dummy_input): {e}")

    except Exception as e:
        print(f"Error loading {pt_path}: {e}")

################################################################################

if __name__ == "__main__":
    
    policy_paths = ["g1/motion.pt", "g1/policy_0.85.pt", "g1/policy.pt"]
    # policy_paths = ["g1/motion.pt"]

    for policy_path in policy_paths:

        inspect_pt_file(policy_path)
        print("\n" * 3)