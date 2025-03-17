import argparse
from generator import LLaVa_NeXT_Video_generator
# To-Do: Add more generators in the future and import them here

def main():
    parser = argparse.ArgumentParser(
        description="Select which generator to run and specify a config file."
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="llava-next-video",
        help="Name of the generator to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration JSON file"
    )
    args = parser.parse_args()

    # Map generator names to their corresponding functions.
    generators = {
        "llava-next-video": LLaVa_NeXT_Video_generator,
        # To-Do: "other": OtherModel_generator,
    }

    gen_key = args.generator.lower()
    if gen_key in generators:
        # Pass the config file path to the generator function.
        csv_filename = generators[gen_key](args.config)
        print(f"Generator '{gen_key}' finished. Results saved to: {csv_filename}")
    else:
        print(f"Unknown generator '{args.generator}'. Available options: {', '.join(generators.keys())}")

if __name__ == "__main__":
    main()