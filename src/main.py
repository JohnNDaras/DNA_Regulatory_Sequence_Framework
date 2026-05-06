import argparse
from src.utils.config_utils import load_yaml
from src.utils.seed import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    set_seed(config.get("seed", 42))
    print(f"Loaded config: {args.config}")

if __name__ == "__main__":
    main()
