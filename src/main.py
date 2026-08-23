from src.utils.config import get_environment, load_config


def main() -> None:
    config = load_config()

    print("=" * 50)
    print("RazorGuard AI")
    print("=" * 50)
    print(f"Environment : {get_environment()}")
    print(f"Version     : {config['project']['version']}")
    print(f"Risk Low    : {config['risk']['low_threshold']}")
    print(f"Risk High   : {config['risk']['high_threshold']}")
    print("=" * 50)


if __name__ == "__main__":
    main()