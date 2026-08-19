from .seed_database import seed_database, parse_args

def main() -> None:
    args = parse_args()
    seed_database(rebuild=args.rebuild, from_cache=args.from_cache)
