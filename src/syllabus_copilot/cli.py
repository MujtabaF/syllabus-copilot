import argparse
from datetime import date
from pathlib import Path

from .extract import extract_syllabus
from .pdf import pdf_to_text


def main() -> None:
    parser = argparse.ArgumentParser(prog="syllabus-copilot")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Extract deadlines from a syllabus PDF")
    extract.add_argument("pdf", type=Path)
    extract.add_argument("--year", type=int, default=date.today().year)

    args = parser.parse_args()

    if args.command == "extract":
        syllabus = extract_syllabus(pdf_to_text(args.pdf), year=args.year)
        print(syllabus.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
