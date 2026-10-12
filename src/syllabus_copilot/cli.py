import argparse
import json
from datetime import date, datetime
from pathlib import Path

from .config import settings
from .evals import PROMPT_VERSION, format_table, load_golden, run_eval, unlabeled
from .extract import extract_syllabus
from .pdf import pdf_to_text


def main() -> None:
    parser = argparse.ArgumentParser(prog="syllabus-copilot")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Extract deadlines from a syllabus PDF")
    extract.add_argument("pdf", type=Path)
    extract.add_argument("--year", type=int, default=date.today().year)

    evaluate = sub.add_parser("eval", help="Score extraction against the golden dataset")
    evaluate.add_argument(
        "--model",
        action="append",
        dest="models",
        help="LiteLLM model string; repeat to compare models (default: LLM_MODEL)",
    )
    evaluate.add_argument("--golden", type=Path, default=Path("evals/golden"))
    evaluate.add_argument("--out", type=Path, default=Path("evals/results"))

    args = parser.parse_args()

    if args.command == "extract":
        syllabus = extract_syllabus(pdf_to_text(args.pdf), year=args.year)
        print(syllabus.model_dump_json(indent=2))
    elif args.command == "eval":
        run_evals(args.models or [settings.llm_model], args.golden, args.out)


def run_evals(models: list[str], golden: Path, out: Path) -> None:
    examples = load_golden(golden)
    print(f"{len(examples)} golden syllabi, prompt version {PROMPT_VERSION}")
    if todo := unlabeled(golden):
        print(f"Skipping {len(todo)} not labeled yet: {', '.join(todo)}")
    print()
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    summaries = []
    for model in models:
        print(f"== {model}")

        def report(r):
            status = f"ERROR {r.error}" if r.error else "ok"
            print(f"  {r.id}: {r.counts.matched}/{r.counts.expected} found, {status}")

        result = run_eval(examples, model, on_example=report)
        summaries.append(result.summary())
        path = out / f"{stamp}_{model.replace('/', '_')}_{PROMPT_VERSION}.json"
        path.write_text(json.dumps(result.to_json(), indent=2, default=str))
        print(f"  saved {path}\n")

    print(format_table(summaries))


if __name__ == "__main__":
    main()
