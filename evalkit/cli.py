import argparse
import json
import sys

from .ci import check_report_with_rules, ci_output
from .comparator import ReportComparator
from .config import load_config, merge_thresholds
from .report import EvaluationReport


def load_json(filepath):
    with open(filepath, "r") as file:
        return json.load(file)


def load_report(filepath):
    data = load_json(filepath)

    if isinstance(data, list):
        if not data:
            raise ValueError(f"Report file is empty: {filepath}")
        data = data[-1]

    return EvaluationReport(
        model_name=data["model"],
        model_version=data.get("model_version"),
        timestamp=data.get("timestamp"),
        run_id=data.get("run_id"),
        metrics=data["metrics"],
        diagnostics=data.get("diagnostics", {}),
        data_profile=data.get("data_profile", {}),
        performance=data.get("performance", {}),
        previous_report=data.get("previous_report"),
        drift=data.get("drift"),
        model_metadata=data.get("model_metadata", {}),
        dataset_metadata=data.get("dataset_metadata", {}),
        feature_monitoring=data.get("feature_monitoring", {}),
        prediction_distribution=data.get("prediction_distribution", {}),
        explainability=data.get("explainability", {}),
        rule_results=data.get("rule_results", {}),
    )


def parse_thresholds(values):
    thresholds = {}

    for value in values:
        raw_value = str(value).strip()
        try:
            metric, threshold = raw_value.split("=", 1)
            metric = metric.strip()
            threshold = threshold.strip()
            if not metric or not threshold:
                raise ValueError
            thresholds[metric] = float(threshold)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "thresholds must use METRIC=VALUE format, for example: accuracy=0.80"
            ) from error

    return thresholds


def add_threshold_argument(parser):
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Allowed metric change; may be repeated",
    )


def print_comparison(results):
    for metric, result in results.items():
        if metric == "_model_version":
            continue

        print(
            f"{metric}: "
            f"{result['previous']:.4f} -> "
            f"{result['current']:.4f} "
            f"({result['status']})"
        )

    version = results.get("_model_version")
    if version:
        print(
            f"model_version: {version['previous']} -> "
            f"{version['current']} "
            f"({'changed' if version['changed'] else 'unchanged'})"
        )


def command_check(args):
    config = load_config(args.config)
    report = load_report(args.report)

    if args.previous:
        report.previous_report = load_json(args.previous)
        if isinstance(report.previous_report, list):
            report.previous_report = report.previous_report[-1]

    thresholds = merge_thresholds(config, parse_thresholds(args.threshold))
    passed = check_report_with_rules(
        report,
        thresholds=thresholds,
        rules=config.get("rules", {}),
    )
    print(ci_output(report, passed, args.output))

    return 0 if passed else 1


def command_compare(args):
    config = load_config(args.config)
    previous = load_json(args.previous)
    current = load_json(args.current)

    if isinstance(previous, list):
        previous = previous[-1]
    if isinstance(current, list):
        current = current[-1]

    results = ReportComparator(previous, current).compare(
        thresholds=merge_thresholds(config, parse_thresholds(args.threshold))
    )
    print_comparison(results)
    return 0


def command_history(args):
    config = load_config(args.config)
    data = load_json(args.file or config["history_file"])
    runs = data if isinstance(data, list) else [data]

    if args.limit:
        runs = runs[-args.limit:]

    start_index = len(data) - len(runs) + 1 if isinstance(data, list) else 1
    for index, run in enumerate(runs, start=start_index):
        print(
            f"{index}: {run.get('model')} "
            f"{run.get('model_version', 'unknown')} "
            f"{run.get('timestamp', 'unknown')}"
        )

    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="evalkit",
        description="ML model evaluation toolkit",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Check a report for regression")
    check.add_argument("--report", required=True)
    check.add_argument("--previous")
    check.add_argument("--config", default="evalkit.yaml")
    check.add_argument(
        "--output",
        choices=["text", "json", "github"],
        default="text",
    )
    add_threshold_argument(check)
    check.set_defaults(handler=command_check)

    compare = commands.add_parser("compare", help="Compare two reports")
    compare.add_argument("--current", required=True)
    compare.add_argument("--previous", required=True)
    compare.add_argument("--config", default="evalkit.yaml")
    add_threshold_argument(compare)
    compare.set_defaults(handler=command_compare)

    history = commands.add_parser("history", help="List saved evaluation reports")
    history.add_argument("--file")
    history.add_argument("--config", default="evalkit.yaml")
    history.add_argument("--limit", type=int)
    history.set_defaults(handler=command_history)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        return args.handler(args)
    except (FileNotFoundError, KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())