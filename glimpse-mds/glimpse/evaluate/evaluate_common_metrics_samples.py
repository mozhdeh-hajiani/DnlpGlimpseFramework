import argparse
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize the model name to be used as a folder name.
    @param model_name: The model name
    @return: The sanitized model name
    
    """
    return model_name.replace("/", "_")


# logging.basicConfig(stream=stdout, level=logging.)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", type=Path, default="")

    args = parser.parse_args()
    return args



def parse_summaries(path: Path):
    """
    :return: a pandas dataframe with at least the columns 'text' and 'summary'
    """
    # read csv file

    df = pd.read_csv(path).dropna()

    # check if the csv file has the correct columns
    if not all([col in df.columns for col in ["gold", "summary"]]):
        raise ValueError("The csv file must have the columns 'text' and 'summary'.")

    return df


def evaluate_rouge(
    df,
):
    # make a list of the tuples (text, summary)

    texts = df.gold.tolist()
    summaries = df.summary.tolist()

    # rouges
    metrics = {"rouge1": [], "rouge2": [], "rougeL": [], "rougeLsum": []}

    rouges = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )

    metrics["rouge1"].extend(
        [
            rouges.score(summary, text)["rouge1"].fmeasure
            for summary, text in zip(summaries, texts)
        ]
    )
    metrics["rouge2"].extend(
        [
            rouges.score(summary, text)["rouge2"].fmeasure
            for summary, text in zip(summaries, texts)
        ]
    )
    metrics["rougeL"].extend(
        [
            rouges.score(summary, text)["rougeL"].fmeasure
            for summary, text in zip(summaries, texts)
        ]
    )
    metrics["rougeLsum"].extend(
        [
            rouges.score(summary, text)["rougeLsum"].fmeasure
            for summary, text in zip(summaries, texts)
        ]
    )

    # compute the mean of the metrics
    # metrics = {k: sum(v) / len(v) for k, v in metrics.items()}

    return metrics


def main():
    args = parse_args()

    # load the model
    df = parse_summaries(args.summaries)

    metrics = evaluate_rouge(df)


    # # add index to the metrics
    # metrics["index"] = [i for i in range(len(df))]

    df = pd.DataFrame.from_dict(metrics)
    df = df.add_prefix(f"common/")

    # merge the metrics with the summaries

    if args.summaries.exists():
        df_old = parse_summaries(args.summaries)

        for col in df.columns:
            if col not in df_old.columns:
                df_old[col] = float("nan")

        # add entry to the dataframe
        for col in df.columns:
            df_old[col] = df[col]

        df = df_old

    df.to_csv(args.summaries, index=False)


if __name__ == "__main__":
    main()