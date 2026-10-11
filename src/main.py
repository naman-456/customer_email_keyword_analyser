import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


REQUIRED_COLUMNS = {
    "email_id",
    "customer_id",
    "customer_name",
    "subject",
    "body",
    "received_at",
}
TOKENIZER = RegexpTokenizer(r"[a-z]+")
EMAIL_STOPWORDS = {
    "hello",
    "hi",
    "hey",
    "regards",
    "sincerely",
    "thanks",
    "thank",
    "dear",
    "best",
}


def clean_text(text):
    """Normalize email text and remove HTML tags and URLs."""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def get_stopwords():
    try:
        return set(stopwords.words("english")) | EMAIL_STOPWORDS
    except LookupError as error:
        raise RuntimeError(
            "NLTK stopwords are missing. Install them with "
            "`python -m nltk.downloader stopwords`."
        ) from error


def extract_keywords(text, stop_words):
    cleaned = clean_text(text)
    return [
        token
        for token in TOKENIZER.tokenize(cleaned)
        if token not in stop_words and len(token) > 1
    ]


def _display_counts(counts, limit):
    return ", ".join(
        f"{word} ({count})"
        for word, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :limit
        ]
    )


def analyze_emails(emails, stop_words, top_n=5):
    """Return per-email analysis, overall keyword frequencies, and CRM topics."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    missing_columns = REQUIRED_COLUMNS.difference(emails.columns)
    if missing_columns:
        raise ValueError(
            "Email dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    keyword_totals = Counter()
    customer_totals = defaultdict(Counter)
    customer_email_counts = Counter()
    customer_names = {}
    email_rows = []

    for row in emails.itertuples(index=False):
        record = row._asdict()
        subject = "" if pd.isna(record["subject"]) else str(record["subject"])
        body = "" if pd.isna(record["body"]) else str(record["body"])
        tokens = extract_keywords(f"{subject} {body}", stop_words)
        counts = Counter(tokens)
        keyword_totals.update(counts)

        customer_id = str(record["customer_id"])
        customer_name = str(record["customer_name"])
        customer_names.setdefault(customer_id, customer_name)
        customer_totals[customer_id].update(counts)
        customer_email_counts[customer_id] += 1

        email_rows.append(
            {
                "email_id": record["email_id"],
                "customer_id": record["customer_id"],
                "customer_name": record["customer_name"],
                "received_at": record["received_at"],
                "cleaned_text": clean_text(f"{subject} {body}"),
                "keywords": _display_counts(counts, top_n),
            }
        )

    keyword_rows = [
        {"keyword": word, "frequency": count}
        for word, count in sorted(
            keyword_totals.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    crm_rows = [
        {
            "customer_id": customer_id,
            "customer_name": customer_names[customer_id],
            "email_count": customer_email_counts[customer_id],
            "topic_summary": _display_counts(customer_totals[customer_id], top_n),
        }
        for customer_id in sorted(customer_totals)
    ]

    return (
        pd.DataFrame(
            email_rows,
            columns=[
                "email_id",
                "customer_id",
                "customer_name",
                "received_at",
                "cleaned_text",
                "keywords",
            ],
        ),
        pd.DataFrame(keyword_rows, columns=["keyword", "frequency"]),
        pd.DataFrame(
            crm_rows,
            columns=[
                "customer_id",
                "customer_name",
                "email_count",
                "topic_summary",
            ],
        ),
    )


def run_analysis(input_path, output_dir, top_n=5):
    emails = pd.read_csv(input_path)
    results = analyze_emails(emails, get_stopwords(), top_n)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = (
        "email_analysis.csv",
        "keyword_frequency.csv",
        "crm_topic_report.csv",
    )
    for result, filename in zip(results, filenames):
        result.to_csv(output_dir / filename, index=False)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract recurring topics from customer email CSV data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "customer_emails.csv",
        help="Path to the customer email CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where analysis CSV reports will be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Maximum keywords shown in each email and customer summary.",
    )
    args = parser.parse_args()

    run_analysis(args.input, args.output_dir, args.top_n)
    print(f"Analysis complete. Reports saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
