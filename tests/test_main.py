import unittest

import pandas as pd

from src.main import analyze_emails, clean_text, extract_keywords


class EmailAnalysisTests(unittest.TestCase):
    def test_clean_text_removes_html_urls_and_punctuation(self):
        self.assertEqual(
            clean_text("<p>Order delayed!</p> Visit https://example.com now."),
            "order delayed visit now",
        )

    def test_extract_keywords_removes_stopwords(self):
        self.assertEqual(
            extract_keywords("The package is delayed; tracking updated.", {"the", "is"}),
            ["package", "delayed", "tracking", "updated"],
        )

    def test_reports_summarize_customer_and_global_keywords(self):
        emails = pd.DataFrame(
            [
                {
                    "email_id": "E1",
                    "customer_id": "C1",
                    "customer_name": "Ava",
                    "subject": "Late package",
                    "body": "Package delivery delayed",
                    "received_at": "2026-09-01",
                },
                {
                    "email_id": "E2",
                    "customer_id": "C1",
                    "customer_name": "Ava",
                    "subject": "Delivery update",
                    "body": "Package still delayed",
                    "received_at": "2026-09-02",
                },
            ]
        )

        email_report, keyword_report, crm_report = analyze_emails(
            emails, stop_words=set(), top_n=3
        )

        self.assertEqual(len(email_report), 2)
        self.assertEqual(
            keyword_report.iloc[0].to_dict(), {"keyword": "package", "frequency": 3}
        )
        self.assertEqual(crm_report.iloc[0]["email_count"], 2)
        self.assertIn("package (3)", crm_report.iloc[0]["topic_summary"])


if __name__ == "__main__":
    unittest.main()
