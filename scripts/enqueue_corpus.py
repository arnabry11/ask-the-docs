import argparse

from app.ingestion.queue import enqueue_corpus, get_ingestion_queue


def main() -> None:
    parser = argparse.ArgumentParser(description="Enqueue the public documentation for ingestion")
    parser.add_argument("--document-id", help="Enqueue one catalog document instead of all pages")
    args = parser.parse_args()

    jobs = enqueue_corpus(get_ingestion_queue(), args.document_id)
    for job in jobs:
        print(f"{job.document_id}: {job.job_id}")
    print(f"Enqueued {len(jobs)} document(s)")


if __name__ == "__main__":
    main()
