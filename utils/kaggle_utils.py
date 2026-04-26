"""Kaggle API utilities for data download and submission."""

import os
import zipfile


def setup_kaggle(username: str, api_token: str):
    """Set Kaggle credentials as environment variables."""
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_API_TOKEN'] = api_token


def download_competition_data(competition: str, output_dir: str):
    """Download competition data from Kaggle."""
    import kaggle

    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading data from {competition}...")
    kaggle.api.competition_download_files(competition, path=output_dir)

    # Unzip
    zip_path = os.path.join(output_dir, f"{competition}.zip")
    if os.path.exists(zip_path):
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(zip_path)
        print("Done!")


def submit_to_kaggle(competition: str, submission_file: str, message: str = "Submission"):
    """Submit predictions to Kaggle competition."""
    import kaggle

    print(f"Submitting {submission_file} to {competition}...")
    kaggle.api.competition_submit(
        file_name=submission_file,
        message=message,
        competition=competition
    )
    print("Submission complete!")


def list_submissions(competition: str):
    """List recent submissions for a competition."""
    import kaggle
    kaggle.api.competition_submissions_cli(competition)
