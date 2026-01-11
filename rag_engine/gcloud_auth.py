import subprocess

def gcloud_token() -> str:
    return subprocess.check_output(
        ["gcloud", "auth", "print-access-token"],
        text=True
    ).strip()

