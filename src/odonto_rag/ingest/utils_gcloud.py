# ----------------- Google token -----------------
def gcloud_token() -> str:
    tok = os.popen("gcloud auth print-access-token").read().strip()
    if not tok:
        raise RuntimeError("No gcloud access token. Run: gcloud auth login")
    return tok


