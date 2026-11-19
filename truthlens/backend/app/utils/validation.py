import os, re
ALLOWED = {
    "image": {"jpg", "jpeg", "png", "webp", "tiff", "bmp"},
    "video": {"mp4", "mov", "webm", "avi"},
    "audio": {"wav", "mp3", "m4a", "flac"},
    "text": {"txt", "md", "docx", "pdf"},
}

def safe_name(name: str) -> str:
    base = os.path.basename((name or "").replace("\\", "/"))
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip(".")
    return base[:120] or "file"

def extension(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""

def check_extension(kind: str, name: str) -> None:
    if extension(name) not in ALLOWED[kind]:
        raise ValueError(f"'{name}' is not a supported {kind} format. Allowed: {', '.join(sorted(ALLOWED[kind]))}")
