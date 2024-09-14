from os import path, getcwd

# Define paths for data and logs
_path = path.dirname(getcwd())
RESUMES_PATH = path.join(_path, "data", "resumes")
LOGS_PATH = path.join(_path, "logs")

resume = (lambda r: f"{path.join(RESUMES_PATH, r)}")
