import os

def scaffold():
    folders = [
        "backend/data/raw",
        "backend/data/processed",
        "backend/utils"
    ]

    for f in folders:
        os.makedirs(f, exist_ok=True)
        print(f"Created directory: {f}")

if __name__ == "__main__":
    scaffold()
