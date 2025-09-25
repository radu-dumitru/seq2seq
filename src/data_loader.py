import zipfile

class DataLoader:
    def __init__(self):
        self.zip_path = "data/dialogs.zip"
        self.inner_name = "dialogs.txt"

    def load_data(self, max_number_lines):
        data = []
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            with zf.open(self.inner_name, "r") as f:
                for i, raw in enumerate(f):
                    if i == max_number_lines:
                        break
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                    if not line:
                        continue
                    q, a = line.split("\t", 1)
                    data.append((q, a))

        print(f"Loaded {len(data)} dialog pairs")
        return data
