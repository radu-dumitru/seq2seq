class DataLoader:
    def __init__(self):
        self.filepath = "data/dialogs.txt"

    def load_data(self, max_number_lines):
        data = []
        with open(self.filepath, "r") as f:
            for i, line in enumerate(f):
                if i == max_number_lines:
                    break

                q, a = line.strip().split("\t")
                data.append((q, a))

        print(f"Loaded {len(data)} dialog pairs")
        return data
