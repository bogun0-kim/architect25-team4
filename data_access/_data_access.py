import os


class DataAccess:
    def __init__(self, root_path):
        self.root_path = root_path

    def get(self, *path, keywords: dict = None):
        import json
        data_path = self.root_path
        for p in path:
            data_path = os.path.join(data_path, p)
        file_paths = []
        if os.path.isfile(data_path):
            file_paths = [data_path]
        elif os.path.isdir(data_path):
            file_paths = [
                p for f in sorted(os.listdir(data_path))
                if os.path.isfile(p := os.path.join(data_path, f))]
        outputs = []
        for raw_data in file_paths:
            with open(raw_data, 'r', encoding='utf8') as f:
                data = json.load(f)
            _expected = True
            if keywords is not None:
                for k, v in keywords.items():
                    if k in data and v not in data[k]:
                        _expected = False
            if _expected:
                outputs.append(data)
        return outputs


# Singleton instance
_DATA_ACCESS: DataAccess


def create_data_access(root_path):
    global _DATA_ACCESS
    _DATA_ACCESS = DataAccess(root_path)


def get_data_access():
    global _DATA_ACCESS
    return _DATA_ACCESS


__all__ = [
    "create_data_access",
    "get_data_access",
]
