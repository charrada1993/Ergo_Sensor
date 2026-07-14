import requests
import json
import base64
import time

class FirebaseStorage:
    def __init__(self, storage_url):
        self.url = storage_url.rstrip('/')

    def _get_url(self, path):
        return f"{self.url}/{path.lstrip('/')}.json"

    def put_file(self, path, filename, file_bytes):
        try:
            b64_string = base64.b64encode(file_bytes).decode('utf-8')
            file_key = filename.replace('.', '_')
            target_url = self._get_url(f"{path}/{file_key}")
            payload = {
                'filename': filename,
                'data': b64_string,
                'timestamp': time.time()
            }
            res = requests.put(target_url, json=payload, timeout=15)
            if res.status_code == 200:
                print(f"[FirebaseStorage] Successfully uploaded {filename} to {path}/{file_key}")
                return True
            else:
                print(f"[FirebaseStorage] Failed to upload {filename}: status {res.status_code}, response: {res.text}")
        except Exception as e:
            print(f"[FirebaseStorage] Error uploading {filename}: {e}")
        return False

    def delete_file(self, path, filename):
        try:
            file_key = filename.replace('.', '_')
            target_url = self._get_url(f"{path}/{file_key}")
            res = requests.delete(target_url, timeout=15)
            if res.status_code == 200:
                print(f"[FirebaseStorage] Successfully deleted {filename} from {path}/{file_key}")
                return True
            else:
                print(f"[FirebaseStorage] Failed to delete {filename}: status {res.status_code}")
        except Exception as e:
            print(f"[FirebaseStorage] Error deleting {filename}: {e}")
        return False

    def get_file(self, path, filename):
        try:
            file_key = filename.replace('.', '_')
            target_url = self._get_url(f"{path}/{file_key}")
            res = requests.get(target_url, timeout=15)
            if res.status_code == 200:
                data = res.json()
                if data and 'data' in data:
                    return base64.b64decode(data['data'])
            else:
                print(f"[FirebaseStorage] Failed to get {filename}: status {res.status_code}")
        except Exception as e:
            print(f"[FirebaseStorage] Error getting {filename}: {e}")
        return None

    def list_files(self, path):
        try:
            target_url = self._get_url(path)
            res = requests.get(target_url, timeout=15)
            if res.status_code == 200:
                data = res.json()
                if data and isinstance(data, dict):
                    files = [v['filename'] for v in data.values() if isinstance(v, dict) and 'filename' in v]
                    files.sort(reverse=True)
                    return files
        except Exception as e:
            print(f"[FirebaseStorage] Error listing files under {path}: {e}")
        return []

    def get_files_metadata(self, path):
        try:
            target_url = self._get_url(path)
            res = requests.get(target_url, timeout=15)
            if res.status_code == 200:
                data = res.json()
                result = {}
                if data and isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, dict) and 'filename' in v:
                            size = 0
                            if 'data' in v:
                                # approximate size from base64 string
                                size = len(v['data']) * 3 // 4
                            result[v['filename']] = {
                                'timestamp': v.get('timestamp', time.time()),
                                'size': size
                            }
                return result
        except Exception as e:
            print(f"[FirebaseStorage] Error getting files metadata under {path}: {e}")
        return {}
