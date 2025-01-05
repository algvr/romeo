import io
import pickle
import zipfile


def read_pkl_from_zip(path):
    if path.lower().endswith(".zip"):
        zip_obj = zipfile.ZipFile(path)
        pkl_bytes = zip_obj.read(zip_obj.namelist()[0])
    else:
        with open(path, "rb") as f:
            pkl_bytes = f.read()
    
    ret = pickle.load(io.BytesIO(pkl_bytes))
    return ret
